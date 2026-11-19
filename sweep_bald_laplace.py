"""
Active-prompting sweep for BALD acquisition variants on a frozen-SAM head.

For each image, SAM's image embedding is computed once and every acquisition
variant runs its own prompting trajectory from the same initial point (GT
centroid, positive).  Labels of acquired points come from the GT mask, as in
active_prompting.py.  Records per step: head IoU (released protocol: head MAP
logits > 0), SAM IoU (SAM logits > 0), distance of each new query to the
nearest existing prompt, and whether the query landed on a pixel the current
head / SAM mask got wrong.

Variants (comma-separated --variants, or the default grid):
  entropy                 released entropy baseline (probit predictive, released NLL posterior)
  entropy_map             entropy of the head's MAP probability (any head)
  ens                     deep-ensemble BALD over --ensemble members (MI across heads)
  bald_released           released diagonal BALD posterior, exact BALD (K -> infinity)
  random
  full:<w>:<n_eff>:<prior> new posterior, full covariance, curvature weighting
                           w in {b, cw} (bernoulli / class_weighted)
  diag:<w>:<n_eff>:<prior> same with a diagonal covariance

Usage:
  python sweep_bald_laplace.py --config config.yaml --run_id v3 --split val \
      --fit_dir <fit dir, class_weighted> --fit_dir <fit dir, bernoulli> \
      --per_group 4 --budget 10 --out out.csv
"""

import argparse
import csv
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image

from active_prompting import (build_exclusion_mask, compute_gt_centroid, resize_gt_mask,
                              scale_points_to_original, binary_entropy, select_random)
from modules.bald_laplace import LastLayerLaplace, bald_from_gaussian_logit, _bregman_softplus
from modules.head_v5 import build_head
from modules.laplace_wrapper import load_laplace
from modules.promptset import build_gaussian_map
from modules.sam_wrapper import load_sam_predictor, run_sam_for_promptset

log = logging.getLogger("sweep")

DEFAULT_GRID = (
    ["entropy", "bald_released", "random"]
    + [f"full:{w}:{n:g}:{p:g}" for w in ("b", "cw") for n in (1e2, 1e3, 1e4, 1e5, 1e6, 1e8) for p in (1.0,)]
    + [f"full:cw:{n:g}:{p:g}" for n in (1e2, 1e3) for p in (1e-2, 1e2)]
    + [f"diag:{w}:{n:g}:1" for w in ("b",) for n in (1e3, 1e5, 1e8)]
)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union > 0 else 1.0


def argmax_outside(acq: np.ndarray, exclusion: np.ndarray):
    a = acq.astype(np.float64, copy=True)
    a[exclusion] = -np.inf
    y, x = np.unravel_index(int(np.argmax(a)), a.shape)
    return int(x), int(y)


@torch.no_grad()
def ensemble_bald(members, x):
    """MI across ensemble heads: mean_m KL(Bern(p_m) || Bern(p_bar)), cancellation-free."""
    f = torch.cat([m(x) for m in members], 1).squeeze(0).float()      # (M, H, W)
    p_bar = torch.sigmoid(f).mean(0)
    q_bar = torch.sigmoid(-f).mean(0)
    b = torch.log(p_bar.clamp_min(1e-38)) - torch.log(q_bar.clamp_min(1e-38))
    return _bregman_softplus(b.unsqueeze(0).expand_as(f), f).mean(0)


class Variant:
    def __init__(self, spec: str, head, la_released, curvatures, device, default_radius=5, members=None):
        self.spec = spec
        self.members = members
        self.head = head
        self.la_rel = la_released
        acq, _, pool = spec.partition("@")
        self.radius = default_radius
        self.margin = 0
        if pool:
            r, _, mg = pool.partition("b")
            self.radius = int(r)
            self.margin = int(mg) if mg else 0
        self.kind = acq.split(":")[0]
        self.la = None
        if self.kind in ("full", "diag"):
            _, w, n_eff, prior = acq.split(":")
            key = {"b": "bernoulli", "cw": "class_weighted"}[w]
            la = LastLayerLaplace(head, device, covariance=self.kind)
            la.H_mean = curvatures[key].to(device, torch.float64)
            la.set_posterior(float(n_eff), float(prior))
            self.la = la

    @torch.no_grad()
    def score(self, feat, map_logits):
        """Return acquisition map (256, 256) numpy float64; None for random."""
        if self.kind == "random":
            return None
        if self.kind == "entropy_map":
            return binary_entropy(torch.sigmoid(map_logits).squeeze()).double().cpu().numpy()
        if self.kind == "ens":
            return ensemble_bald(self.members, self._x).double().cpu().numpy()
        if self.kind == "entropy":
            probs = self.la_rel._probit_predictive(feat, map_logits)
            return binary_entropy(probs.squeeze()).double().cpu().numpy()
        if self.kind == "bald_released":
            phi = self.la_rel._phi(feat).double()
            mu = map_logits.reshape(-1).double()
            var = (phi.pow(2) * self.la_rel.diag_var_bald.double().to(phi.device)).sum(1)
            b, _ = bald_from_gaussian_logit(mu, var, n_nodes=32, dtype=torch.float32)
            return b.reshape(256, 256).double().cpu().numpy()
        b, _ = self.la.bald(feat, n_nodes=32, dtype=torch.float32)
        return b.squeeze(0).double().cpu().numpy()


@torch.no_grad()
def run_trajectory(variant, predictor, head, la_rel, gt, init_xy, H0, W0, budget, sigma, rng, device):
    pos, neg = [list(init_xy)], []
    gt_b = gt > 0
    rows = []
    for step in range(budget):
        pos_o = scale_points_to_original(pos, H0, W0)
        neg_o = scale_points_to_original(neg, H0, W0) if neg else []
        sam = run_sam_for_promptset(predictor, pos_o, neg_o, (H0, W0), multimask_output=True,
                                    target_size=(256, 256))
        lg = sam["logits_256"]
        pm = build_gaussian_map(pos, 256, 256, sigma)
        nm = build_gaussian_map(neg, 256, 256, sigma) if neg else np.zeros((256, 256), np.float32)
        x = torch.from_numpy(np.stack([np.clip(lg, -20, 20), pm, nm], 0)).float().unsqueeze(0).to(device)
        feat = head.backbone(x)
        ml = head.final_conv(feat)
        head_mask = (ml.squeeze().cpu().numpy() > 0)
        sam_mask = lg > 0
        row = {"step": step, "iou_head": iou(head_mask, gt_b), "iou_sam": iou(sam_mask, gt_b)}
        if step < budget - 1:
            pts = pos + neg
            excl = build_exclusion_mask(pts, 256, 256, variant.radius)
            if variant.margin > 0:
                mg = variant.margin
                excl[:mg, :] = True; excl[-mg:, :] = True; excl[:, :mg] = True; excl[:, -mg:] = True
            variant._x = x
            acq = variant.score(feat, ml)
            if acq is None:
                nx, ny = select_random(256, 256, excl, rng)
            else:
                nx, ny = argmax_outside(acq, excl)
            d = min(math.hypot(nx - px, ny - py) for px, py in pts)
            row.update({"qx": nx, "qy": ny, "q_dist": d,
                        "q_err_head": int(head_mask[ny, nx] != gt_b[ny, nx]),
                        "q_err_sam": int(sam_mask[ny, nx] != gt_b[ny, nx])})
            (pos if gt_b[ny, nx] else neg).append([nx, ny])
        rows.append(row)
    return rows


def set_image_cached(predictor, r, cache_dir, device):
    """SamPredictor.set_image with an on-disk cache of the image embedding."""
    if cache_dir:
        f = Path(cache_dir) / f"{r.safe_id}.pt"
        if f.exists():
            c = torch.load(f, map_location=device)
            predictor.reset_image()
            predictor.features = c["features"].to(device)
            predictor.original_size = tuple(c["original_size"])
            predictor.input_size = tuple(c["input_size"])
            predictor.is_image_set = True
            return
    predictor.set_image(np.array(Image.open(r.image_path).convert("RGB")))
    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        torch.save({"features": predictor.features.detach().cpu(),
                    "original_size": predictor.original_size,
                    "input_size": predictor.input_size}, Path(cache_dir) / f"{r.safe_id}.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run_id", default="v3")
    ap.add_argument("--head_checkpoint", default=None)
    ap.add_argument("--released_laplace_dir", default=None, help="'none' to skip (heads without a released fit)")
    ap.add_argument("--ensemble", nargs="*", default=[], help="checkpoint paths of ensemble members (for 'ens')")
    ap.add_argument("--fit_dir", action="append", default=[],
                    help="fit_bald_laplace.py output dir (repeatable; one per curvature weighting)")
    ap.add_argument("--curvature_dir", default=None, help="(legacy) dir with curv_<weighting>.pt files")
    ap.add_argument("--split", default="val")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--groups", nargs="*", default=None)
    ap.add_argument("--max_images", type=int, default=0)
    ap.add_argument("--image_seed", type=int, default=0, help="seed for the image subset")
    ap.add_argument("--budget", type=int, default=10)
    ap.add_argument("--exclusion_radius", type=int, default=5)
    ap.add_argument("--variants", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--emb_cache", default=None, help="Directory for cached SAM image embeddings")
    ap.add_argument("--per_group", type=int, default=0, help="Stratified subset: N images per group")
    ap.add_argument("--reverse", action="store_true", help="process images in reverse order")
    ap.add_argument("--skip_csv", nargs="*", default=[],
                    help="skip images already present in these CSVs (checked before each image)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n_shards", type=int, default=1)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    cfg = yaml.safe_load(open(args.config))
    root = Path(cfg["data"]["project_root"])
    sigma = cfg["prompt_generation"]["gaussian_sigma"]
    dev = args.device

    if args.head_checkpoint:
        ckpt = Path(args.head_checkpoint)
    elif cfg["data"].get("head_checkpoint"):
        ckpt = Path(cfg["data"]["head_checkpoint"])
    else:
        ckpt = root / "checkpoints/no_image" / f"run_{args.run_id}" / "best.pt"
    def load_head(path):
        h = build_head(cfg["head"])
        st = torch.load(path, map_location="cpu")
        h.load_state_dict(st.get("model_state_dict", st))
        return h.eval().to(dev)
    head = load_head(ckpt)
    la_rel = None
    if args.released_laplace_dir != "none":
        if args.released_laplace_dir:
            lap_dir = Path(args.released_laplace_dir)
        elif cfg["data"].get("laplace_dir"):
            lap_dir = Path(cfg["data"]["laplace_dir"])
        else:
            lap_dir = root / "laplace/no_image" / f"run_{args.run_id}"
        la_rel, _ = load_laplace(head, lap_dir, dev)
    members = [load_head(m) for m in args.ensemble]
    if members:
        log.info("ensemble: %d members", len(members))

    curvatures = {}
    for fd in args.fit_dir:
        st_ = torch.load(Path(fd) / "bald_laplace_state.pt", map_location="cpu")
        curvatures[st_["weighting"]] = st_["H_mean"]
    if args.curvature_dir:
        cdir = Path(args.curvature_dir)
        for k in ("bernoulli", "class_weighted"):
            if k not in curvatures and (cdir / f"curv_{k}.pt").exists():
                curvatures[k] = torch.load(cdir / f"curv_{k}.pt", map_location="cpu")["H_mean"]
    specs_ = args.variants.split(",") if args.variants else DEFAULT_GRID
    if not curvatures and any(v.split(":")[0] in ("full", "diag") for v in specs_):
        ap.error("give --fit_dir (or --curvature_dir)")
    log.info("curvatures available: %s", sorted(curvatures))

    specs = args.variants.split(",") if args.variants else DEFAULT_GRID
    variants = [Variant(s, head, la_rel, curvatures, dev, args.exclusion_radius, members) for s in specs]
    log.info("%d variants: %s", len(variants), specs)

    man = pd.read_csv(args.manifest) if args.manifest else pd.read_csv(root / "manifests" / f"{args.split}.csv")
    if args.groups:
        man = man[man["group"].isin(args.groups)]
    if args.per_group:
        man = man.groupby("group", group_keys=False).apply(
            lambda g: g.sample(n=min(args.per_group, len(g)), random_state=args.image_seed))
    if args.max_images and len(man) > args.max_images:
        man = man.sample(n=args.max_images, random_state=args.image_seed)
    man = man.sort_values("sample_id").iloc[args.shard::args.n_shards]
    if args.reverse:
        man = man.iloc[::-1]
    log.info("%d images", len(man))

    predictor = load_sam_predictor(cfg["sam"]["checkpoint"], cfg["sam"]["model_type"], dev)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        prev = pd.read_csv(out, usecols=["sample_id"])
        done = set(prev.sample_id.unique())
    fields = ["sample_id", "group", "variant", "step", "iou_head", "iou_sam",
              "qx", "qy", "q_dist", "q_err_head", "q_err_sam"]
    fh = open(out, "a", newline="")
    wr = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
    if not done:
        wr.writeheader()

    t0 = time.time()
    for i, (_, r) in enumerate(man.iterrows()):
        sid = str(r.sample_id)
        if sid in done:
            continue
        if args.skip_csv:
            taken = False
            for sc in args.skip_csv:
                try:
                    if Path(sc).exists() and sid in set(pd.read_csv(sc, usecols=["sample_id"]).sample_id):
                        taken = True
                        break
                except Exception:
                    pass
            if taken:
                continue
        H0, W0 = int(r.image_h), int(r.image_w)
        set_image_cached(predictor, r, args.emb_cache, dev)
        gt = resize_gt_mask(r.mask_path, H0, W0)
        init = compute_gt_centroid(gt)
        for v in variants:
            rng = np.random.default_rng(args.seed)
            rows = run_trajectory(v, predictor, head, la_rel, gt, init, H0, W0, args.budget, sigma, rng, dev)
            for row in rows:
                row.update({"sample_id": sid, "group": r.group, "variant": v.spec})
            wr.writerows(rows)
        fh.flush()
        if (i + 1) % 10 == 0:
            el = time.time() - t0
            log.info("%d/%d images  %.0fs  (%.1fs/img)", i + 1, len(man), el, el / (i + 1))
    fh.close()
    log.info("done in %.0fs -> %s", time.time() - t0, out)


if __name__ == "__main__":
    main()
