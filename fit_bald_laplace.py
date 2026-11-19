"""
Fit the full-covariance, tempered last-layer Laplace used for BALD acquisition.

Stage 1 (this script): accumulate the per-pixel-mean GGN (P x P) of the head's
final 1x1 conv over a random subset of training prompt sets.  Because the
posterior is tempered by an explicit effective sample size (n_eff), only the
*mean* curvature is needed, and a few thousand prompt sets estimate a 17 x 17
matrix very accurately.

Stage 2: choose n_eff and prior_precision on validation data with
sweep_bald_laplace.py, then write them into the saved state with --n_eff/--prior
(or pass them here directly if already known).

Usage:
    python fit_bald_laplace.py --config config.yaml --run_id v3
    python fit_bald_laplace.py --config config.yaml --run_id v3 --n_eff 1e5 --prior 1.0
    python fit_bald_laplace.py --config config.yaml --run_id v3 --set_only --n_eff 1e5 --prior 1.0
"""

import argparse
import hashlib
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from modules.bald_laplace import LastLayerLaplace, load_bald_laplace, save_bald_laplace
from modules.dataset import HDF5HeadDataset
from modules.head_v5 import build_head


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--head_checkpoint", default=None, help="Override head checkpoint path")
    ap.add_argument("--out_dir", default=None, help="Default: <project_root>/laplace/no_image/run_<id>_bald")
    ap.add_argument("--n_fit_sets", type=int, default=4096, help="Random training prompt sets for the curvature")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--covariance", choices=["full", "diag"], default="full")
    ap.add_argument("--weighting", choices=["class_weighted", "bernoulli"], default="class_weighted",
                    help="class_weighted = curvature of BCE(pos_weight) the head was trained with")
    ap.add_argument("--n_eff", type=float, default=None)
    ap.add_argument("--prior", type=float, default=None)
    ap.add_argument("--set_only", action="store_true",
                    help="Do not refit; load the saved curvature and only set n_eff/prior")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    log = logging.getLogger("fit_bald_laplace")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    project_root = Path(cfg["data"]["project_root"])
    device = args.device or cfg["sam"].get("device", "cuda")
    if not torch.cuda.is_available():
        device = "cpu"

    if args.head_checkpoint:
        ckpt = Path(args.head_checkpoint)
    elif cfg["data"].get("head_checkpoint"):
        ckpt = Path(cfg["data"]["head_checkpoint"])
    else:
        ckpt = project_root / "checkpoints" / "no_image" / f"run_{args.run_id}" / "best.pt"
    out_dir = Path(args.out_dir) if args.out_dir else \
        project_root / "laplace" / "no_image" / f"run_{args.run_id}_bald"

    head = build_head(cfg["head"])     # "no_image" (v3) or "dilated_v5"
    state = torch.load(ckpt, map_location="cpu")
    head.load_state_dict(state.get("model_state_dict", state))
    head.eval().to(device)

    if args.set_only:
        la, meta = load_bald_laplace(head, out_dir, device)
        meta.pop("n_eff", None)
        meta.pop("prior_precision", None)
    else:
        h5 = project_root / "training_data" / "no_image" / "train.h5"
        ds = HDF5HeadDataset(str(h5), logit_clip=cfg["training"].get("logit_clip", 20.0))
        rng = np.random.default_rng(args.seed)
        n = min(args.n_fit_sets, len(ds))
        idx = np.sort(rng.choice(len(ds), size=n, replace=False))
        loader = DataLoader(Subset(ds, idx.tolist()), batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
        la = LastLayerLaplace(head, device, covariance=args.covariance)
        pos_weight = None
        if args.weighting == "class_weighted":
            import pandas as pd
            m = pd.read_csv(project_root / "manifests" / "train.csv")
            pos_weight = float(m["bg_pixels"].sum() / max(m["fg_pixels"].sum(), 1))
            log.info("pos_weight = bg/fg from training manifest = %.4f", pos_weight)
        log.info("Accumulating %s curvature over %d random training prompt sets (seed %d)...",
                 args.weighting, n, args.seed)
        la.fit(loader, weighting=args.weighting, pos_weight=pos_weight)
        meta = {}

    if args.n_eff is not None:
        if args.prior is None:
            ap.error("--prior is required with --n_eff")
        la.set_posterior(args.n_eff, args.prior, args.covariance)
        log.info("Posterior set: covariance=%s n_eff=%.3g prior=%.3g data share %.4f-%.4f",
                 la.covariance, la.n_eff, la.prior_precision,
                 la.data_share().min().item(), la.data_share().max().item())

    meta.update({
        "source_checkpoint": str(ckpt),
        "source_checkpoint_md5": md5(ckpt),
        "fit_seed": args.seed,
        "target_layer": "final_conv",
        "likelihood": "bernoulli",
        "curvature": "GGN, per-pixel mean, tempered by n_eff",
        "bald": "exact Gauss-Hermite (64 nodes), float64",
    })
    save_bald_laplace(la, out_dir, meta)
    log.info("Saved to %s", out_dir)


if __name__ == "__main__":
    main()
