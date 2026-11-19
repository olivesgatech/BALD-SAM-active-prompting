"""
Stage 5 (v5): train a corrective head with the RAM-resident loader.

Differences from train_no_image_head.py:
  * RAM-resident batching (modules/dataset_ram.py) instead of the HDF5
    DataLoader — removes the 922 ms/step data bottleneck. Falls back to HDF5
    with a loud warning if the split does not fit in memory.
  * Pluggable head via modules/head_v5.build_head (`no_image` | `dilated_v5`).
  * --seed makes a run reproducible AND makes ensemble members differ: the
    seed drives weight init, batch order, and (optionally) a bootstrap
    resample of the training split.

Usage:
    python train_v5_head.py --config config_v5.yaml --run_id v5 --seed 0
    python train_v5_head.py --config config_v5.yaml --run_id v5_m03 --seed 3
    python train_v5_head.py --config config_v5.yaml --run_id v5 --seed 0 --sanity_check
"""

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import yaml
from torch.cuda.amp import GradScaler, autocast

import pandas as pd

from modules.checkpoint import load_checkpoint, save_checkpoint
from modules.head_v5 import build_head
from modules.losses import CombinedLoss
from modules.metrics import dice_from_logits, iou_from_logits
from modules.wandb_utils import init_wandb


def setup_logging(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
        force=True,
    )


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_pos_weight(manifest_train_csv: str, device: str) -> torch.Tensor:
    df = pd.read_csv(manifest_train_csv)
    total_fg, total_bg = df["fg_pixels"].sum(), df["bg_pixels"].sum()
    ratio = total_bg / max(total_fg, 1)
    logging.info("pos_weight = bg/fg = %.1f / %.1f = %.4f", total_bg, total_fg, ratio)
    return torch.tensor([ratio], dtype=torch.float32, device=device)


def build_scheduler(optimizer, tcfg, num_epochs):
    st = tcfg.get("scheduler", "cosine").lower()
    if st == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min=tcfg.get("scheduler_min_lr", 1e-6))
    if st == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", patience=tcfg.get("scheduler_plateau_patience", 3),
            min_lr=tcfg.get("scheduler_min_lr", 1e-6))
    if st == "none":
        return None
    raise ValueError(f"Unknown scheduler: {st}")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def build_loaders(cfg, device, batch_size):
    """
    Returns (train_loader, val_loader, mode) where mode is 'ram' or 'hdf5'.
    Raises if neither works — never silently degrades to a wrong split.
    """
    project_root = Path(cfg["data"]["project_root"])
    data_dir = project_root / "training_data" / "no_image"
    clip = cfg["training"].get("logit_clip", 20.0)

    if cfg["training"].get("ram_resident", True):
        try:
            from modules.dataset_ram import build_ram_loaders
            L = build_ram_loaders(data_dir, ["train", "val"], batch_size, device, logit_clip=clip)
            return L["train"], L["val"], "ram"
        except MemoryError as e:
            logging.warning("RAM-resident loading unavailable (%s). Falling back to HDF5.", e)

    from torch.utils.data import DataLoader
    from modules.dataset import HDF5HeadDataset
    nw = cfg["training"].get("num_workers", 8)
    kw = dict(num_workers=nw, pin_memory=True, persistent_workers=nw > 0)
    if nw > 0:
        kw["prefetch_factor"] = cfg["training"].get("prefetch_factor", 2)
    tr = HDF5HeadDataset(str(data_dir / "train.h5"), logit_clip=clip)
    va = HDF5HeadDataset(str(data_dir / "val.h5"), logit_clip=clip)
    return (DataLoader(tr, batch_size=batch_size, shuffle=True, **kw),
            DataLoader(va, batch_size=batch_size, shuffle=False, **kw), "hdf5")


def epoch_batches(loader, mode, device, batch_size, epoch, seed, shuffle):
    """Uniform iteration over either loader type, with per-epoch shuffling."""
    if mode == "ram":
        return loader.split.iter_batches(
            batch_size, device, shuffle=shuffle,
            seed=None if not shuffle else seed * 100_003 + epoch)
    return loader


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------

def train_one_epoch(model, batches, optimizer, criterion, scaler, device, use_amp, wandb_run, gstep):
    model.train()
    total_loss, n = 0.0, 0
    for batch in batches:
        x = batch["input"].to(device, non_blocking=True)
        y = batch["target"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=use_amp):
            loss = criterion(model(x), y)
        if use_amp:
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        else:
            loss.backward(); optimizer.step()
        total_loss += loss.item() * x.size(0); n += x.size(0); gstep += 1
        if gstep % 100 == 0:
            wandb_run.log({"train/step_loss": loss.item(), "train/global_step": gstep})
    return (total_loss / n if n else 0.0), gstep


@torch.no_grad()
def evaluate(model, batches, criterion, device, use_amp):
    model.eval()
    tl = ti = td = 0.0
    nb = 0
    for batch in batches:
        x = batch["input"].to(device, non_blocking=True)
        y = batch["target"].to(device, non_blocking=True)
        with autocast(enabled=use_amp):
            pred = model(x)
            loss = criterion(pred, y)
        tl += loss.item(); ti += iou_from_logits(pred, y); td += dice_from_logits(pred, y); nb += 1
    nb = max(nb, 1)
    return tl / nb, ti / nb, td / nb


@torch.no_grad()
def sam_baseline_iou(batches, device):
    """IoU of raw SAM (channel 0 > 0) on the same batches — the bar to beat."""
    tot, nb = 0.0, 0
    for batch in batches:
        x = batch["input"].to(device); y = batch["target"].to(device)
        tot += iou_from_logits(x[:, :1], y); nb += 1
    return tot / max(nb, 1)


def run_sanity_check(model, batches, device, cfg):
    """
    Overfit a handful of samples.

    The bar is NOT "IoU -> 1". This head is a *convolution of SAM's logits and
    the prompt maps* with no access to sample identity or pixels, so where SAM
    is confidently wrong over a region wider than the receptive field, no
    setting of the weights can recover it — a perfect-memorisation bar would
    fail a perfectly healthy head. What must hold is that optimisation works
    and that the head learns something SAM did not already provide:

        1. the loss falls substantially (>50%)
        2. the head beats raw SAM on the very samples it is overfitting
    """
    logging.info("=== Sanity check: overfit 4 samples ===")
    model.train()
    criterion = CombinedLoss(dice_lambda=cfg["training"]["dice_lambda"])
    opt = optim.AdamW(model.parameters(), lr=1e-2)
    b = next(iter(batches))
    x, y = b["input"][:4].to(device), b["target"][:4].to(device)

    sam_iou = iou_from_logits(x[:, :1], y)
    with torch.no_grad():
        loss0 = criterion(model(x), y).item()

    for step in range(300):
        opt.zero_grad(); loss = criterion(model(x), y); loss.backward(); opt.step()
        if step % 75 == 0:
            logging.info("  step=%d loss=%.4f iou=%.4f", step, loss.item(),
                         iou_from_logits(model(x).detach(), y))

    with torch.no_grad():
        pred = model(x)
        final_loss, final_iou = criterion(pred, y).item(), iou_from_logits(pred, y)

    drop = 1.0 - final_loss / max(loss0, 1e-9)
    logging.info("  loss %.4f -> %.4f (%.0f%% drop) | IoU %.4f vs raw SAM %.4f (%+.4f)",
                 loss0, final_loss, 100 * drop, final_iou, sam_iou, final_iou - sam_iou)

    fails = []
    if drop < 0.5:
        fails.append(f"loss dropped only {100*drop:.0f}% (need >50%)")
    if final_iou <= sam_iou:
        fails.append(f"IoU {final_iou:.4f} did not beat raw SAM {sam_iou:.4f}")
    if fails:
        logging.error("Sanity check FAILED: %s", "; ".join(fails))
        sys.exit(1)
    logging.info("Sanity check PASSED: loss -%.0f%%, IoU %+.4f over raw SAM",
                 100 * drop, final_iou - sam_iou)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--sanity_check", action="store_true")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    project_root = Path(cfg["data"]["project_root"])
    run_dir = project_root / "checkpoints" / "no_image" / f"run_{args.run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(project_root / "logs" / f"train_{args.run_id}.log")
    logging.info("=== v5 training: run_id=%s seed=%d ===", args.run_id, args.seed)

    set_seed(args.seed)
    (run_dir / "config.yaml").write_text(yaml.dump(cfg))

    tcfg, hcfg = cfg["training"], cfg["head"]
    device = cfg["sam"].get("device", "cuda") if torch.cuda.is_available() else "cpu"
    bs = tcfg["batch_size"]

    wandb_run = init_wandb(
        cfg, job_type="train", run_name=f"train_{args.run_id}",
        extra_config={"run_id": args.run_id, "seed": args.seed, "resumed": args.resume},
    )

    train_loader, val_loader, mode = build_loaders(cfg, device, bs)
    logging.info("Data mode: %s", mode)

    model = build_head(hcfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_final = sum(p.numel() for p in model.final_conv.parameters())
    logging.info("Head: %s  %d params (final_conv %d, %.1f%%)  RF=%s px",
                 type(model).__name__, n_params, n_final, 100 * n_final / n_params,
                 getattr(model, "receptive_field", "n/a"))
    wandb_run.log({"model/n_params": n_params, "model/n_params_final_conv": n_final})

    tb = lambda ep, sh: epoch_batches(train_loader, mode, device, bs, ep, args.seed, sh)
    vb = lambda: epoch_batches(val_loader, mode, device, bs, 0, args.seed, False)

    if args.sanity_check:
        run_sanity_check(model, tb(0, True), device, cfg)
        wandb_run.finish()
        return

    base = sam_baseline_iou(vb(), device)
    logging.info("Raw SAM val IoU (bar to beat): %.4f", base)
    wandb_run.log({"val/sam_baseline_iou": base})

    optimizer = optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    use_amp = tcfg.get("amp", True) and device != "cpu"
    scaler = GradScaler(enabled=use_amp)
    pos_weight = compute_pos_weight(str(project_root / "manifests" / "train.csv"), device)
    criterion = CombinedLoss(dice_lambda=tcfg["dice_lambda"], pos_weight=pos_weight)

    num_epochs = tcfg["num_epochs"]
    scheduler = build_scheduler(optimizer, tcfg, num_epochs)
    sched_type = tcfg.get("scheduler", "cosine").lower()
    patience = tcfg.get("early_stop_patience", 10)

    start_epoch, best_val_iou, patience_counter, gstep = 0, 0.0, 0, 0
    if args.resume:
        st = load_checkpoint(run_dir, model, optimizer, scheduler, scaler)
        start_epoch, best_val_iou = st["epoch"] + 1, st["best_val_iou"]

    epoch = start_epoch - 1
    for epoch in range(start_epoch, num_epochs):
        t0 = time.time()
        train_loss, gstep = train_one_epoch(
            model, tb(epoch, True), optimizer, criterion, scaler, device, use_amp, wandb_run, gstep)
        val_loss, val_iou, val_dice = evaluate(model, vb(), criterion, device, use_amp)

        if scheduler is not None:
            scheduler.step(val_iou) if sched_type == "plateau" else scheduler.step()

        is_best = val_iou > best_val_iou
        best_val_iou, patience_counter = (val_iou, 0) if is_best else (best_val_iou, patience_counter + 1)
        lr = optimizer.param_groups[0]["lr"]
        logging.info(
            "Epoch %3d/%d  train=%.4f  val=%.4f  iou=%.4f (SAM %.4f, %+.4f)  dice=%.4f  lr=%.2e  %.0fs  %s",
            epoch + 1, num_epochs, train_loss, val_loss, val_iou, base, val_iou - base,
            val_dice, lr, time.time() - t0,
            "[BEST]" if is_best else f"[patience {patience_counter}/{patience}]")

        wandb_run.log({
            "epoch": epoch + 1, "train/loss": train_loss, "val/loss": val_loss,
            "val/iou": val_iou, "val/dice": val_dice, "val/best_iou": best_val_iou,
            "val/iou_over_sam": val_iou - base, "train/lr": lr,
            "train/patience_counter": patience_counter, "train/epoch_seconds": time.time() - t0,
        })
        save_checkpoint(run_dir, model, optimizer, scheduler, scaler, epoch, best_val_iou, is_best)
        if patience_counter >= patience:
            logging.info("Early stopping after %d epochs without improvement.", patience)
            break

    summary = {
        "run_id": args.run_id, "seed": args.seed, "head": type(model).__name__,
        "n_params": n_params, "best_val_iou": best_val_iou,
        "sam_baseline_val_iou": base, "iou_over_sam": best_val_iou - base,
        "epochs_run": epoch + 1, "data_mode": mode,
    }
    (run_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    wandb_run.log({f"summary/{k}": v for k, v in summary.items() if isinstance(v, (int, float))})

    try:
        import wandb as _wandb
        if _wandb.run is not None:
            art = _wandb.Artifact(name=f"head_{args.run_id}", type="model", metadata=summary)
            art.add_file(str(run_dir / "best.pt"))
            _wandb.log_artifact(art)
    except Exception as e:
        logging.warning("wandb artifact log failed: %s", e)

    wandb_run.finish()
    logging.info("Done. best val IoU=%.4f (raw SAM %.4f, delta %+.4f)",
                 best_val_iou, base, best_val_iou - base)

    if best_val_iou <= base:
        logging.error("STOP: head (%.4f) did not beat raw SAM (%.4f).", best_val_iou, base)
        sys.exit(2)


if __name__ == "__main__":
    main()
