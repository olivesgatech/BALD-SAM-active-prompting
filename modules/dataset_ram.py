"""
RAM-resident loader for the no-image head.

Why this exists
---------------
`HDF5HeadDataset` does four scalar HDF5 reads per sample and upcasts each
float16 array to float32 on the CPU before collation. Measured on this machine
(batch 64, 8 workers):

    pure GPU train step  :   17.7 ms
    dataloader alone     :  922.1 ms      <- 98% of wall clock

The full train split is only ~23.5 GB in native dtype and the box has ~49 GB
free, so the entire split fits in RAM. Reading it once sequentially runs at
~2 GB/s (≈12 s), after which batch assembly is a pure in-memory gather and the
loop becomes GPU-bound.

Design
------
- Arrays stay **float16 / uint8** in RAM and on the PCIe transfer; the cast to
  float32 happens on the GPU, where it is free.
- No `DataLoader`, no workers, no collate: `iter_batches()` gathers a batch by
  fancy-indexing the resident numpy arrays.
- `logit_clip` is applied on GPU after the cast, so RAM holds raw values.

Falls back loudly rather than silently: if the split does not fit in the
memory budget, construction raises and the caller can drop back to HDF5.
"""

import logging
import time
from typing import Dict, Iterator, Optional

import numpy as np
import torch

log = logging.getLogger(__name__)

_KEYS_IN = ("logits", "pos_map", "neg_map")


def _available_ram_gb() -> float:
    """Available (not merely free) RAM in GB, read from /proc/meminfo."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / (1024.0 ** 2)
    except Exception:
        pass
    return float("inf")


class RamSplit:
    """
    One split (train/val/test) held entirely in RAM.

    Attributes:
        x:  (N, 3, H, W) float16  — [logits, pos_map, neg_map]
        y:  (N, 1, H, W) uint8    — GT mask
    """

    def __init__(
        self,
        h5_path: str,
        logit_clip: float = 20.0,
        ram_headroom_gb: float = 6.0,
        read_chunk: int = 2048,
    ):
        import h5py

        self.h5_path = h5_path
        self.logit_clip = logit_clip

        with h5py.File(h5_path, "r") as f:
            total = f["logits"].shape[0]
            if "valid" in f:
                idx = np.where(f["valid"][:])[0]
                if len(idx) < total:
                    log.warning("RamSplit: excluding %d invalid rows from %s",
                                total - len(idx), h5_path)
            else:
                idx = np.arange(total)

            n = len(idx)
            H, W = f["logits"].shape[1], f["logits"].shape[2]
            need_gb = (n * 3 * H * W * 2 + n * H * W) / 1e9
            avail = _available_ram_gb()
            if need_gb + ram_headroom_gb > avail:
                raise MemoryError(
                    f"RamSplit needs {need_gb:.1f} GB (+{ram_headroom_gb:.1f} GB headroom) "
                    f"for {h5_path} but only {avail:.1f} GB available. "
                    f"Set training.ram_resident=false to use the HDF5 loader."
                )

            log.info("RamSplit: loading %s (%d samples, %.1f GB) into RAM...",
                     h5_path, n, need_gb)
            t0 = time.time()

            self.x = np.empty((n, 3, H, W), dtype=np.float16)
            self.y = np.empty((n, 1, H, W), dtype=np.uint8)

            # Contiguous slab reads: ~5x faster than per-sample scalar indexing.
            contiguous = np.array_equal(idx, np.arange(total))
            for s in range(0, n, read_chunk):
                e = min(s + read_chunk, n)
                sel = slice(idx[s], idx[e - 1] + 1) if contiguous else idx[s:e].tolist()
                for c, k in enumerate(_KEYS_IN):
                    self.x[s:e, c] = f[k][sel]
                self.y[s:e, 0] = f["gt"][sel]

            self.sample_id = np.asarray(f["sample_id"][:])[idx]
            self.promptset_id = np.asarray(f["promptset_id"][:])[idx]

        self.n = n
        log.info("RamSplit: loaded %d samples in %.1f s (%.2f GB/s)",
                 n, time.time() - t0, need_gb / max(time.time() - t0, 1e-6))

    def __len__(self) -> int:
        return self.n

    def to_gpu(self, sel: np.ndarray, device: str) -> Dict[str, torch.Tensor]:
        """Gather rows `sel` and move to GPU, casting to float32 there."""
        xb = torch.from_numpy(self.x[sel]).to(device, non_blocking=True)
        yb = torch.from_numpy(self.y[sel]).to(device, non_blocking=True)
        xb = xb.float().clamp_(-self.logit_clip, self.logit_clip)
        return {"input": xb, "target": yb.float()}

    def iter_batches(
        self,
        batch_size: int,
        device: str,
        shuffle: bool = False,
        seed: Optional[int] = None,
        drop_last: bool = False,
    ) -> Iterator[Dict[str, torch.Tensor]]:
        order = np.arange(self.n)
        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(order)
        last = self.n - (self.n % batch_size) if drop_last else self.n
        for s in range(0, last, batch_size):
            sel = np.sort(order[s:s + batch_size])   # sorted gather is cache-friendlier
            yield self.to_gpu(sel, device)

    def num_batches(self, batch_size: int, drop_last: bool = False) -> int:
        return self.n // batch_size if drop_last else (self.n + batch_size - 1) // batch_size


class _RamLoaderShim:
    """
    Makes a RamSplit quack like a torch DataLoader for code that only does
    `for batch in loader:` and `len(loader)` — e.g. the Laplace fitting path.

    Batches are already on `device`, which is harmless: `.to(device)` on an
    already-resident tensor is a no-op.
    """

    def __init__(self, split: RamSplit, batch_size: int, device: str,
                 shuffle: bool = False, seed: Optional[int] = None):
        self.split, self.batch_size, self.device = split, batch_size, device
        self.shuffle, self.seed = shuffle, seed

    def __iter__(self):
        return self.split.iter_batches(self.batch_size, self.device,
                                       shuffle=self.shuffle, seed=self.seed)

    def __len__(self):
        return self.split.num_batches(self.batch_size)


def build_ram_loaders(data_dir, splits, batch_size, device, logit_clip=20.0):
    """
    Load the requested splits into RAM and return {split: _RamLoaderShim}.

    Raises MemoryError if a split does not fit — callers should catch and
    fall back to the HDF5 loader rather than proceed silently.
    """
    from pathlib import Path
    data_dir = Path(data_dir)
    out = {}
    for s in splits:
        rs = RamSplit(str(data_dir / f"{s}.h5"), logit_clip=logit_clip)
        out[s] = _RamLoaderShim(rs, batch_size, device)
        out[s].raw = rs
    return out
