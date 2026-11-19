"""
Next-prompt selection API for interactive use (GUIs, custom loops).

    from modules.acquire import Acquirer
    acq = Acquirer.from_repo("v3_laplace", device="cuda")      # see PRESETS
    x, y, score = acq.next_point(sam_logits_256, pos_points, neg_points)

Inputs are in the head's 256 x 256 frame:
    sam_logits_256 : (256, 256) float array, SAM logits resized to 256 x 256
    pos_points, neg_points : lists of [x, y] (column, row) in 256 x 256 coordinates
Returns the selected (x, y) in the same frame and the (256, 256) acquisition map.

Candidate pool: pixels at least `exclusion_radius` px from every prompt (default 5;
16 is better for sessions of up to ~10 prompts, 5 for ~20) and outside the
receptive-field border band (`border_margin`, default = the head's RF radius).
The label of the selected point (positive / negative) comes from the user.
"""

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from modules.bald_laplace import ensemble_bald_map, load_bald_laplace
from modules.head_v5 import build_head
from modules.promptset import build_gaussian_map

PRESETS = {
    # name: (head config, head checkpoint, posterior dir or None, ensemble glob or None, kind)
    "v3_laplace":  ({"variant": "no_image", "channels_in": 3, "hidden_channels": 16},
                    "checkpoints/v3/best.pt", "checkpoints/v3/bald_laplace_bernoulli", None, "bald_laplace"),
    "v5_laplace":  ({"variant": "dilated_v5", "channels_in": 3, "hidden_channels": 32, "dilations": [1, 2, 4, 1]},
                    "checkpoints/v5/best.pt", "checkpoints/v5/bald_laplace_bernoulli", None, "bald_laplace"),
    "v5_ensemble": ({"variant": "dilated_v5", "channels_in": 3, "hidden_channels": 32, "dilations": [1, 2, 4, 1]},
                    "checkpoints/v5/best.pt", None, "checkpoints/v5_ensemble/m*/best.pt", "bald_ensemble"),
    "v3_entropy":  ({"variant": "no_image", "channels_in": 3, "hidden_channels": 16},
                    "checkpoints/v3/best.pt", None, None, "entropy"),
}


def rf_radius(head: nn.Module) -> int:
    """Receptive-field radius (px) of the head's conv backbone."""
    r = 0
    for m in head.backbone.modules():
        if isinstance(m, nn.Conv2d):
            r += m.dilation[0] * (m.kernel_size[0] - 1) // 2
    return r


def head_input(sam_logits_256: np.ndarray, pos: Sequence, neg: Sequence,
               sigma: float = 8.0, logit_clip: float = 20.0) -> torch.Tensor:
    pm = build_gaussian_map([list(p) for p in pos], 256, 256, sigma)
    nm = build_gaussian_map([list(p) for p in neg], 256, 256, sigma) if len(neg) else np.zeros((256, 256), np.float32)
    lg = np.clip(np.asarray(sam_logits_256, dtype=np.float32), -logit_clip, logit_clip)
    return torch.from_numpy(np.stack([lg, pm, nm], 0)).float().unsqueeze(0)


def candidate_exclusion(points: Sequence, radius: int, border_margin: int, H: int = 256, W: int = 256) -> np.ndarray:
    """True = not a candidate: within `radius` of an existing prompt or within `border_margin` of the border."""
    excl = np.zeros((H, W), dtype=bool)
    yy, xx = np.ogrid[:H, :W]
    for px, py in points:
        excl |= (xx - px) ** 2 + (yy - py) ** 2 <= radius * radius
    if border_margin > 0:
        m = border_margin
        excl[:m, :] = True; excl[-m:, :] = True; excl[:, :m] = True; excl[:, -m:] = True
    return excl


class Acquirer:
    def __init__(self, head: nn.Module, kind: str = "bald_laplace", laplace=None, ensemble=None,
                 exclusion_radius: int = 5, border_margin: Optional[int] = None,
                 sigma: float = 8.0, device: str = "cuda"):
        if kind not in ("bald_laplace", "bald_ensemble", "entropy"):
            raise ValueError(kind)
        self.head, self.kind, self.laplace, self.ensemble = head, kind, laplace, ensemble
        self.exclusion_radius = exclusion_radius
        self.border_margin = rf_radius(head) if border_margin is None else border_margin
        self.sigma, self.device = sigma, device

    @classmethod
    def from_repo(cls, preset: str, root: str = ".", device: str = "cuda", **kw) -> "Acquirer":
        hcfg, ck, post, ens_glob, kind = PRESETS[preset]
        root = Path(root)

        def load(path):
            h = build_head(hcfg)
            st = torch.load(path, map_location="cpu")
            h.load_state_dict(st.get("model_state_dict", st))
            return h.eval().to(device)

        head = load(root / ck)
        laplace = load_bald_laplace(head, root / post, device)[0] if post else None
        ensemble = [load(p) for p in sorted(root.glob(ens_glob))] if ens_glob else None
        return cls(head, kind, laplace, ensemble, device=device, **kw)

    @torch.no_grad()
    def score(self, x: torch.Tensor) -> np.ndarray:
        x = x.to(self.device)
        feat = self.head.backbone(x)
        if self.kind == "bald_laplace":
            b, _ = self.laplace.bald(feat, n_nodes=32, dtype=torch.float32)
            return b.squeeze(0).double().cpu().numpy()
        if self.kind == "bald_ensemble":
            return ensemble_bald_map(self.ensemble, x).double().cpu().numpy()
        p = torch.sigmoid(self.head.final_conv(feat)).squeeze().double().clamp(1e-12, 1 - 1e-12)
        return (-(p * p.log() + (1 - p) * (1 - p).log())).cpu().numpy()

    @torch.no_grad()
    def next_point(self, sam_logits_256: np.ndarray, pos: Sequence, neg: Sequence) -> Tuple[int, int, np.ndarray]:
        x = head_input(sam_logits_256, pos, neg, self.sigma)
        s = self.score(x)
        excl = candidate_exclusion(list(pos) + list(neg), self.exclusion_radius, self.border_margin)
        a = s.copy()
        a[excl] = -np.inf
        yy, xx = np.unravel_index(int(np.argmax(a)), a.shape)
        return int(xx), int(yy), s

    @torch.no_grad()
    def refined_logits(self, sam_logits_256: np.ndarray, pos: Sequence, neg: Sequence) -> np.ndarray:
        """The head's refined (256, 256) logit map for the current prompts."""
        x = head_input(sam_logits_256, pos, neg, self.sigma).to(self.device)
        return self.head(x).squeeze().cpu().numpy()
