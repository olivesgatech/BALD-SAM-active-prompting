"""
Last-layer Laplace posterior for BALD acquisition: full covariance, tempered
curvature, exact (quadrature) BALD.

Model
-----
The head's final 1x1 conv is linear per pixel:

    f(q) = w^T phi(q),   phi(q) = [backbone features at q, 1]  (P = C_in + 1)

Posterior over w (Bernoulli likelihood, generalized Gauss-Newton curvature):

    H_mean = (1/N_pix) * sum_{n,q} s'(f_nq) phi_nq phi_nq^T     (P x P, full)
    Prec   = n_eff * H_mean + prior_precision * I
    Sigma  = Prec^{-1}

`n_eff` is the effective number of independent observations.  Pixels within a
prompt set are strongly correlated, so treating all ~3.4e9 training pixels as
independent (n_eff = N_pix) collapses the posterior, and dividing by N_pix
(n_eff = 1) lets the prior swamp the data.  n_eff is a tempering
hyperparameter chosen on validation data (see sweep_bald_laplace.py).

Per-pixel predictive logit:  f(q) ~ N(mu(q), v(q)),
    mu(q) = w_MAP^T phi(q),   v(q) = phi(q)^T Sigma phi(q)

With the full covariance, v(q) is a leverage score: it is large when phi(q)
points in a direction the (curvature-weighted) training features rarely
cover, not merely when ||phi(q)|| is large.

BALD at pixel q (mutual information between y_q and w):

    BALD(q) = H[ E_f sigmoid(f) ] - E_f H[ sigmoid(f) ]

computed with Gauss-Hermite quadrature in float64.  This is the K -> infinity
limit of Monte Carlo BALD: deterministic, no sampling noise, and no float32
underflow when the posterior is narrow.
"""

import json
import logging
import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)

_PI_OVER_8 = math.pi / 8.0
_GH_CACHE: Dict[Tuple[int, str], Tuple[torch.Tensor, torch.Tensor]] = {}


# ---------------------------------------------------------------------------
# Exact BALD for a Bernoulli with Gaussian logit
# ---------------------------------------------------------------------------
def _gauss_hermite(n: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Nodes/weights for E[g(z)], z ~ N(0, 1) (probabilists' Hermite)."""
    key = ("hermite", n, str(device))
    if key not in _GH_CACHE:
        z, w = np.polynomial.hermite_e.hermegauss(n)
        w = w / w.sum()
        _GH_CACHE[key] = (
            torch.as_tensor(z, dtype=torch.float64, device=device),
            torch.as_tensor(w, dtype=torch.float64, device=device),
        )
    return _GH_CACHE[key]


def _gauss_laguerre(n: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Nodes/weights for int_0^inf e^{-r} g(r) dr."""
    key = ("laguerre", n, str(device))
    if key not in _GH_CACHE:
        r, a = np.polynomial.laguerre.laggauss(n)
        _GH_CACHE[key] = (
            torch.as_tensor(r, dtype=torch.float64, device=device),
            torch.as_tensor(a, dtype=torch.float64, device=device),
        )
    return _GH_CACHE[key]


def _binary_entropy_pq(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """H(p) with q = 1 - p supplied separately (avoids cancellation)."""
    return -(p * torch.log(p.clamp_min(1e-300)) + q * torch.log(q.clamp_min(1e-300)))


def _moments_hermite(m, s, n):
    """E[sigmoid(f)], E[sigmoid(-f)], E[h(sigmoid(f))] for f ~ N(m, s^2); m, s: (N,)."""
    z, w = _gauss_hermite(n, m.device)
    f = m[:, None] + s[:, None] * z
    p, q = torch.sigmoid(f), torch.sigmoid(-f)
    h = p * F.softplus(-f) + q * F.softplus(f)          # h(sigmoid(f)), stable
    return (p * w).sum(1), (q * w).sum(1), (h * w).sum(1)


def _moments_split(m, s, n):
    """
    Same moments, exact split at f = 0 (for large s, where sigmoid(f) is sharp
    on the scale of s and Gauss-Hermite cannot resolve it).

    With N(r) = density of f at r and Nm(r) = density of f at -r:
      sigmoid(f) = 1[f>0] - sign(f) * sigmoid(-|f|)
      => E[sigmoid(f)]    = Phi(m/s)  - int_0^inf sigmoid(-r) (N(r) - Nm(r)) dr
         E[sigmoid(-f)]   = Phi(-m/s) + int_0^inf sigmoid(-r) (N(r) - Nm(r)) dr
         E[h(sigmoid(f))] = int_0^inf h(sigmoid(r)) (N(r) + Nm(r)) dr      (h o sigmoid is even)
    Each remaining integrand is e^{-r} times a smooth function of r, so
    Gauss-Laguerre is accurate: e^{r} sigmoid(-r) = sigmoid(r), and
    e^{r} h(sigmoid(r)) grows only linearly in r.
    """
    r, a = _gauss_laguerre(n, m.device)
    c = 1.0 / (s[:, None] * math.sqrt(2.0 * math.pi))
    Np = c * torch.exp(-0.5 * ((r[None, :] - m[:, None]) / s[:, None]) ** 2)
    Nm = c * torch.exp(-0.5 * ((r[None, :] + m[:, None]) / s[:, None]) ** 2)
    sig_r = torch.sigmoid(r)
    # e^{r} h(sigmoid(r)) = e^{r} [sigmoid(r) softplus(-r) + sigmoid(-r) softplus(r)]
    #                     = sigmoid(r) * (e^{r} softplus(-r) + softplus(r))
    eh = sig_r * (torch.exp(r) * F.softplus(-r) + F.softplus(r))
    corr = (a * sig_r * (Np - Nm)).sum(1)
    p_bar = torch.special.ndtr(m / s) - corr
    q_bar = torch.special.ndtr(-m / s) + corr
    h_exp = (a * eh * (Np + Nm)).sum(1)
    return p_bar, q_bar, h_exp


def _bregman_softplus(b: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """
    KL( Bern(sigmoid(a)) || Bern(sigmoid(b)) ) = softplus(b) - softplus(a) - sigmoid(a) (b - a),
    the Bregman divergence of softplus.  Evaluated without catastrophic
    cancellation (safe in float32):
      * uses D(b, a) = D(-b, -a) to work with a <= 0 (small softplus / sigmoid);
      * 5th-order Taylor in d = b - a when |d| < 0.1 (truncation < 1e-6 relative).
    """
    flip = a > 0
    a = torch.where(flip, -a, a)
    b = torch.where(flip, -b, b)
    d = b - a
    sa = torch.sigmoid(a)
    s1 = sa * (1.0 - sa)                    # sigma'
    s2 = s1 * (1.0 - 2.0 * sa)              # sigma''
    s3 = s1 * (1.0 - 6.0 * s1)              # sigma'''
    s4 = s2 * (1.0 - 12.0 * s1)             # sigma''''
    taylor = d * d * (0.5 * s1 + d * (s2 / 6.0 + d * (s3 / 24.0 + d * s4 / 120.0)))
    direct = F.softplus(b) - F.softplus(a) - sa * d
    return torch.where(d.abs() < 0.1, taylor, direct).clamp_min(0.0)


@torch.no_grad()
def bald_from_gaussian_logit(
    mu: torch.Tensor, var: torch.Tensor, n_nodes: int = 64, s_switch: float = 2.0,
    dtype: torch.dtype = torch.float64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    BALD and predictive entropy for y ~ Bernoulli(sigmoid(f)), f ~ N(mu, var):

        BALD = H[E sigmoid(f)] - E H[sigmoid(f)] = E_f KL( Bern(sigmoid(f)) || Bern(p_bar) )

    std <= s_switch: Gauss-Hermite on the expected-KL form (no cancellation, so
                     float32 is accurate even for tiny variances);
    std >  s_switch: exact split at f = 0 + Gauss-Laguerre (BALD is O(0.1) here,
                     no cancellation issue).
    Returns (bald, predictive_entropy), shape of mu, in `dtype`.
    """
    shape = mu.shape
    m = mu.reshape(-1).to(dtype)
    v = var.reshape(-1).to(dtype).clamp_min(0.0)
    s = v.sqrt()
    bald = torch.empty_like(m)
    h_total = torch.empty_like(m)
    # var <= 1e-6: second-order series BALD = var * sigma'(mu) / 2 (relative error O(var))
    tiny = v <= 1e-6
    if tiny.any():
        mt = m[tiny]
        pt, qt = torch.sigmoid(mt), torch.sigmoid(-mt)
        bald[tiny] = 0.5 * v[tiny] * pt * qt
        h_total[tiny] = _binary_entropy_pq(pt, qt)
    lo = (s <= s_switch) & ~tiny
    if lo.any():
        z, w = _gauss_hermite(n_nodes, m.device)
        z, w = z.to(dtype), w.to(dtype)
        ml, sl = m[lo], s[lo]
        f = ml[:, None] + sl[:, None] * z
        p_bar = (torch.sigmoid(f) * w).sum(1)
        q_bar = (torch.sigmoid(-f) * w).sum(1)
        b = torch.log(p_bar.clamp_min(1e-300 if dtype == torch.float64 else 1e-38)) - \
            torch.log(q_bar.clamp_min(1e-300 if dtype == torch.float64 else 1e-38))
        bald[lo] = (_bregman_softplus(b[:, None].expand_as(f), f) * w).sum(1)
        h_total[lo] = _binary_entropy_pq(p_bar, q_bar)
    hi = s > s_switch
    if hi.any():
        p_hi, q_hi, h_hi = _moments_split(m[hi].double(), s[hi].double(), n_nodes)
        ht = _binary_entropy_pq(p_hi, q_hi)
        bald[hi] = (ht - h_hi).clamp_min(0.0).to(dtype)
        h_total[hi] = ht.to(dtype)
    return bald.reshape(shape), h_total.reshape(shape)


@torch.no_grad()
def ensemble_bald_map(members, x: torch.Tensor) -> torch.Tensor:
    """
    Deep-ensemble BALD for one input x (1, 3, H, W): mutual information across
    heads, computed as mean_m KL( Bern(p_m) || Bern(p_bar) ) (no cancellation,
    so it does not underflow when the heads nearly agree).  Returns (H, W).
    """
    f = torch.cat([m(x) for m in members], 1).squeeze(0).float()      # (M, H, W)
    p_bar = torch.sigmoid(f).mean(0)
    q_bar = torch.sigmoid(-f).mean(0)
    b = torch.log(p_bar.clamp_min(1e-38)) - torch.log(q_bar.clamp_min(1e-38))
    return _bregman_softplus(b.unsqueeze(0).expand_as(f), f).mean(0)


# ---------------------------------------------------------------------------
# Posterior
# ---------------------------------------------------------------------------
class LastLayerLaplace:
    """
    Gaussian (Laplace) posterior over the final 1x1 conv of a head exposing
    `.backbone` (input -> features) and `.final_conv` (features -> 1 logit).

    Usage::
        la = LastLayerLaplace(head, device)
        la.fit(train_loader)                       # accumulates H_mean once
        la.set_posterior(n_eff=1e5, prior_precision=1.0)
        feat = la.features(x)                      # (B, C, H, W)
        bald, ent = la.bald(feat)                  # (B, H, W) each
    """

    def __init__(self, head: nn.Module, device: str, covariance: str = "full"):
        if covariance not in ("full", "diag"):
            raise ValueError("covariance must be 'full' or 'diag'")
        self.head = head
        self.device = device
        self.covariance = covariance
        self.C_in = head.final_conv.weight.shape[1]
        self.P = self.C_in + 1

        self.H_mean: Optional[torch.Tensor] = None   # (P, P) float64
        self.weighting: Optional[str] = None
        self.pos_weight: Optional[float] = None
        self.n_pixels_fit = 0
        self.n_sets_fit = 0
        self.n_eff: Optional[float] = None
        self.prior_precision: Optional[float] = None
        self.Sigma: Optional[torch.Tensor] = None    # (P, P) float64

    # ------------------------------------------------------------------
    @property
    def w_map(self) -> torch.Tensor:
        fc = self.head.final_conv
        return torch.cat([fc.weight.detach().reshape(-1),
                          fc.bias.detach().reshape(-1)]).double().to(self.device)

    @torch.no_grad()
    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.head.backbone(x)

    @staticmethod
    def _phi(feat: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) -> (B*H*W, C+1) with a bias column."""
        B, C, H, W = feat.shape
        phi = feat.permute(0, 2, 3, 1).reshape(B * H * W, C)
        return torch.cat([phi, torch.ones_like(phi[:, :1])], dim=1)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def fit(self, loader, max_batches: Optional[int] = None,
            weighting: str = "bernoulli", pos_weight: Optional[float] = None) -> None:
        """
        Accumulate the full per-pixel-mean GGN over `loader` (float64).

        weighting:
          "bernoulli"      curvature of the plain Bernoulli NLL:  s'(f)
          "class_weighted" curvature of BCEWithLogits(pos_weight=pw), the data
                           term the head was trained on:  s'(f) * (pw*y + 1 - y)
        For a last layer f = w^T phi the GGN equals the exact Hessian of the
        BCE term (d^2 f / dw^2 = 0).
        """
        if weighting not in ("bernoulli", "class_weighted"):
            raise ValueError("weighting must be 'bernoulli' or 'class_weighted'")
        if weighting == "class_weighted" and pos_weight is None:
            raise ValueError("class_weighted curvature needs pos_weight")
        self.head.eval()
        H_sum = torch.zeros(self.P, self.P, dtype=torch.float64, device=self.device)
        n_pix, n_sets = 0, 0
        for i, batch in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            x = batch["input"].to(self.device).float()
            feat = self.head.backbone(x)
            logits = self.head.final_conv(feat)
            p = torch.sigmoid(logits)
            lam = (p * (1.0 - p)).reshape(-1, 1).double()
            if weighting == "class_weighted":
                y = batch["target"].to(self.device).reshape(-1, 1).double()
                lam = lam * (pos_weight * y + (1.0 - y))
            phi = self._phi(feat).double()
            H_sum += (phi * lam).T @ phi
            n_pix += phi.shape[0]
            n_sets += x.shape[0]
        if n_pix == 0:
            raise RuntimeError("fit(): loader produced no data")
        self.H_mean = H_sum / n_pix
        self.n_pixels_fit = n_pix
        self.n_sets_fit = n_sets
        self.weighting = weighting
        self.pos_weight = pos_weight
        evals = torch.linalg.eigvalsh(self.H_mean)
        log.info("fit(): %s curvature, %d prompt sets, %d pixels; H_mean eigenvalues [%.3e, %.3e]",
                 weighting, n_sets, n_pix, evals.min().item(), evals.max().item())

    def set_posterior(self, n_eff: float, prior_precision: float,
                      covariance: Optional[str] = None) -> None:
        if self.H_mean is None:
            raise RuntimeError("Call fit() or load_state_dict() first.")
        if covariance is not None:
            if covariance not in ("full", "diag"):
                raise ValueError("covariance must be 'full' or 'diag'")
            self.covariance = covariance
        H = float(n_eff) * self.H_mean
        if self.covariance == "diag":
            H = torch.diag(torch.diagonal(H))
        prec = H + float(prior_precision) * torch.eye(self.P, dtype=torch.float64, device=self.device)
        L = torch.linalg.cholesky(prec)
        self.Sigma = torch.cholesky_inverse(L)
        self.n_eff = float(n_eff)
        self.prior_precision = float(prior_precision)

    def data_share(self) -> torch.Tensor:
        """Fraction of posterior precision supplied by data, per eigen-direction of the precision."""
        H = self.n_eff * self.H_mean
        if self.covariance == "diag":
            h = torch.diagonal(H)
        else:
            h = torch.linalg.eigvalsh(H)
        return h / (h + self.prior_precision)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def logit_mean_var(self, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-pixel predictive logit mean and variance, each (B, H, W) float64."""
        if self.Sigma is None:
            raise RuntimeError("Call set_posterior() first.")
        B, _, Hh, Ww = feat.shape
        phi = self._phi(feat).double()
        mu = phi @ self.w_map
        var = ((phi @ self.Sigma) * phi).sum(1).clamp_min(0.0)
        return mu.reshape(B, Hh, Ww), var.reshape(B, Hh, Ww)

    @torch.no_grad()
    def bald(self, feat: torch.Tensor, n_nodes: int = 64,
             dtype: torch.dtype = torch.float64) -> Tuple[torch.Tensor, torch.Tensor]:
        """Exact BALD and predictive entropy maps, each (B, H, W).
        dtype=float32 is ~5x faster on consumer GPUs; relative error < 1e-4 wherever
        BALD > 1e-11 (validated against a dense float64 reference)."""
        mu, var = self.logit_mean_var(feat)
        return bald_from_gaussian_logit(mu, var, n_nodes=n_nodes, dtype=dtype)

    @torch.no_grad()
    def predict_proba(self, feat: torch.Tensor) -> torch.Tensor:
        """Probit-approximate predictive probability, (B, H, W) float64."""
        mu, var = self.logit_mean_var(feat)
        return torch.sigmoid(mu / torch.sqrt(1.0 + _PI_OVER_8 * var))

    # ------------------------------------------------------------------
    def state_dict(self) -> dict:
        return {
            "format": "bald_laplace_v1",
            "H_mean": self.H_mean.cpu(),
            "n_pixels_fit": self.n_pixels_fit,
            "n_sets_fit": self.n_sets_fit,
            "n_eff": self.n_eff,
            "prior_precision": self.prior_precision,
            "covariance": self.covariance,
            "weighting": self.weighting,
            "pos_weight": self.pos_weight,
            "C_in": self.C_in,
            "P": self.P,
        }

    def load_state_dict(self, state: dict) -> None:
        if state.get("format") != "bald_laplace_v1":
            raise ValueError("Not a bald_laplace_v1 state")
        if state["P"] != self.P:
            raise ValueError(f"State has P={state['P']} but head has P={self.P}")
        self.H_mean = state["H_mean"].to(self.device, torch.float64)
        self.n_pixels_fit = state["n_pixels_fit"]
        self.n_sets_fit = state["n_sets_fit"]
        self.weighting = state.get("weighting")
        self.pos_weight = state.get("pos_weight")
        if state.get("n_eff") is not None:
            self.set_posterior(state["n_eff"], state["prior_precision"], state["covariance"])


def save_bald_laplace(la: LastLayerLaplace, out_dir: Path, meta: dict) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(la.state_dict(), out_dir / "bald_laplace_state.pt")
    full_meta = {
        "format": "bald_laplace_v1",
        "covariance": la.covariance,
        "curvature_weighting": la.weighting,
        "pos_weight": la.pos_weight,
        "n_eff": la.n_eff,
        "prior_precision": la.prior_precision,
        "n_params": la.P,
        "n_sets_fit": la.n_sets_fit,
        "n_pixels_fit": la.n_pixels_fit,
    }
    if la.Sigma is not None:
        full_meta["data_share_min"] = float(la.data_share().min())
        full_meta["data_share_max"] = float(la.data_share().max())
    full_meta.update(meta)
    with open(out_dir / "bald_laplace_meta.json", "w") as f:
        json.dump(full_meta, f, indent=2)


def load_bald_laplace(head: nn.Module, out_dir: Path, device: str) -> Tuple[LastLayerLaplace, dict]:
    out_dir = Path(out_dir)
    state = torch.load(out_dir / "bald_laplace_state.pt", map_location="cpu")
    la = LastLayerLaplace(head, device, covariance=state.get("covariance", "full"))
    la.load_state_dict(state)
    meta = {}
    meta_path = out_dir / "bald_laplace_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
    return la, meta
