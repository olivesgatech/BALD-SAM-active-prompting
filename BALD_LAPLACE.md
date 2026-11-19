# BALD-Laplace (full-covariance, tempered): method

## Model

The head's final layer is a 1x1 conv, i.e. linear per pixel:

    f(q) = w^T phi(q),        phi(q) = [C backbone features at pixel q, 1]   (P = C + 1: 17 for v3, 33 for v5)

**Curvature.** Generalized Gauss-Newton of the BCE data term, as a per-pixel mean
over training prompt sets (full P x P matrix):

    H = (1/N_pix) * sum_{n,q} c_nq * s'(f_nq) * phi_nq phi_nq^T,     s'(f) = s(f)(1 - s(f))

with `c = 1` (`--weighting bernoulli`, the default) or `c = pos_weight*y + (1 - y)`
(`--weighting class_weighted`, the BCE term the heads are trained with). For a
linear last layer the GGN equals the exact Hessian of that term. The Dice term is
not included. A few thousand random training prompt sets estimate H to a few
percent (two random subsets of 2,048 and 4,096 sets agree to within 3% in every
eigenvalue).

**Posterior.**

    Sigma = ( n_eff * H + prior_precision * I )^{-1}

`n_eff` is the effective number of independent observations. Pixels inside a
prompt set are strongly correlated, so neither extreme is right: `n_eff = N_pix`
(every pixel independent) collapses the posterior; `n_eff = 1` (dividing the
summed curvature by the pixel count) lets the prior dominate, and then the
predictive variance is just ||phi||^2 / prior. With `n_eff = 1e5`, `prior = 1`
the data supply 97.7-100% of the posterior precision in every eigen-direction.

The posterior is centred at the trained weights (the deployed predictor). The
heads are early-stopped and trained with BCE + Dice, so these weights are not an
exact stationary point of the BCE term; the Gaussian is a GGN posterior around
the trained weights, the usual practice for early-stopped networks.

**Predictive logit.** f(q) ~ N(mu(q), v(q)) with

    mu(q) = w^T phi(q),        v(q) = phi(q)^T Sigma phi(q)

With the full covariance and a data-dominated posterior, v(q) is a leverage
score, v(q) ~ phi^T H^{-1} phi / n_eff: large when phi(q) points in a direction
the curvature-weighted training features rarely cover, not merely when
||phi(q)|| is large.

**BALD** (mutual information between the label at q and w; y depends on w only
through f(q), so BALD is exact given mu and v):

    BALD(q) = H[ E sigmoid(f) ] - E H[ sigmoid(f) ] = E_f KL( Bern(sigmoid(f)) || Bern(p_bar) )

Computed by quadrature rather than Monte Carlo: Gauss-Hermite on the expected-KL
form for std <= 2 (no cancellation, accurate in float32), an exact split at
f = 0 with Gauss-Laguerre for std > 2, and the series v * s'(mu) / 2 for
v <= 1e-6. Against a dense float64 reference over var in [1e-10, 1e4],
|mu| <= 20: relative error <= 7e-6 (float64) and <= 2e-4 (float32) wherever
BALD > 1e-11. No sampling noise, so results do not depend on a sample count K.

For small v, BALD ~= v(q) * s'(mu(q)) / 2: epistemic logit variance times the
Bernoulli variance at the current prediction. Since v ~ phi^T H^{-1} phi / n_eff
in that regime, the BALD *ranking* does not depend on `n_eff` or the prior once
`n_eff` is large (validation: `n_eff = 1e5` and `1e8` choose the same first query on every image and
94% of all queries).

Deep-ensemble BALD (option C) is computed the same cancellation-free way:
mean over heads of KL( Bern(p_m) || Bern(p_bar) ).

## Candidate pool

Two settings decide which pixels can be queried:

* **Border margin** (always on): the heads' convs zero-pad, so pixels within the
  receptive-field radius of the border (2 px for v3, 8 px for v5) have features
  computed from padding. With a data-dominated full covariance these pixels get
  very large variance (in a smoke test, 39% of queries landed on the border
  without the margin). `--border_margin` (CLI) / automatic in `modules/acquire.py`.
* **Exclusion radius around existing prompts.** The heads read prompts as
  Gaussian maps (sigma = 8 px), and training prompts are sampled uniformly inside
  foreground and background, so the posterior is most uncertain next to prompts
  placed on boundaries. Whether to let BALD query there is a budget trade-off
  (PointPrompt test, all methods):
  * up to ~10 prompts: a radius of `2 * sigma = 16 px` is better (spreads prompts out);
  * ~20 prompts: the original 5 px radius is better (late prompts refine boundaries
    next to earlier ones).

  Defaults: 5 px. Use 16 px for short sessions.
