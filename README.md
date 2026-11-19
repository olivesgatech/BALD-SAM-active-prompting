# BALD acquisition for prompt-aware heads on frozen SAM

Code and checkpoints for choosing the next point prompt with BALD on top of a
frozen SAM (ViT-H). A small head reads `[SAM logits, positive-prompt map,
negative-prompt map]` at 256 x 256 and outputs refined logits; uncertainty over
the head drives the choice of the next prompt.

This release adds a corrected BALD-Laplace posterior and packages three options
with trained checkpoints. Method details: [BALD_LAPLACE.md](BALD_LAPLACE.md).

## Options

| | Option | Head | Uncertainty | Extra training | Per-step cost (head only) |
|---|---|---|---|---|---|
| **B** | **BALD-Laplace on the dilated head** (recommended single model) | `checkpoints/v5` (28,673 params) | full-covariance, tempered last-layer Laplace (33 params); exact BALD | none (fit included) | 1 forward + 33x33 quadratic form |
| **C** | Deep-ensemble BALD | `checkpoints/v5_ensemble` (10 x v5) | mutual information across 10 heads | none here; 10 heads to retrain on new data | 10 forwards |
| **A** | BALD-Laplace on the 3-layer head | `checkpoints/v3` (2,785 params) | same as B (17 params) | none (fit included) | 1 forward + 17x17 quadratic form |

`checkpoints/v3/laplace_released` is the earlier diagonal fit, kept for
comparison; `checkpoints/v4_dropout` is the Dropout2d head used for the
MC-dropout baseline.

**Candidate pool.** All options exclude a border band equal to the head's
receptive-field radius (2 px for v3, 8 px for v5). The exclusion radius around
existing prompts trades off with the session length: 16 px is better for up to
~10 prompts, 5 px (default) for ~20 prompts. See BALD_LAPLACE.md.

## Results

PointPrompt test images (147: 8 per category for 17 categories, plus all 20 Salt_dome images), 1 seed,
initial prompt at the GT centroid, labels from the GT mask, 20 prompts.
IoU of SAM's own mask; AUC = mean IoU gain over steps 1-19; gain = IoU(step 19) - IoU(step 0).

| Acquisition (5 px exclusion) | AUC | gain | Salt_dome AUC | Salt_dome gain |
|---|---|---|---|---|
| C: ensemble BALD, 10 x v5 | **0.103** | **0.141** | 0.194 | 0.319 |
| **B: BALD-Laplace, v5 head** | 0.095 | 0.135 | **0.222** | 0.377 |
| entropy, v5 head | 0.088 | 0.140 | 0.214 | 0.396 |
| entropy, v3 head (earlier baseline) | 0.079 | 0.100 | 0.067 | 0.132 |
| A: BALD-Laplace, v3 head | 0.053 | 0.064 | 0.129 | 0.212 |
| earlier diagonal BALD-Laplace, v3 head | 0.051 | 0.111 | 0.178 | 0.444 |
| random | 0.034 | -0.041 | 0.059 | 0.045 |

Paired comparisons over the 147 images (bootstrap 95% CI, permutation p):

* B vs C: AUC -0.008 [-0.033, +0.015], p = 0.52; gain -0.007 [-0.038, +0.024], p = 0.69 -- no detectable difference, at one tenth of the training and head-inference cost.
* B vs earlier diagonal BALD-Laplace: AUC +0.044 [+0.017, +0.072], p = 0.001.
* B vs entropy on the same head: AUC +0.007, p = 0.60 (no detectable difference).
* A vs entropy on the v3 head: AUC -0.026, p = 0.03. On the v3 head the corrected posterior does not beat entropy; the stronger head is what makes BALD competitive.

Validation (68 PointPrompt validation images, 20 prompts): B vs C AUC -0.010
(p = 0.65) and B vs entropy on the v5 head -0.010 (p = 0.68), no detectable
difference, as on test. B vs the earlier diagonal BALD-Laplace (+0.014, p = 0.44)
and A vs entropy on the v3 head (-0.006, p = 0.77) are not significant on
validation, so treat those two test-set differences as less certain.

Short sessions (AUC over the first 9 added prompts, 16 px exclusion):
C 0.091, B 0.080, entropy (v5) 0.072. On Salt_dome, B leads over the first 9
prompts with either radius (0.118-0.123 vs 0.083-0.087 for C).

## Uncertainty quality (independent of prompting)

How well each map predicts where independently trained models disagree: Spearman
correlation, per input, with the mutual-information map of 5 held-out v5 heads
(trained like `v5_ensemble` but not used by any method). 300 PointPrompt test
inputs (ID) and 780 shifted inputs (SAM2 / SAM3 / medically fine-tuned SAM
logits on the same images, SAM1 on chest X-ray and retinal images).

| Uncertainty map | ID | shifted |
|---|---|---|
| ensemble BALD, 5 heads | **0.574** | **0.587** |
| entropy, v5 head | 0.527 | 0.566 |
| B: BALD-Laplace, v5 head | 0.516 | 0.567 |
| full-head diagonal Laplace, v5 (50 samples) | 0.410 | 0.526 |
| A: BALD-Laplace, v3 head | 0.399 | 0.456 |
| MC dropout, v5 head (50 samples) | 0.392 | 0.335 |
| earlier diagonal BALD-Laplace, v3 head | 0.302 | 0.223 |

Correlation of the predictive logit std with ||phi|| (1.0 = the variance is just
the feature norm): earlier diagonal fit 1.00, A 0.74, B 0.45.

Known behaviour: all of these maps, including the ensemble's, are concentrated
near the most recently added prompt (about 10x that region's share of the image
for B and for the ensemble). The heads were trained with prompts sampled inside
foreground and background, so prompts placed on object boundaries are outside
the training distribution and the models genuinely disagree next to them.

## Install

    pip install -r requirements.txt
    pip install git+https://github.com/facebookresearch/segment-anything.git

and download the SAM ViT-H checkpoint `sam_vit_h_4b8939.pth`. Set the paths in
`configs/config_v3.yaml` / `configs/config_v5.yaml` (`images_root`, `masks_root`,
`project_root`, `sam.checkpoint`). Run commands from the repository root.

## Use from Python (GUI / custom loop)

    from modules.acquire import Acquirer
    acq = Acquirer.from_repo("v5_laplace", device="cuda")   # or "v5_ensemble", "v3_laplace", "v3_entropy"
    x, y, score_map = acq.next_point(sam_logits_256, pos_points, neg_points)
    refined = acq.refined_logits(sam_logits_256, pos_points, neg_points)

    acq_short = Acquirer.from_repo("v5_laplace", exclusion_radius=16)   # sessions of up to ~10 prompts

Points are `[x, y]` in the 256 x 256 frame; `sam_logits_256` is SAM's logit map
(best multimask branch by SAM score) resized to 256 x 256, as produced by
`modules/sam_wrapper.run_sam_for_promptset`. The label of the returned point
comes from the user.

## Active-prompting evaluation

    # Option B
    python active_prompting.py --config configs/config_v5.yaml --run_id B \
        --strategies bald_laplace_full entropy random \
        --bald_laplace_dir checkpoints/v5/bald_laplace_bernoulli --border_margin 8

    # Option C
    python active_prompting.py --config configs/config_v5.yaml --run_id C \
        --strategies bald_ensemble entropy random \
        --ensemble checkpoints/v5_ensemble/m*/best.pt --border_margin 8

    # Option A
    python active_prompting.py --config configs/config_v3.yaml --run_id A \
        --strategies bald_laplace_full entropy random \
        --bald_laplace_dir checkpoints/v3/bald_laplace_bernoulli --border_margin 2

Add `--exclusion_radius 16` for short sessions. `--exclusion_radius` and
`--border_margin` apply to every strategy in a run, so baselines share the
candidate pool. With `config_v5.yaml` there is no diagonal Laplace fit, so
`entropy`, ECE and NLL use the head's MAP probability.

`sweep_bald_laplace.py` is a faster evaluator for comparing many acquisition
variants on the same images (caches SAM image embeddings; records SAM IoU, head
IoU and query geometry). Variant syntax: `full:b:1e5:1@5b8` = full covariance,
Bernoulli curvature, n_eff 1e5, prior 1, 5 px exclusion, 8 px border;
`ens@5b8` = ensemble BALD (with `--ensemble ...`); `entropy_map@5b8`; `random@5b8`.

## Fitting on your own head or data

The data pipeline (stages 1-4.5) produces `training_data/no_image/train.h5`:

    python build_manifests.py --config CONFIG
    python generate_promptsets.py --config CONFIG
    python precompute_sam_outputs.py --config CONFIG
    python build_training_indices.py --config CONFIG
    python pack_to_hdf5.py --config CONFIG

Train a head (`train_no_image_head.py` for v3; `train_v5_head.py --seed S` for v5
and for each ensemble member), then fit and set the posterior:

    python fit_bald_laplace.py --config CONFIG --run_id ID --weighting bernoulli \
        --head_checkpoint PATH/best.pt --out_dir OUT --n_eff 1e5 --prior 1.0

`n_eff = 1e5`, `prior = 1`, Bernoulli curvature were selected on PointPrompt
validation images. For large `n_eff` the BALD ranking does not depend on `n_eff`
or the prior, so the setting transfers; `sweep_bald_laplace.py` re-checks it on
your validation split.
