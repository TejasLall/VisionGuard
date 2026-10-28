# Methodology

This document explains every stage of the VisionGuard pipeline. File references point to the implementation.

```
image ─► validate ─► resize ─► [A] illumination-aware preprocessing ─► feature extraction (fine, coarse)
      ─► PCA normal subspace (per scale) ─► residual map per scale ─► upsample + smooth ─► z-normalise
      ─► [B] fusion  S = α·z_fine + (1−α)·z_coarse ─► [C] threshold T ─► binary mask ─► overlay / label
```

The three bracketed stages are our **proposed enhancements**; each can be switched on or off independently.
The underlying idea (model "normal" patches with a PCA subspace and score by reconstruction residual) is inspired by
*SubspaceAD* (Lendering et al., CVPR 2026) – see `LITERATURE_REVIEW.md`. **Our implementation is independent and
simplified; it is not a reproduction of that paper.**

## 1. Image loading and validation — `src/data/image_io.py`
Files are decoded with OpenCV (`imdecode`, so Windows paths with spaces/Unicode work). Only `.jpg .jpeg .png .bmp` are
accepted; empty, corrupt, too small (< 16 px) or oversized files raise `ImageLoadError`. Grey-scale and RGBA images are
converted to 3-channel RGB `uint8`. Images are then resized to a square analysis resolution (default 256 × 256; aspect
ratio is not preserved).

## 2. Contribution A – illumination-aware preprocessing — `src/preprocessing/illumination.py`
Disabled for the baseline. When enabled, stages are applied in this order, modifying only luminance except gray-world:

1. **Gray-world colour normalisation** (optional, off by default): scale each channel so that the R, G, B means are equal.
2. **Flat-field correction:** `L' = L / G_σ(L) · mean(G_σ(L))` where `G_σ` is a Gaussian blur with
   `σ = 0.2·min(H, W)`. This removes slowly varying shading (uneven lighting, soft shadows).
3. **Luminance normalisation:** `L'' = (L' − mean(L')) · g + μ_train`, with gain `g = σ_train / std(L')` clipped to
   [0.4, 2.5]. `μ_train` and `σ_train` are the average luminance mean/std of the **normal training images** (learned in
   `fit`, never from test images).
4. **CLAHE** (Contrast Limited Adaptive Histogram Equalisation) on the L channel of CIELAB, clip limit 2, 8 × 8 tiles.
   The mode is `always`, `auto` (only when a simple illumination check – mean outside [70, 190], low contrast or strong
   low-frequency shading – flags the image) or `off`.

Colour conversions go RGB → LAB(uint8) → RGB. **Caveat:** CLAHE amplifies local contrast, including noise and possibly
defects; flat-fielding can create halos at strong edges. Both are applied identically to training and test images, and the
`--op-ablation` experiment measures each operation separately. The dashboard shows original and processed images plus
luminance histograms so users can inspect the effect.

## 3. Feature extraction — `src/features/`
### 3.1 Handcrafted backend (default, CPU, not a deep model)
Per-pixel channels (17): CIELAB (3); |Sobel-x|, |Sobel-y|, gradient magnitude, |Laplacian| of L (4); uniform LBP (P = 8,
R = 1) one-hot (10). A **scale** pools these over a square window (box filter ⇒ window means; for LBP this is the local
histogram) and adds the local standard deviation of L and of the gradient magnitude ⇒ 19 features per patch, sub-sampled
with the scale's stride:

| Scale | Window (receptive field) | Stride | Grid at 256 px | Intended for |
|---|---|---|---|---|
| fine | 9 px | 4 px | 64 × 64 | scratches, cracks, small spots |
| coarse | 33 px | 8 px | 32 × 32 | larger regions, texture changes |

The scales therefore differ in **receptive field, feature resolution and neighbourhood aggregation**; the coarse map is
*not* a resized copy of the fine map.

### 3.2 Optional DINOv2 backend — `src/features/deep_features.py`
Pretrained `facebook/dinov2-small` (Hugging Face), input 448 px ⇒ 32 × 32 spatial tokens (patch size 14); the CLS token
is dropped so localisation is possible. *Fine* = mean of hidden states of layers 6–8 (more local); *coarse* = final-layer
tokens averaged over a 3 × 3 token neighbourhood (larger context). **These layer choices are ours, not claimed to match
the paper.** If DINOv2 cannot be loaded the program raises a clear error, or – only if `allow_fallback` is set – uses the
handcrafted backend and labels it "FALLBACK (NOT DINOv2)" everywhere.

## 4. PCA normal subspace and reconstruction residual — `src/models/pca_detector.py`
For each scale, all patch vectors `x ∈ ℝ^D` of the **normal** training images are standardised,
`z = (x − μ) / s` (`s` floored relative to the median feature std to protect against constant dimensions). With
`C = ZᵀZ/(N−1) = V Λ Vᵀ`, the `k` leading eigenvectors `V_k` span the normal subspace. A patch is reconstructed and its
residual measured as

```
ẑ = V_k V_kᵀ z                (⇔ x̂ = μ + S V_k V_kᵀ S⁻¹ (x − μ), S = diag(s))
r = z − ẑ
score(x) = ‖r‖₂
```

`k` is either fixed or the smallest value whose cumulative explained variance reaches a ratio (default 0.95), bounded by
`min(D, N−1)`. A patch whose appearance is not explained by the main directions of normal variation leaves a large
residual. Patch coordinates are kept, so `score_map` returns an `h × w` map that is up-sampled (bilinear) to the image
size and smoothed with a Gaussian (σ = 4 px).

## 5. Contribution B – multi-scale fusion — `src/models/multiscale_detector.py`
One PCA per scale yields raw maps `S_fine`, `S_coarse`. Because residual magnitudes of different scales are not
comparable, each is z-normalised with the mean and standard deviation of **normal calibration** pixel scores of that scale:
`z_s = (S_s − μ_s)/σ_s`. The fused map is

```
S_fused = α · z_fine + (1 − α) · z_coarse ,   0 ≤ α ≤ 1
```

`α = 1` is the single-scale baseline. The z-normalisation is applied to *all* variants so that the units are identical
(it does not change a single-scale AUROC because it is a monotone transform).

## 6. Calibration data and leakage prevention
Residuals of the images used to fit a PCA are optimistically small. We therefore calibrate with **K-fold cross-fitting**
(default K = 5): for each fold a PCA is fitted on the other folds and the held-out **normal training** images are scored.
These held-out scores supply `μ_s, σ_s` and the threshold statistics. The final detector is then fitted on all normal
training images. Test images, defective images and ground-truth masks never enter fitting or calibration. With a single
image cross-fitting is impossible and the code falls back to biased resubstitution with a warning (recorded in
`calibration_mode`). Unit tests verify that fitting works with all test data deleted and that the threshold is unchanged by
predicting test images.

## 7. Contribution C – thresholding — `src/models/thresholding.py`
Let `S_normal = {max(S_fused(I_j))}` over held-out normal calibration images `I_j`.

| Method | Threshold |
|---|---|
| fixed (baseline) | `T = T₀` (default 3.0 in z-units; category-independent) |
| mean + std | `T = mean(S_normal) + k·std(S_normal)` (default k = 3) |
| percentile | `T = P-th percentile of S_normal` (default P = 99) |

The same `T` is applied to the map: `mask = S_fused ≥ T` (optionally removing blobs smaller than `min_area`), and an image
is **anomalous iff its score ≥ T** (equivalently its mask is non-empty). These are heuristics: with few calibration images
the estimates are noisy, and maxima of anomaly maps are not Gaussian so `k` has no exact false-alarm interpretation.
Scores are **normalised anomaly scores, not calibrated probabilities**.

## 8. Evaluation — `src/evaluation/`
* **Image-level:** AUROC and average precision of the image score (max of the fused map); precision, recall, F1 and false
  positive rate at the calibrated threshold; class counts are always reported. Accuracy is deliberately not reported because
  test sets are imbalanced.
* **Pixel-level (needs ground-truth masks):** AUROC, AP, precision/recall/F1, micro-averaged over all test pixels at the
  analysis resolution (masks resized with nearest neighbour). Not reported for datasets without masks.
* **Oracle F1:** the best F1 over all thresholds *using test labels*; shown only as a reference upper bound.
* **Size study:** defects are binned by GT area fraction – small < 1 %, medium 1–5 %, large ≥ 5 % (fixed a priori). Each bin is
  evaluated against all normal test images.
* **Latency:** mean end-to-end ms/image on this machine for the scales each variant actually needs.

## 9. Experiments — `src/experiments/`
* **Variants:** A baseline; B + illumination; C + multi-scale; D + adaptive threshold; E all three. Same split, resolution,
  features, PCA setting and metric code.
* **Robustness:** the *test* images are perturbed (brightness × 0.5/1.5, gamma 0.5/2.0, linear light gradient, soft
  shadow, Gaussian noise σ = 10) while models stay fitted/calibrated on clean normal images. This tests the *whole
  deployed system including its fixed/adaptive thresholds*. Perturbations are **synthetic** and do not replace real
  lighting changes.
* Outputs: `results.csv`, `size_analysis.csv`, `summary.csv`, `summary.md`, `config.json`, plots.
