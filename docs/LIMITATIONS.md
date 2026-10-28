# Limitations

Honest limitations of the design and of what has been measured. Where a statement depends on experiments, it is phrased
as a risk to be checked in `EXPERIMENTAL_RESULTS.md`, not as a finding.

## Method limitations
* **PCA is linear and global.** One subspace per scale (fitted on all patches of a category) cannot represent multi-modal
  normal appearance, and unlike position-specific models (e.g. PaDiM) it ignores *where* a patch is. Objects with strict
  geometry (cable, transistor) may have defects that are only "wrong arrangement" of normal-looking patches; a patch-wise
  residual cannot see these.
* **False positives.** Normal variation absent from the few training images (new pose, harmless texture variation, dust)
  leaves a large residual. Small normal sets make this worse. The false-positive rate is reported (`image_fpr`) so it cannot
  be hidden behind AUROC.
* **Handcrafted features are weak.** Colour/gradient/LBP statistics carry no semantic information; subtle or semantic
  defects (wrong component, missing part) are expected to be difficult. The optional DINOv2 backend is stronger in
  principle but was only exercised as far as stated in `EXPERIMENTAL_RESULTS.md`.
* **Dependence on pretrained features (deep backend).** Quality is bounded by the frozen backbone; domain shift between
  ImageNet-like pre-training data and industrial images is possible; the layer choices are ours and untuned.

## Illumination-aware preprocessing
* **Normalisation can hide defects.** CLAHE amplifies local contrast (also noise and compression artefacts), flat-fielding
  can remove genuine low-frequency defects (large dents, gradual discolouration) and create halos at strong edges, and
  gray-world assumes the average colour is neutral. Such operations may *reduce* performance on some categories – the
  per-operation ablation exists to check this.
* **Measured example (synthetic data):** with Gaussian sensor noise (σ = 10), the full preprocessing chain dropped image
  AUROC from 0.843 to 0.422, because CLAHE and contrast normalisation amplify the noise (`EXPERIMENTAL_RESULTS.md`, §3–4).
* **Thresholds do not transfer across lighting.** Thresholds calibrated on clean normal images produced false-positive rates
  between 0.375 and 1.0 under the synthetic lighting shifts (§3). Recalibrate whenever the imaging conditions change.
* **Synthetic ≠ real.** Our robustness test multiplies pixels by smooth fields. Real lighting changes also move specular
  highlights, change shadows' geometry and interact with surface reflectance; passing the synthetic test is no guarantee for
  a factory floor.
* Luminance statistics for normalisation are learned from the (few) normal training images and include any variation they
  contain.

## Multi-scale fusion
* `α` is a hand-set hyper-parameter (default 0.5), not tuned per category; tuning it on test data would be leakage. Z-normalised
  scales assume that calibration statistics generalise to test images.
* More scales cost more compute (measured as latency) and add one more source of false positives.

## Thresholding
* `mean + k·std` and percentiles estimated from a **small** number of calibration images are noisy; the 99th percentile of
  20 numbers is essentially the maximum. `k` and the percentile are not derived from a target false-alarm rate – the score
  distribution is skewed, not Gaussian.
* Cross-fitting calibrates with models trained on (K−1)/K of the data, so thresholds are slightly conservative; with one
  image only biased resubstitution is possible.
* A single threshold is used for both the image decision and the pixel mask; the pixel threshold derived from per-image
  *maxima* is high, so pixel recall at the calibrated threshold is typically lower than the best achievable (oracle) F1.
* A fixed threshold does not transfer between categories or lighting conditions – that is the point of the comparison.

## Data and evaluation
* **Dataset bias.** MVTec AD has controlled lighting, aligned objects and single-category training; success there does not
  imply success on real production lines. Test sets are imbalanced (many more defective than normal images in most categories),
  so accuracy would be misleading and is not reported.
* **Domain shift.** Models are category-specific and fitted on a handful of images; a changed camera, lens or product
  revision requires refitting.
* Pixel metrics are computed at 256 × 256, not at the original resolution; very thin defects lose area when masks are
  down-sampled.
* The synthetic dataset is trivially separable and exists only to test the software.
* The MVTec AD experiment is only as complete as stated in `EXPERIMENTAL_RESULTS.md` (the dataset is not auto-downloaded and
  requires a manual form + licence acceptance).

## Runtime and hardware
* Handcrafted backend: CPU only, tens of milliseconds per image at 256 px on a laptop (measured values in the results).
* DINOv2 backend: needs PyTorch + `transformers` + a model download; CPU inference is noticeably slower and memory usage
  scales with the number of training patches (`max_patches` caps this).
* The default environment of this project was a slow network, which is why the deep backend is optional.
