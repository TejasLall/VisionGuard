# Experimental Results

Every number in this document was copied from a CSV/markdown file produced by a run on 2026-10-29 (folders under
`outputs/`). Nothing was estimated or typed in by hand. Rows marked **Not run** have no measurements.

## 0. Status

| Experiment | Dataset | Status | Output folder |
|---|---|---|---|
| Five-variant ablation, handcrafted backend | **MVTec AD** | **Not run.** The dataset was not available locally, and the official download needs a registration form that only you can complete | — |
| Robustness and defect-size study | **MVTec AD** | **Not run** (same reason) | — |
| Five-variant ablation, handcrafted backend | synthetic (software test) | run | `outputs/synthetic_ablation_20261029_030734` |
| Robustness study, handcrafted backend | synthetic | run | `outputs/synthetic_robustness_20261029_031449` |
| Illumination-operation ablation | synthetic | run | `outputs/synthetic_opablation_20261029_032945` |
| Five-variant ablation, DINOv2-small backend | synthetic | run | `outputs/synthetic_dinov2_20261029_033913` |

> **Read this before quoting anything below.** The synthetic dataset (`src/data/synthetic.py`) has two procedurally
> generated categories with obvious defects. It exists to check that the software is correct end to end. It is
> almost trivially separable (every variant reaches image AUROC 1.000 on clean images), so these results **say nothing
> about real industrial performance and cannot rank the methods on real data.** The robustness results are more
> informative, but they still describe *synthetic perturbations of synthetic images*.

**Common setup:** analysis resolution 256 px; PCA at 0.95 explained variance; 5-fold cross-fitted calibration on
normal training images; Gaussian smoothing σ = 4 px; α = 0.5; fixed T = 3.0 (z-units); adaptive T = mean + 3·std;
seed 0. Synthetic data per category: 24 normal training images, and a test set of 12 normal + 24 defective images
(8 scratch, 8 spot, 8 patch), all with masks.

## 1. Five-variant ablation, synthetic, handcrafted backend (mean of 2 categories)

| Variant | Image AUROC | Image P | Image R | Image F1 | Image FPR | Pixel AUROC | Pixel AP | Pixel P | Pixel R | Pixel F1 | Pixel F1 (oracle) | ms / image | Mean T |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A Baseline | 1.000 | 0.667 | 1.000 | 0.800 | 1.000 | 1.000 | 0.976 | 0.261 | 1.000 | 0.412 | 0.918 | 139 | 3.000 |
| B + illumination | 1.000 | 0.667 | 1.000 | 0.800 | 1.000 | 1.000 | 0.968 | 0.272 | 0.999 | 0.427 | 0.895 | 172 | 3.000 |
| C + multi-scale | 1.000 | 0.667 | 1.000 | 0.800 | 1.000 | 0.999 | 0.954 | 0.231 | 1.000 | 0.371 | 0.886 | 159 | 3.000 |
| D + adaptive threshold | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 0.976 | 0.613 | 0.993 | 0.748 | 0.918 | 145 | 12.888 |
| E Full system | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.999 | 0.938 | 0.395 | 0.992 | 0.565 | 0.868 | 179 | 7.380 |

Class distribution: 24 normal / 48 defective test images in total (anomalous fraction 0.67).
Pixel-level anomalous fraction is small, which is why pixel AP and F1 are much lower than pixel AUROC.

**What this shows (synthetic data only):**
* AUROC cannot separate the variants here because every variant ranks the images perfectly.
* The **fixed threshold T = 3.0 is far too low** for these maps: every normal test image is flagged (FPR = 1.0).
  The adaptive threshold, calibrated on held-out normal training images, lands between 4.75 and 15.5 z-units depending on the category and variant and gives
  FPR = 0 with full recall. Of the three contributions, **this is the only one with a measurable benefit in the clean
  setting.** It also shows the weakness of a category-independent constant.
* Multi-scale fusion (C, E) did **not** improve pixel F1 or pixel AP on this data; they were slightly lower.
* Model files are 4.4–4.5 MB each, mostly the stored float16 calibration maps.

## 2. Same ablation with the DINOv2-small backend (synthetic, 10 training images per category)

| Variant | Image AUROC | Image F1 | Image FPR | Pixel AUROC | Pixel AP | Pixel F1 | Pixel F1 (oracle) | ms / image | Mean T |
|---|---|---|---|---|---|---|---|---|---|
| A | 1.000 | 0.807 | 0.958 | 0.999 | 0.903 | 0.153 | 0.877 | 1526 | 3.000 |
| B | 1.000 | 0.800 | 1.000 | 0.999 | 0.918 | 0.144 | 0.872 | 1598 | 3.000 |
| C | 1.000 | 0.829 | 0.833 | 0.997 | 0.715 | 0.039 | 0.780 | 1579 | 3.000 |
| D | 1.000 | 1.000 | 0.000 | 0.999 | 0.903 | 0.353 | 0.877 | 1518 | 10.291 |
| E | 1.000 | 0.990 | 0.042 | 0.997 | 0.707 | 0.246 | 0.765 | 1574 | 8.403 |

This run is **evidence that the deep backend works end to end**: pretrained `facebook/dinov2-small` loaded from
Hugging Face, with 32 × 32 × 384 patch tokens at a 448 px input on CPU (PyTorch 2.15 CPU build). It is not a
performance claim. DINOv2 was about 10× slower per image than the handcrafted backend on this CPU. As with the
handcrafted backend, the coarse-scale fusion lowered pixel AP here.

## 3. Robustness to synthetic illumination and noise (handcrafted, mean of 2 categories)

Models are fitted and calibrated on clean normal images; only the **test** images are perturbed.

**Image AUROC** (Δ relative to identity in brackets):

| Transform | A | B (+illum) | C (+multi) | D (+thr) | E (full) |
|---|---|---|---|---|---|
| identity | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| dark ×0.5 | 0.851 (−0.149) | **0.943** (−0.057) | 0.865 | 0.851 | 0.875 |
| bright ×1.5 | 0.995 | 0.974 | 0.939 | 0.995 | 0.934 |
| gamma 0.5 | 1.000 | 1.000 | 0.995 | 1.000 | 1.000 |
| gamma 2.0 | 0.672 (−0.328) | **0.799** (−0.201) | 0.727 | 0.672 | **0.807** |
| uneven gradient | 0.780 | **0.835** | 0.800 | 0.780 | 0.800 |
| shadow | 0.795 | **0.922** | 0.800 | 0.795 | 0.905 |
| noise σ = 10 | **0.843** | 0.422 (−0.578) | 0.783 | **0.843** | 0.451 |

**Pixel AUROC** (selected): dark ×0.5 → A 0.963, B 0.998; gamma 2.0 → A 0.922, B 0.982; noise → A 0.982, B 0.468, E 0.317.

**Image FPR at the calibrated threshold** (D and E use the adaptive threshold): identity 0.000 / 0.000; dark 0.500 / 0.500;
bright 0.500 / 0.375; gamma 2.0 1.000 / 1.000; shadow 0.458 / 1.000; noise 1.000 / 1.000. A, B and C (fixed T = 3)
had FPR = 1.000 under every transform.

**Findings (synthetic):**
1. Illumination-aware preprocessing **reduced the AUROC loss** under darkening, gamma 2.0, uneven lighting and shadows.
2. It **badly hurt robustness to sensor noise** (image AUROC 0.843 → 0.422, pixel AUROC 0.982 → 0.468). This is
   consistent with CLAHE and contrast stretching amplifying noise. It is a concrete example of the risk listed in
   `LIMITATIONS.md`, so the preprocessing must not be applied blindly.
3. Thresholds calibrated on clean data **do not transfer** to shifted lighting: FPR rises from 0 to between 0.375 and 1.0. Adaptive
   thresholding fixes the calibration problem only for the conditions it was calibrated under.
4. Multi-scale fusion gave mixed, mostly small changes.

## 4. Which illumination operation matters? (synthetic, single scale, adaptive threshold, image AUROC)

| Transform | P0 none | P1 CLAHE | P2 flat-field | P3 lum-norm | P4 CLAHE (auto) | P5 all |
|---|---|---|---|---|---|---|
| dark ×0.5 | 0.851 | 0.837 | 0.783 | 0.948 | **0.962** | 0.943 |
| bright ×1.5 | 0.995 | 0.993 | 0.983 | **1.000** | 0.993 | 0.974 |
| gamma 0.5 | **1.000** | 0.924 | 0.993 | **1.000** | 0.931 | **1.000** |
| gamma 2.0 | 0.672 | 0.675 | 0.729 | 0.741 | 0.741 | **0.799** |
| uneven gradient | 0.780 | 0.832 | 0.825 | 0.726 | **0.915** | 0.835 |
| shadow | 0.795 | 0.889 | 0.851 | 0.825 | 0.727 | **0.922** |
| noise σ = 10 | **0.843** | 0.681 | 0.762 | 0.700 | 0.774 | 0.422 |

No single operation wins everywhere. Luminance normalisation does most of the work for global brightness changes;
flat-field plus CLAHE helps with shadows; every operation lowers the noise robustness, and the combination lowers it most.
Choosing operations from these numbers would itself be tuning on (synthetic) test data, so the defaults were **not**
changed after seeing them.

## 5. Small versus large defects (synthetic, identity transform)

Bins are fixed by ground-truth area: small < 1 %, medium 1–5 %, large ≥ 5 %.

| Bin | n defective | Pixel recall at T: A / C / D / E | Image AUROC (all variants) |
|---|---|---|---|
| small | 32 | 0.999 / 0.999 / 0.973 / 0.942 | 0.998–1.000 pixel, 1.000 image |
| medium | 16 | 1.000 / 1.000 / 0.996 / 1.000 | 1.000 |
| large | 0 | — (the synthetic generator produced no defects ≥ 5 %) | — |

On this data, multi-scale did **not** improve small-defect detection, because the baseline already finds them all.
The higher adaptive threshold (D, E) slightly lowers pixel recall on small defects, which is the expected
precision/recall trade-off. A meaningful size study needs MVTec AD.

## 6. MVTec AD results (template – Not run)

Run, for example:
`python scripts/run_experiments.py --dataset data/mvtec_ad --categories bottle cable hazelnut metal_nut screw --n-train 20 --name mvtec_ablation`
and `python scripts/run_robustness.py --dataset data/mvtec_ad --categories bottle hazelnut --n-train 20`.
Then copy the numbers from `summary.md` into these tables.

| Category | Variant | Image AUROC | Pixel AUROC | Image P / R / F1 | Image FPR | Pixel P / R / F1 | ms / image |
|---|---|---|---|---|---|---|---|
| bottle | A–E | Not run | Not run | Not run | Not run | Not run | Not run |
| cable | A–E | Not run | Not run | Not run | Not run | Not run | Not run |
| hazelnut | A–E | Not run | Not run | Not run | Not run | Not run | Not run |
| metal_nut | A–E | Not run | Not run | Not run | Not run | Not run | Not run |
| screw | A–E | Not run | Not run | Not run | Not run | Not run | Not run |

## 7. Reproducing these results

```powershell
python scripts/run_experiments.py --synthetic --name synthetic_ablation
python scripts/run_robustness.py  --synthetic --name synthetic_robustness
python scripts/run_robustness.py  --synthetic --op-ablation --name synthetic
python scripts/run_experiments.py --synthetic --backend dinov2 --n-train 10 --name synthetic_dinov2
```
Seeds are fixed. The handcrafted ablation was run twice and gave identical metrics. Latencies depend on the machine
and on background load.
