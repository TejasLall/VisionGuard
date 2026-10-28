# VisionGuard — Industrial Anomaly Inspection

**Adaptive Multi-Scale Subspace Anomaly Detection for Robust Industrial Defect Inspection**
*An illumination-aware, few-shot computer vision system for detecting and localising defects using PCA-based subspace
modelling, multi-scale analysis and adaptive statistical thresholding.*

B.Tech mini-project for *Principles of Computer Vision*. Everything here runs locally on Windows (CPU is enough).

> **Academic honesty.** The research inspiration is *SubspaceAD: Training-Free Few-Shot Anomaly Detection via Subspace
> Modeling* (Lendering, Akdag, Bondarev; CVPR 2026; arXiv:2602.23013). This repository is an **independent, simplified
> implementation inspired by that research direction – not a reproduction** of the paper, and its numbers are not comparable
> to the paper's. The three enhancements are *experimentally tested ideas*, not claimed inventions (similar ideas exist in
> the literature, see `docs/LITERATURE_REVIEW.md`). No result in this repository is hand-written: every number comes from a run
> whose output files are saved under `outputs/`.

## 1. Problem
Factories have many images of good products and few of defective ones. We learn what *normal* looks like from good images only,
flag images that deviate, and highlight where – without labelled defect examples and with a threshold derived automatically from
normal data.

## 2. Method in one paragraph
Dense patch features of normal images → PCA normal subspace (`x̂ = μ + V_k V_kᵀ(x − μ)`) → anomaly score = reconstruction
residual `‖x − x̂‖` per patch → map back to the image → binary mask by thresholding. Details: `docs/METHODOLOGY.md`.

## 3. Our three enhancements (each can be switched on/off independently)
| | Enhancement | Implementation |
|---|---|---|
| A | **Illumination-aware preprocessing** – flat-field, luminance normalisation, CLAHE (+ optional gray-world) | `src/preprocessing/illumination.py` |
| B | **Multi-scale detection** – fine (9 px window, 64² grid) and coarse (33 px window, 32² grid) maps fused as `α·fine + (1−α)·coarse` | `src/features/`, `src/models/multiscale_detector.py` |
| C | **Adaptive thresholding** – fixed (baseline), `mean + k·std`, percentile; learned from held-out *normal* data only | `src/models/thresholding.py` |

Experimental variants: **A** baseline · **B** + illumination · **C** + multi-scale · **D** + adaptive threshold · **E** full system.

## 4. Architecture
```
VisionGuard/
  app.py                  Streamlit entry point (navigation)
  app_pages/              dashboard, inspect, compare, training, evaluation pages
  .streamlit/config.toml  dashboard theme (light/dark)
  config/default.yaml     all tunables (mirrors src/config.py)
  src/app_support.py      shared Streamlit helpers (caching, model discovery, widgets)
  src/data/               image_io, dataset_loader, dataset_validator, synthetic data generator
  src/preprocessing/      illumination.py            (ImagePreprocessor)
  src/features/           handcrafted.py, deep_features.py (DINOv2), factory in __init__.py
  src/models/             pca_detector.py, multiscale_detector.py, thresholding.py, pipeline.py, model_manager.py
  src/evaluation/         metrics.py, evaluator.py   (AnomalyEvaluator)
  src/experiments/        runner.py (ExperimentRunner), robustness.py, single_eval.py, reporting.py
  src/visualization/      plots.py
  scripts/                check_environment, prepare_dataset, train, predict, evaluate, run_experiments, run_robustness
  tests/                  pytest suite
  docs/                   proposal, abstract, methodology, literature review, results, limitations, viva Q&A
  data/ models/ outputs/  (git-ignored contents) datasets, saved models, experiment results
```

## 5. Installation (Windows, PowerShell)
```powershell
cd "VisionGuard"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts/check_environment.py
```
Tested with Python 3.13. If PowerShell blocks activation run `Set-ExecutionPolicy -Scope Process Bypass` first.

**Optional deep backend (DINOv2).** Downloads PyTorch (~200 MB CPU build) and, on first use, the model weights
(`facebook/dinov2-small`, ~90 MB, from Hugging Face):
```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-deep.txt
```
Use it with `--backend dinov2` (scripts) or the "DINOv2" option in the dashboard. If weights cannot be loaded the program
stops with a clear message; with `--allow-fallback` it uses the handcrafted backend and labels it
**"FALLBACK (NOT DINOv2)"** everywhere.

## 6. Dataset setup
**Quick software test (no download):**
```powershell
python scripts/prepare_dataset.py synthetic          # writes data/synthetic (synthetic, for software testing only)
```
**MVTec AD (main benchmark).** It is *not* downloaded automatically (registration form; licence CC BY-NC-SA 4.0,
non-commercial; about 5 GB):
1. Open <https://www.mvtec.com/company/research/datasets/mvtec-ad>, fill in the form, download `mvtec_anomaly_detection.tar.xz`.
2. `python scripts/prepare_dataset.py extract C:\path\to\mvtec_anomaly_detection.tar.xz --out data\mvtec_ad`
   (checks free disk space, refuses unsafe archive paths) – or extract with 7-Zip.
3. `python scripts/prepare_dataset.py validate data\mvtec_ad`

Categories and defect folders are discovered from disk (`<cat>/train/good`, `<cat>/test/<good|defect>`,
`<cat>/ground_truth/<defect>/<id>_mask.png`). **Custom data:** `custom_dataset/normal/`, `custom_dataset/test/normal/`,
`custom_dataset/test/defective/`, optional `custom_dataset/ground_truth/<image-stem>.png`. Without masks, only image-level
metrics are reported.

## 7. Train / calibrate, predict, evaluate
```powershell
# fit on NORMAL training images only (saves <cat>__handcrafted__raw and __illum models in models/)
python scripts/train.py --dataset data/mvtec_ad --category bottle --n-train 20

# inspect one image (heat map, mask, overlay, 6-panel figure -> outputs/predictions)
python scripts/predict.py --model bottle__handcrafted__illum --image path\to\image.png --multiscale --alpha 0.5 --threshold mean_std --k 3

# score a saved model on the category's test split (ROC/PR, histograms, examples -> outputs/eval_*)
python scripts/evaluate.py --dataset data/mvtec_ad --category bottle --model bottle__handcrafted__illum --multiscale --threshold mean_std
```
Calibration: held-out normal training images (K-fold cross-fitting) – never test images or masks.

## 8. Experiments
```powershell
# five-variant comparison A-E (CSV, JSON, plots, summary.md -> outputs/<name>_<timestamp>/)
python scripts/run_experiments.py --dataset data/mvtec_ad --categories bottle hazelnut metal_nut --n-train 20 --name mvtec_ablation

# synthetic illumination robustness + defect-size study
python scripts/run_robustness.py --dataset data/mvtec_ad --categories bottle hazelnut --n-train 20

# which illumination operation helps? (CLAHE / flat-field / luminance-norm / auto-CLAHE / all)
python scripts/run_robustness.py --dataset data/mvtec_ad --categories bottle --op-ablation

# software smoke test on synthetic data (NOT a benchmark)
python scripts/run_experiments.py --synthetic
```
Add `--backend dinov2` to use the deep features (slower on CPU). Seeds are fixed (`--seed`, default 0).
Results of the runs made for this repository are summarised in `docs/EXPERIMENTAL_RESULTS.md`.

## 9. Dashboard
```powershell
python -m streamlit run app.py      # opens http://localhost:8501
```
Pages: **Dashboard**, **Inspect an Image** (upload, choose model, preprocessing/scale/threshold, view original · preprocessed ·
heat map · mask · overlay · scores, download PNGs), **Compare Methods** (A–E side by side), **Dataset & Training** (validate a
dataset folder, fit and save models, reload), **Evaluation** (metrics, ROC/PR, per-category tables, latency, robustness and
size results from saved experiment folders).

## 10. Tests
```powershell
python -m pytest
```
Covers image loading and invalid files, preprocessing, PCA maths, model save/load, map dimensions, thresholds, masks, metrics,
dataset validation, leakage guards and experiment output generation (using the synthetic dataset).

## 11. Troubleshooting
| Symptom | Fix |
|---|---|
| `ModuleNotFoundError` | activate the venv, `pip install -r requirements.txt` |
| pip "no matching distribution" on a slow network | re-run with `pip install --retries 10 --timeout 60 -r requirements.txt` |
| "No MVTec-style categories found" | point at the folder *containing* `bottle/`, `cable/`, ...; run `prepare_dataset.py validate` |
| "DINOv2 backend needs torch…" | install the optional packages (section 5) or use the handcrafted backend |
| Dashboard says no model | fit one on the *Dataset & Training* page or with `scripts/train.py` |
| Many false positives with 2–5 training images | expected: few images → noisy PCA and thresholds; use more normal images |
| Slow runs | lower `--image-size`, use `--n-train`, fewer categories |

## 12. Known limitations
PCA is linear/global; handcrafted features are weak; illumination normalisation can hide defects; thresholds from few images
are noisy; the synthetic lighting tests are not real lighting; MVTec is controlled and imbalanced. Full discussion in
`docs/LIMITATIONS.md`.

## 13. Licences and data
Code: for coursework use. MVTec AD is © MVTec Software GmbH under CC BY-NC-SA 4.0 (non-commercial); it is not included in this
repository. Please cite the dataset and the SubspaceAD paper if you use them (see `docs/LITERATURE_REVIEW.md`).
