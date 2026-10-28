# Project Proposal

## Title
**Adaptive Multi-Scale Subspace Anomaly Detection for Robust Industrial Defect Inspection**
*An illumination-aware, few-shot computer vision system for detecting and localising defects using pretrained visual
features, PCA-based subspace modelling, multi-scale analysis and adaptive statistical thresholding.*

Course: Principles of Computer Vision (B.Tech). Application name: **VisionGuard – Industrial Anomaly Inspection**.

## 1. Introduction
Visual inspection is a core quality-control task. Manufacturers typically have thousands of images of good parts but only a
handful of defective ones, and new defect types appear without warning. This project builds a working inspection system that
learns what "normal" looks like from good images only and highlights regions that deviate from it.

## 2. Problem statement
Given a small set of normal images of a product, (i) decide whether a new image is defective, (ii) localise the suspicious
region, and (iii) do so reliably under lighting changes, for defects of different sizes, with a decision threshold derived
automatically from normal data.

## 3. Objectives
1. Implement a transparent PCA-residual anomaly detector (baseline) in modular, tested Python.
2. Implement three enhancements – illumination-aware preprocessing, multi-scale fusion, adaptive thresholding – that can be
   enabled independently.
3. Provide a CPU handcrafted-feature backend and an optional pretrained DINOv2 backend.
4. Evaluate on MVTec AD (image- and pixel-level) with a fair five-variant ablation, a synthetic illumination-robustness
   study and a defect-size study.
5. Deliver a Streamlit dashboard, reproducible scripts and honest documentation.

## 4. Research motivation
*SubspaceAD* (Lendering, Akdag, Bondarev; CVPR 2026, arXiv:2602.23013) shows that frozen DINOv2 patch features plus a PCA
subspace give strong training-free few-shot detection. Open practical questions that a student project can study with
controlled experiments: is the approach sensitive to lighting; do multiple feature scales help small defects; and how should the
decision threshold be set without labelled defects? Related work already studies shift-robust subspace methods (e.g. SPARC,
arXiv:2608.18585) and multi-scale few-shot methods, so we **do not** claim novelty for the individual ideas (see
`LITERATURE_REVIEW.md`).

## 5. Proposed contributions (hypotheses to be tested, not results)
* **A – Illumination-aware preprocessing** (flat-field, luminance normalisation, CLAHE, optional gray-world).
  *Hypothesis:* improves robustness to brightness/shading changes. *Risk:* may amplify noise or hide defects.
* **B – Multi-scale detection** (fine 9 px/stride 4 and coarse 33 px/stride 8 windows, `S = α·S_fine + (1−α)·S_coarse`).
  *Hypothesis:* helps both small and large defects. *Risk:* extra false positives and compute.
* **C – Adaptive statistical thresholding** (`mean + k·std` or percentile of held-out normal scores).
  *Hypothesis:* gives a better-balanced precision/recall than a fixed constant across categories and shifts.
  *Risk:* noisy with few calibration images; Gaussian assumption violated.

## 6. Methodology (summary; details in `METHODOLOGY.md`)
Load → validate → resize → [A] → dense features (two scales) → PCA per scale (`x̂ = μ + V_k V_kᵀ(x−μ)`, residual score) →
upsample, smooth, z-normalise → [B] fuse → [C] threshold → mask and overlay. Calibration uses K-fold cross-fitting on normal
training images; test data and masks are used for evaluation only.

## 7. Tools and technologies
Python 3.13, NumPy, SciPy, OpenCV, scikit-learn, scikit-image, Pillow, Matplotlib, pandas, PyYAML, Streamlit, pytest;
optional PyTorch + Hugging Face `transformers` for DINOv2. Runs locally on Windows; no Docker, database or cloud services.

## 8. Expected deliverables
Source code and tests; Streamlit dashboard; scripts for training, prediction, evaluation, experiments, robustness;
CSV/JSON result files and plots; README; this proposal, abstract, methodology, literature review, limitations, results
report and viva questions.

## 9. Evaluation plan
* **Data:** MVTec AD categories chosen by the user (e.g. bottle, cable, hazelnut, metal_nut, screw); a synthetic dataset for
  software testing; optional custom datasets.
* **Variants:** A baseline, B +illumination, C +multi-scale, D +adaptive threshold, E full system – identical split,
  resolution, features and metrics.
* **Metrics:** image AUROC/AP; precision/recall/F1/FPR at the calibrated threshold; pixel AUROC/AP/P/R/F1 (masks required);
  latency; model size; class distribution reported with every table.
* **Robustness:** synthetic brightness, gamma, uneven-illumination, shadow and noise perturbations of test images.
* **Size study:** small < 1 %, medium 1–5 %, large ≥ 5 % of the image area.
* **Leakage rules:** fit and calibrate on normal training images only; no test-driven choices of thresholds or α.

## 10. Expected outcomes versus measured results
*Expected outcomes* are the hypotheses above. *Measured results* appear only in `EXPERIMENTAL_RESULTS.md` and are copied
from real run outputs. Where an experiment has not been run it is marked "Not run".
