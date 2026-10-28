# Literature Review

> **Scope and honesty note.** This is a *targeted* review written during project setup (searches on
> 2026-10-29), not a systematic survey. Bibliographic details marked **[verified]** were checked against the
> arXiv/venue page during this session. Items marked **[not re-verified]** are well-known references quoted from
> general knowledge or from a search-result snippet; check them before putting them in a formal report.
> Numbers quoted from papers are **their** reported findings, never ours.

## 1. The reference paper: SubspaceAD

**Lendering, C., Akdag, E., Bondarev, E.** *SubspaceAD: Training-Free Few-Shot Anomaly Detection via Subspace Modeling.*
CVPR 2026. arXiv:2602.23013 (v1 26 Feb 2026, latest v3 13 May 2026), DOI 10.48550/arXiv.2602.23013. **[verified]**
Official code: <https://github.com/CLendering/SubspaceAD> (found via search; not used by this project).

* **Problem.** Industrial inspection with only a few normal images per category (k = 1, 2, 4 shots).
* **Method (as described in the abstract).** Extract patch-level features from a frozen DINOv2 backbone, fit a PCA
  model to the normal patches to estimate the subspace of normal variation, and score a test patch by its
  **reconstruction residual** against that subspace. No memory bank, no auxiliary data, no vision-language tuning.
* **Reported findings (theirs).** The abstract of the version we read reports one-shot results of 97.1 % image-level /
  97.5 % pixel-level AUROC on MVTec AD and 93.2 % / 98.2 % on VisA. Reported numbers changed between arXiv versions
  (the revision "corrects the AU-PRO evaluation"), so always cite the version you read.
* **Not verified by us.** Exact DINOv2 variant, layers, PCA variance target, image-level aggregation and input
  resolution. We did not read the full text in enough detail to rely on them, so **this project does not claim to
  reproduce them**.
* **Limitations (general, not from the paper).** A single linear subspace per category cannot represent multi-modal
  normal appearance; performance depends on the frozen backbone; sensitivity to nuisance factors such as lighting is
  not addressed by the core method.
* **Relevance.** It is the inspiration for our PCA-residual detector. Our baseline is a **simplified, independently
  written implementation inspired by the research direction**, with an optional DINOv2 backend and a default
  CPU-only handcrafted backend. It is **not an exact reproduction** and our numbers are not comparable to theirs.

## 2. Closely related prior work (important for novelty)

| Work | What it does | Relation to ours |
|---|---|---|
| **Li, "Physics- and Statistics-Guided Normal Subspace Purification for Training-Free Few-Shot Industrial Anomaly Detection"**, IET Image Processing 2026, DOI 10.1049/ipr2.70493 **[not re-verified: from a search snippet]** | Builds on SubspaceAD; suppresses specular/illumination disturbances in features and uses a robust (MAD-based) PCA; evaluated on MVTec AD and VisA. | Prior work already targets illumination-robust subspace anomaly detection, with a more sophisticated feature-space method. **We therefore cannot claim that illumination robustness for subspace AD is new.** |
| **Han, Chu, Nowak, Chin, "SPARC: Subspace Position-Aware Robust Few-Shot Calibration for Distribution-Shifted Industrial Anomaly Detection"**, arXiv:2608.18585, 19 Aug 2026 **[verified: abstract]** | Few-shot calibration (≤ 8 verified-normal images) estimates deployment-time nuisance (lighting, fixture, sensor) per spatial patch cell and projects it out of patch features; gains reported on shift-prone benchmarks, small/mixed on benchmarks without engineered shift. | Addresses distribution shift (including illumination) in feature space. The abstract mentions neither multi-scale processing nor threshold adaptation. Its note that gains can be small without real shift is a useful caution for our own results. |
| **RadioCore** (CVPR 2026 Workshops, VISION26) **[not re-verified]** | Training-free few-shot segmentation with a *multi-scale* memory bank and kNN matching. | Multi-scale few-shot anomaly detection already exists; our multi-scale idea is not new. |
| **FastRecon**, Fang et al., ICCV 2023 **[not re-verified]** | Few-shot feature reconstruction for anomaly detection. | Related reconstruction-based few-shot approach. |

**Consequence for our claim.** Illumination-aware processing, multi-scale analysis and statistical thresholds are each
established ideas. We found no paper (in this *limited* search) that evaluates exactly the simple, image-space,
CPU-friendly combination used here in a controlled ablation, but absence in a short search is not evidence of
novelty. The project's contribution is therefore stated as a **reproducible, honest ablation and robustness study**
of three well-motivated enhancements on top of a PCA-residual baseline, not as a new state-of-the-art method.

## 3. Foundational and supporting papers

* **Bergmann, P., Fauser, M., Sattlegger, D., Steger, C.** *MVTec AD – A Comprehensive Real-World Dataset for
  Unsupervised Anomaly Detection.* CVPR 2019; extended in *The MVTec Anomaly Detection Dataset…*, IJCV, DOI
  10.1007/s11263-020-01400-4. Dataset page: <https://www.mvtec.com/company/research/datasets/mvtec-ad>. **[page verified:
  >5,000 images, 15 categories, pixel-precise annotations, CC BY-NC-SA 4.0 non-commercial licence; author list not
  re-verified]**
  *Relevance:* our benchmark. Only normal images exist in `train/`; test images carry defect labels and masks, which
  we use for evaluation only.
* **Defard, T., Setkov, A., Loesch, A., Audigier, R.** *PaDiM: a Patch Distribution Modeling Framework for Anomaly
  Detection and Localization.* ICPR 2020 workshop (1st Intl. Workshop on Industrial Machine Learning), arXiv:2011.08785.
  **[verified]**
  *Method:* pretrained-CNN patch embeddings, per-position multivariate Gaussians, Mahalanobis scoring.
  *Limitation:* assumes aligned images for position-specific statistics; memory grows with feature dimension.
  *Relevance:* conceptual ancestor of "model the normal feature distribution, score deviations"; PCA residual is the
  residual-subspace counterpart of the Mahalanobis distance.
* **Roth, K., Pemula, L., Zepeda, J., Schölkopf, B., Brox, T., Gehler, P.** *Towards Total Recall in Industrial
  Anomaly Detection (PatchCore).* CVPR 2022, arXiv:2106.08265. **[verified]**
  *Method:* memory bank of representative nominal patch features with nearest-neighbour scoring; reports up to 99.6 %
  image AUROC on MVTec AD (their finding). *Limitation:* memory/lookup cost. *Relevance:* standard strong baseline;
  also uses local neighbourhood aggregation and Gaussian smoothing of score maps (which we mirror with
  `smooth_sigma`).
* **Damm, S., Laszkiewicz, M., Lederer, J., Fischer, A.** *AnomalyDINO: Boosting Patch-based Few-shot Anomaly Detection
  with DINOv2.* WACV 2025 (oral), arXiv:2405.14529. **[verified]**
  *Method:* training-free patch-level nearest-neighbour matching with DINOv2 features for one-/few-shot detection and
  segmentation. *Relevance:* motivates using DINOv2 patch tokens; SubspaceAD positions itself against it.
* **Zuiderveld, K.** *Contrast Limited Adaptive Histogram Equalization.* In *Graphics Gems IV*, Academic Press, 1994,
  pp. 474–485. **[not re-verified]** *Relevance:* the CLAHE algorithm used in Contribution A (OpenCV
  implementation).
* **Oquab, M. et al.** *DINOv2: Learning Robust Visual Features without Supervision.* arXiv:2304.07193 (2023).
  **[not re-verified]** *Relevance:* the pretrained backbone family used by the optional deep backend
  (`facebook/dinov2-small` from the Hugging Face hub).

## 4. Thresholding from normal data

Deriving an alert threshold as `mean + k·σ` of anomaly scores on normal validation data, or as a high percentile of those
scores, is common engineering practice (search results included informal guidance with k = 2–3 and 95th/99th
percentiles, and a caveat that the Gaussian assumption is often violated, favouring percentile or robust MAD variants).
We found no authoritative industrial benchmark that prescribes a universal k, so **our default k = 3 and percentile = 99
are arbitrary, configurable choices**, and their behaviour is *measured* in our experiments, not assumed.

## 5. Gap addressed by this project

1. A transparent, from-scratch PCA-residual baseline that runs on a student laptop CPU (handcrafted features) and
   optionally with DINOv2 patch tokens.
2. A **fair ablation** (A–E) of three enhancements under identical data splits and evaluation rules.
3. A **controlled synthetic-illumination robustness test** and a **defect-size analysis**, reported with all failures.

Whether any enhancement helps is an empirical question answered only in `EXPERIMENTAL_RESULTS.md`.

## 6. Reported findings versus our own experiments

| Statement | Source |
|---|---|
| SubspaceAD one-shot AUROC 97.1 % / 97.5 % (MVTec AD) | **The paper's abstract (v3)** |
| PatchCore up to 99.6 % image AUROC | **The paper's abstract** |
| Any number in `EXPERIMENTAL_RESULTS.md` | **Our own runs**, with our simplified methods; not comparable to the above |
