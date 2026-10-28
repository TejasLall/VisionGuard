# Abstract

Industrial manufacturers possess abundant images of normal products but few examples of defects, which makes supervised
defect classification impractical and fails on previously unseen defect types. We study few-shot, training-free anomaly
detection and localisation using a normal-subspace model: dense patch features of normal images are modelled by principal
component analysis (PCA), and a test patch is scored by its reconstruction residual against that subspace. Our baseline is
an independently implemented, simplified method inspired by the 2026 SubspaceAD approach (not a reproduction of it), with a
CPU-friendly handcrafted feature backend and an optional pretrained DINOv2 backend. On top of this baseline we investigate
three enhancements and evaluate them in a controlled ablation: (A) illumination-aware preprocessing based on flat-field
correction, luminance normalisation and CLAHE; (B) multi-scale anomaly maps built from features with different receptive
fields and fused with a calibrated weighted sum; and (C) adaptive thresholds estimated from held-out normal calibration
scores, either as mean plus k standard deviations or as an empirical percentile, instead of a fixed constant. Models and
thresholds use normal training images only, and ground-truth masks are used solely for evaluation. We report image- and
pixel-level AUROC, precision, recall, F1, false-positive rate and latency on MVTec AD categories, together with a
controlled robustness study using synthetic brightness, gamma, shadow and noise perturbations and an analysis by defect
size. Whether each enhancement helps, and at what cost, is determined empirically and reported together with its failures
and limitations.
