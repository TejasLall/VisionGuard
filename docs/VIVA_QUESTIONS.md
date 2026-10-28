# Viva Questions and Answers

Answers are written so you can explain them in your own words. Where an answer depends on experiments, look up the actual
numbers in `EXPERIMENTAL_RESULTS.md` – do not quote numbers you did not measure.

**1. What is computer vision?**
The field that makes computers extract useful information from images and video – for example detecting objects, measuring
shapes or finding defects. Our project is an industrial-inspection application: image in, "defective or not and where" out.

**2. What is anomaly detection?**
Finding samples that do not fit the pattern of "normal" data. In our case an anomaly is a patch or image that looks unlike the
good product images we learned from.

**3. Supervised versus unsupervised learning – which is ours?**
Supervised learning needs labelled examples of every class (here: many defect photos). Unsupervised/one-class learning uses only
normal data. We use only normal images to fit the model, so it is unsupervised for training; the labels of test images are
used *only to measure* how well it works.

**4. Why not just train a defect classifier?**
Defective samples are rare, new defect types appear, and a classifier can only recognise types it has seen. A normal-only model
can flag anything unusual, including unseen defects (at the price of more false alarms).

**5. What are visual features?**
Numbers that describe image content, e.g. colour, edges, texture or the outputs of a neural network. Our default features per
small window are Lab colour, edge strengths (Sobel, Laplacian), Local Binary Pattern texture histograms and local standard
deviations (19 numbers). The optional deep features are DINOv2 patch tokens.

**6. What is PCA?**
Principal Component Analysis finds orthogonal directions along which the data varies most. Keeping the top `k` directions gives
a compact subspace that captures the dominant variation of the data.

**7. Eigenvectors and eigenvalues – how do they appear here?**
We compute the covariance matrix `C` of the (standardised) normal patch features. Its eigenvectors are the principal directions;
the eigenvalues say how much variance each direction explains. We keep the largest ones until about 95 % of the variance is
explained.

**8. What is dimensionality reduction and why use it?**
Representing data with fewer numbers while keeping what matters. Here it defines the "normal subspace": normal patches lie close
to it, anomalous patches do not.

**9. What is reconstruction error / residual?**
Project a feature onto the subspace and map it back: `x̂ = μ + V_k V_kᵀ (x − μ)`. The residual `r = x − x̂` is what the subspace
cannot explain. Its length `‖r‖` is the anomaly score: small for normal patches, large for unusual ones.

**10. Why do you standardise the features?**
Features have different units and ranges. Without standardisation the largest-valued feature would dominate the PCA. We subtract
the training mean and divide by the training standard deviation (with a floor to avoid dividing by ~0).

**11. What is an anomaly heatmap?**
The patch scores placed back at their image positions, up-sampled and smoothed, so each pixel has a score; hot colours mean
"more unlike normal". Thresholding it gives the binary defect mask.

**12. Why use DINOv2, and is it used by default?**
DINOv2 is a self-supervised vision transformer producing rich patch descriptors without task-specific training, which suits
few-shot detection; the SubspaceAD paper builds on it. In our project it is an *optional* backend. The default is the handcrafted
CPU backend, and the code never pretends the fallback is DINOv2 (it is labelled everywhere).

**13. What is CLAHE?**
Contrast Limited Adaptive Histogram Equalisation: the image is divided into tiles, each tile's histogram is equalised to
increase local contrast, and a clip limit prevents noise from being over-amplified. We apply it to the lightness channel only.

**14. What is illumination normalisation and why can it hurt?**
Making images look as if they were lit the same way (flat-field division, matching brightness statistics, CLAHE). It can hurt
because it may amplify noise or remove real defects that look like lighting changes (e.g. a smooth discolouration). We therefore
measure each operation separately.

**15. Why multiple scales?**
Small defects (scratches) need fine, small-window features; large defects (big dents) are better captured by large-window
features. Our scales differ in receptive field (9 px vs 33 px) and grid resolution, not just image size. We fuse them as
`S = α·S_fine + (1−α)·S_coarse` after z-normalising each map.

**16. How does adaptive thresholding work?**
We score held-out normal images, take each image's maximum anomaly score, and set `T = mean + k·std` (or a percentile). Images or
pixels above `T` are flagged. The threshold therefore adapts to each category's normal score distribution; a fixed constant does not.

**17. Why not set the threshold using the defective test images?**
That would be test-set leakage: the reported performance would be optimistic because the threshold was tuned on the answers. We
only use normal training images (cross-fitted so they are *held out* from the model that scores them).

**18. Image-level versus pixel-level AUROC?**
Image-level AUROC measures whether defective images get higher scores than normal ones (one score per image). Pixel-level AUROC
measures whether defective pixels get higher scores than normal pixels (needs ground-truth masks). Both are threshold-free.

**19. Precision, recall, F1?**
Precision = of flagged items, the fraction truly defective. Recall = of truly defective items, the fraction flagged. F1 is their
harmonic mean. They depend on the threshold, unlike AUROC.

**20. Why don't you report accuracy?**
The test sets are imbalanced (often mostly defective, or in production mostly normal). A model that says "always defective" can
look accurate. We report class counts together with AUROC, precision, recall, F1 and false-positive rate.

**21. What is train/test leakage and how do you prevent it?**
Information from test data influencing training or tuning. We fit PCA and calibrate thresholds on normal *training* images only;
masks and test images are used just for scoring; a unit test even deletes all test data and checks that fitting and the threshold
are unchanged.

**22. How do you test robustness?**
We keep the model fitted on clean images and perturb only the test images with synthetic brightness, gamma, light-gradient,
shadow and noise transforms, then compare metrics with and without our illumination-aware processing. These are *synthetic*
tests; real lighting could behave differently.

**23. What exactly is our novelty?**
Not a new algorithm. Each idea (illumination handling, multi-scale, statistical thresholds) already exists in the literature, and
related work on illumination-robust subspace detection exists. Our contribution is a transparent, reproducible implementation and
a fair ablation and robustness study of the combination on a simple CPU-friendly setting, with honest reporting of what works and
what does not.

**24. What is the difference between the SubspaceAD paper and our implementation?**
The paper uses frozen DINOv2 patch features with PCA and a reconstruction residual and reports state-of-the-art few-shot results.
We wrote our own simplified version: a default handcrafted backend, an optional small DINOv2 backend with our own layer choices,
our own calibration and scoring details. It is inspired by the paper, **not** a reproduction, and our numbers are not comparable to
the paper's.

**25. What does "cross-fitting" mean and why use it?**
Residuals on the images a PCA was fitted on are unrealistically small. For calibration we split the normal images into K folds, fit
on K−1 folds and score the held-out fold, so thresholds reflect how *unseen* normal images behave.

**26. What are the main limitations?**
PCA is linear and global; handcrafted features are weak; few training images give noisy thresholds and more false positives;
normalisation may hide defects; synthetic lighting is not real lighting; MVTec is controlled and imbalanced. See `LIMITATIONS.md`.

**27. Why are z-scores used for the maps, and are they probabilities?**
Raw residuals from different scales have different magnitudes; z-normalising with normal calibration statistics makes them
combinable. They are *normalised anomaly scores*, not calibrated probabilities – we never present them as probabilities.

**28. What happens with very few training images, e.g. 2?**
PCA still works (the number of patches is large) but the normal model is poor and cross-fitting uses very few images per fold, so
thresholds are unreliable. The code warns and, with 1 image, falls back to biased resubstitution and records it.

**29. How would you demonstrate that multi-scale helps small defects?**
Use the size-study table: compare variants A and C on defects with area < 1 % of the image (image recall and pixel recall by size
bin). If C is not better there, report that honestly.

**30. What did your experiments actually show?**
Only what is in `EXPERIMENTAL_RESULTS.md`. So far those are runs on the *synthetic* test data, where every variant ranks
clean images perfectly. The adaptive threshold removed the false alarms that the fixed threshold produced. Preprocessing
helped under synthetic darkening and shadows but hurt badly under sensor noise. Multi-scale fusion did not help on that data.
The MVTec AD benchmark has not been run yet, so we make no claim about real performance.

**31. How could you extend the project?**
Position-aware PCA, robust PCA, learned α per category, real lighting data, more backbones, or AU-PRO evaluation.
