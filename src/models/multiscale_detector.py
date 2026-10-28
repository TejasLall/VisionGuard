"""Contribution B - multi-scale anomaly maps with calibrated fusion.

One :class:`PCAAnomalyDetector` is fitted per feature scale (``fine`` and
``coarse``; see the feature extractors for how the scales differ in receptive
field and resolution).  For each scale:

1. patch residuals ``score_s(i, j)`` on the scale's own grid,
2. bilinear up-sampling to the analysis resolution and Gaussian smoothing,
3. z-score normalisation with the mean/std of the *normal calibration* pixel
   scores of that scale,  ``z_s = (score_s - mu_s) / sigma_s``.

Normalisation is needed because residual magnitudes of different scales are
not comparable.  The fused map is

    S_fused = alpha * z_fine + (1 - alpha) * z_coarse        (0 <= alpha <= 1)

``alpha = 1`` reproduces the single-scale (fine) baseline.

Calibration data
----------------
Residuals of the very images a PCA was fitted on are optimistically small, so
calibrating on them would give thresholds that are too low.  By default we use
K-fold *cross-fitting*: for each fold a PCA is fitted on the other folds and
scores are collected on the held-out fold.  Only normal training images are
involved.  The final detectors are then fitted on all normal images.
``calibration='resubstitution'`` (scores on the fitting images) is available
but biased, and is used automatically when only one image is provided.
"""
from __future__ import annotations

import logging
from typing import Sequence

import cv2
import numpy as np

from src.config import DetectorConfig
from src.features.base import SCALE_NAMES
from src.models.pca_detector import PCAAnomalyDetector

log = logging.getLogger(__name__)


def image_score_from_map(score_map: np.ndarray, mode: str = "max", topk_frac: float = 0.01) -> float:
    """Image-level score from a pixel map: maximum, or mean of the top ``topk_frac`` pixels."""
    if mode == "max":
        return float(score_map.max())
    flat = score_map.ravel()
    k = max(1, int(round(topk_frac * flat.size)))
    return float(np.partition(flat, -k)[-k:].mean())


class MultiScaleAnomalyDetector:
    def __init__(self, cfg: DetectorConfig, map_size: int, seed: int = 0):
        self.cfg = cfg
        self.map_size = int(map_size)
        self.seed = seed
        self.detectors_: dict[str, PCAAnomalyDetector] = {}
        self.norm_: dict[str, tuple[float, float]] = {}
        self.calib_raw_: dict[str, np.ndarray] = {}      # (N, H, W) held-out raw maps
        self.calibration_mode_: str = "none"

    # ----------------------------------------------------------------- maps
    def _to_map(self, patch_scores: np.ndarray) -> np.ndarray:
        m = cv2.resize(patch_scores.astype(np.float32), (self.map_size, self.map_size), interpolation=cv2.INTER_LINEAR)
        if self.cfg.smooth_sigma > 0:
            m = cv2.GaussianBlur(m, (0, 0), self.cfg.smooth_sigma)
        return m

    def raw_maps(self, feats: dict[str, np.ndarray], scales: Sequence[str] | None = None) -> dict[str, np.ndarray]:
        """Un-normalised smoothed anomaly maps (H, W) per scale."""
        scales = scales or tuple(self.detectors_)
        return {s: self._to_map(self.detectors_[s].score_map(feats[s])) for s in scales}

    def normalize(self, raw: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        out = {}
        for s, m in raw.items():
            mu, sd = self.norm_[s]
            out[s] = ((m - mu) / sd).astype(np.float32)
        return out

    @staticmethod
    def fuse(z_maps: dict[str, np.ndarray], alpha: float) -> np.ndarray:
        """alpha * fine + (1 - alpha) * coarse; ``alpha = 1`` -> fine only."""
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must lie in [0, 1]")
        if alpha >= 1.0 or "coarse" not in z_maps:
            return z_maps["fine"]
        if alpha <= 0.0:
            return z_maps["coarse"]
        return alpha * z_maps["fine"] + (1.0 - alpha) * z_maps["coarse"]

    # ------------------------------------------------------------------ fit
    def _stack(self, feats: Sequence[dict[str, np.ndarray]], scale: str, idx: Sequence[int], rng) -> np.ndarray:
        X = np.concatenate([feats[i][scale].reshape(-1, feats[i][scale].shape[-1]) for i in idx], axis=0)
        if X.shape[0] > self.cfg.max_patches:
            X = X[rng.choice(X.shape[0], self.cfg.max_patches, replace=False)]
        return X

    def fit(self, feats: Sequence[dict[str, np.ndarray]], scales: Sequence[str] = SCALE_NAMES) -> "MultiScaleAnomalyDetector":
        """Fit on feature maps of NORMAL images and calibrate score normalisation."""
        n = len(feats)
        if n < 1:
            raise ValueError("no training features")
        rng = np.random.default_rng(self.seed)
        scales = tuple(scales)
        all_idx = list(range(n))
        self.detectors_ = {
            s: PCAAnomalyDetector(self.cfg.n_components).fit(self._stack(feats, s, all_idx, rng)) for s in scales
        }

        mode = self.cfg.calibration
        folds = min(self.cfg.calib_folds, n)
        if mode == "crossfit" and n >= 2 and folds >= 2:
            self.calibration_mode_ = f"crossfit-{folds}fold"
            perm = rng.permutation(n)
            held: dict[str, list[np.ndarray | None]] = {s: [None] * n for s in scales}
            for f in range(folds):
                test_idx = perm[f::folds]
                held_out = set(test_idx.tolist())
                train_idx = [i for i in perm.tolist() if i not in held_out]
                for s in scales:
                    det = PCAAnomalyDetector(self.cfg.n_components).fit(self._stack(feats, s, train_idx, rng))
                    for i in test_idx:
                        held[s][i] = self._to_map(det.score_map(feats[i][s]))
            self.calib_raw_ = {s: np.stack(held[s]).astype(np.float32) for s in scales}
        else:
            self.calibration_mode_ = "resubstitution"
            if n == 1 or mode == "crossfit":
                log.warning("Cross-fitting impossible with %d image(s); using biased resubstitution scores.", n)
            self.calib_raw_ = {
                s: np.stack([self._to_map(self.detectors_[s].score_map(f[s])) for f in feats]).astype(np.float32)
                for s in scales
            }

        for s in scales:
            m = self.calib_raw_[s]
            self.norm_[s] = (float(m.mean()), max(float(m.std()), 1e-8))
        return self

    # ----------------------------------------------------------- calibration
    def calibration_scores(self, alpha: float) -> np.ndarray:
        """Image-level scores (per cfg.image_score) of held-out NORMAL images for a given alpha."""
        scales = list(self.calib_raw_)
        n = self.calib_raw_[scales[0]].shape[0]
        out = np.empty(n)
        for i in range(n):
            z = self.normalize({s: self.calib_raw_[s][i] for s in scales})
            out[i] = image_score_from_map(self.fuse(z, alpha), self.cfg.image_score, self.cfg.topk_frac)
        return out

    # ----------------------------------------------------------- persistence
    def get_state(self) -> dict[str, np.ndarray]:
        state: dict[str, np.ndarray] = {}
        for s, det in self.detectors_.items():
            for k, v in det.get_state().items():
                state[f"pca/{s}/{k}"] = v
            state[f"norm/{s}"] = np.array(self.norm_[s], np.float64)
            state[f"calib/{s}"] = self.calib_raw_[s].astype(np.float16)
        return state

    def set_state(self, state: dict[str, np.ndarray], calibration_mode: str = "unknown") -> "MultiScaleAnomalyDetector":
        scales = sorted({k.split("/")[1] for k in state if k.startswith("pca/")}, key=SCALE_NAMES.index)
        for s in scales:
            sub = {k.split("/")[2]: state[k] for k in state if k.startswith(f"pca/{s}/")}
            self.detectors_[s] = PCAAnomalyDetector(self.cfg.n_components).set_state(sub)
            mu, sd = state[f"norm/{s}"]
            self.norm_[s] = (float(mu), float(sd))
            self.calib_raw_[s] = state[f"calib/{s}"].astype(np.float32)
        self.calibration_mode_ = calibration_mode
        return self
