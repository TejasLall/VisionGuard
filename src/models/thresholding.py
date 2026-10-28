"""Contribution C - thresholds estimated from NORMAL calibration data only.

``S_normal`` is the set of image-level anomaly scores (maximum of the fused,
normalised anomaly map) of held-out normal calibration images.

* ``fixed``      : T = fixed_value                  (category-independent baseline)
* ``mean_std``   : T = mean(S_normal) + k * std(S_normal)
* ``percentile`` : T = P-th percentile of S_normal

The same T is applied to the anomaly map to obtain the binary mask, so an
image is flagged defective exactly when its mask is non-empty.

These are heuristics: with few calibration images the estimates of the mean,
the standard deviation and especially the upper percentile are noisy, and
the Gaussian-tail interpretation of ``k`` is not guaranteed because maxima
of anomaly maps are not Gaussian.  Ground-truth masks or defective images are
never used here.
"""
from __future__ import annotations

import logging

import numpy as np

from src.config import THRESHOLD_METHODS, ThresholdConfig

log = logging.getLogger(__name__)


class AdaptiveThreshold:
    def __init__(self, cfg: ThresholdConfig | None = None):
        self.cfg = cfg or ThresholdConfig()
        if self.cfg.method not in THRESHOLD_METHODS:
            raise ValueError(f"method must be one of {THRESHOLD_METHODS}")
        self.threshold_: float | None = None
        self.stats_: dict[str, float] = {}

    def fit(self, normal_scores: np.ndarray | None = None) -> "AdaptiveThreshold":
        c = self.cfg
        if c.method == "fixed":
            self.threshold_ = float(c.fixed_value)
            self.stats_ = {}
            return self
        s = np.asarray(normal_scores if normal_scores is not None else [], dtype=np.float64).ravel()
        s = s[np.isfinite(s)]
        if s.size == 0:
            raise ValueError(f"'{c.method}' threshold needs normal calibration scores, got none")
        if s.size < 5:
            log.warning("Only %d calibration scores: the %s threshold is a very noisy estimate.", s.size, c.method)
        mean, std = float(s.mean()), float(s.std(ddof=1)) if s.size > 1 else 0.0
        self.stats_ = {"n": float(s.size), "mean": mean, "std": std, "max": float(s.max())}
        if c.method == "mean_std":
            self.threshold_ = mean + c.k * std
        else:
            self.threshold_ = float(np.percentile(s, c.percentile))
        return self

    @property
    def threshold(self) -> float:
        if self.threshold_ is None:
            raise RuntimeError("threshold not fitted")
        return self.threshold_

    def describe(self) -> str:
        c = self.cfg
        if c.method == "fixed":
            return f"fixed T = {c.fixed_value:g}"
        if c.method == "mean_std":
            return f"T = mean + {c.k:g} * std of normal image scores = {self.threshold:.3f}"
        return f"T = {c.percentile:g}th percentile of normal image scores = {self.threshold:.3f}"


def apply_threshold(score_map: np.ndarray, threshold: float, min_area: int = 0) -> np.ndarray:
    """Binary mask ``score_map >= threshold``; optionally drop blobs smaller than ``min_area`` px."""
    mask = score_map >= threshold
    if min_area > 0 and mask.any():
        import cv2

        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        keep = np.zeros(n, bool)
        keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
        mask = keep[labels]
    return mask
