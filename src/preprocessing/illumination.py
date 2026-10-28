"""Contribution A - illumination-aware preprocessing.

Pipeline (each stage optional, applied in this order on the same image):

1. ``gray_world``  - per-channel gain so the R, G, B means are equal.
2. ``flatfield``   - L <- L / GaussianBlur(L, sigma) * mean(blur): removes slowly
                     varying shading (uneven lighting, soft shadows).
3. ``lum_norm``    - affine map of L so its mean/std match the average mean/std of
                     the *normal training images* (learned in :meth:`fit`).
4. ``CLAHE``       - Contrast Limited Adaptive Histogram Equalisation on L.

All colour conversions go RGB -> LAB (uint8) -> RGB so that only luminance is
modified by stages 2-4.  The image is returned unchanged when
``PreprocessConfig.enabled`` is False, giving the baseline.
"""
from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

from src.config import PreprocessConfig


class ImagePreprocessor:
    def __init__(self, cfg: PreprocessConfig | None = None):
        self.cfg = cfg or PreprocessConfig()
        self.target_mean_: float | None = None
        self.target_std_: float | None = None

    # ------------------------------------------------------------------ stats
    @staticmethod
    def diagnostics(rgb: np.ndarray) -> dict[str, float]:
        """Simple illumination descriptors of an RGB image."""
        L = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[..., 0].astype(np.float32)
        sigma = 0.2 * min(L.shape)
        low = cv2.GaussianBlur(L, (0, 0), sigma)
        return {
            "mean_L": float(L.mean()),
            "std_L": float(L.std()),
            "lowfreq_std": float(low.std()),
            "dark_frac": float((L < 20).mean()),
            "bright_frac": float((L > 235).mean()),
        }

    def clahe_recommended(self, rgb: np.ndarray) -> bool:
        """Heuristic used by ``clahe_mode='auto'``: only correct images that look badly lit."""
        d, c = self.diagnostics(rgb), self.cfg
        return (
            d["mean_L"] < c.auto_min_mean
            or d["mean_L"] > c.auto_max_mean
            or d["std_L"] < c.auto_min_std
            or d["lowfreq_std"] > c.auto_max_lowfreq_std
        )

    # --------------------------------------------------------------- stages
    @staticmethod
    def _gray_world(rgb: np.ndarray) -> np.ndarray:
        f = rgb.astype(np.float32)
        means = f.reshape(-1, 3).mean(0)
        gains = means.mean() / np.maximum(means, 1.0)
        return np.clip(f * gains, 0, 255).astype(np.uint8)

    def _stages_before_norm(self, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Apply gray-world + flat-field; return (LAB uint8 image, float L channel)."""
        c = self.cfg
        if c.gray_world:
            rgb = self._gray_world(rgb)
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        L = lab[..., 0].astype(np.float32)
        if c.flatfield:
            sigma = max(c.flatfield_sigma_frac * min(L.shape), 1.0)
            bg = cv2.GaussianBlur(L, (0, 0), sigma)
            L = L / np.maximum(bg, 1.0) * float(bg.mean())
        return lab, L

    # ------------------------------------------------------------------ API
    def fit(self, images: Sequence[np.ndarray]) -> "ImagePreprocessor":
        """Learn target luminance statistics from NORMAL training images only."""
        if self.cfg.enabled and self.cfg.lum_norm and len(images):
            stats = [(L.mean(), L.std()) for L in (self._stages_before_norm(i)[1] for i in images)]
            self.target_mean_ = float(np.mean([s[0] for s in stats]))
            self.target_std_ = float(np.mean([s[1] for s in stats]))
        return self

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        c = self.cfg
        if not c.enabled:
            return rgb
        lab, L = self._stages_before_norm(rgb)
        if c.lum_norm and self.target_mean_ is not None:
            sd = float(L.std())
            gain = np.clip(self.target_std_ / max(sd, 1e-3), *c.lum_gain_limits)
            L = (L - L.mean()) * gain + self.target_mean_
        L8 = np.clip(L, 0, 255).astype(np.uint8)
        apply_clahe = c.clahe_mode == "always" or (c.clahe_mode == "auto" and self.clahe_recommended(rgb))
        if apply_clahe:
            clahe = cv2.createCLAHE(clipLimit=c.clahe_clip_limit, tileGridSize=(c.clahe_tile_grid,) * 2)
            L8 = clahe.apply(L8)
        lab[..., 0] = L8
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # ------------------------------------------------------------ persistence
    def get_state(self) -> dict[str, float | None]:
        return {"target_mean": self.target_mean_, "target_std": self.target_std_}

    def set_state(self, state: dict[str, float | None]) -> None:
        self.target_mean_ = state.get("target_mean")
        self.target_std_ = state.get("target_std")
