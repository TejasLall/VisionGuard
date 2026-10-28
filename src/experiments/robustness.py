"""Controlled SYNTHETIC illumination/noise perturbations for robustness tests.

These transforms are applied to TEST images only (after the model has been
fitted and calibrated on clean normal images).  Labels and ground-truth masks
are unchanged.  They simulate lighting changes in software; they are not a
substitute for real-world lighting variation captured by a camera.
"""
from __future__ import annotations

from typing import Callable

import cv2
import numpy as np
import pandas as pd

Transform = Callable[[np.ndarray, int], np.ndarray]   # (rgb uint8, per-image seed) -> rgb uint8


def _to_u8(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0, 255).astype(np.uint8)


def identity(img: np.ndarray, seed: int = 0) -> np.ndarray:
    return img


def brightness(factor: float) -> Transform:
    """Multiply intensities by ``factor`` (<1 darker, >1 brighter)."""
    def f(img: np.ndarray, seed: int = 0) -> np.ndarray:
        return _to_u8(img.astype(np.float32) * factor)
    return f


def gamma(g: float) -> Transform:
    """Gamma correction out = 255 * (in/255) ** g  (g>1 darkens mid-tones, g<1 brightens)."""
    def f(img: np.ndarray, seed: int = 0) -> np.ndarray:
        return _to_u8(255.0 * (img.astype(np.float32) / 255.0) ** g)
    return f


def gradient_light(strength: float = 0.6) -> Transform:
    """Uneven illumination: multiplicative linear ramp from 1 to (1 - strength) at a random angle."""
    def f(img: np.ndarray, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        h, w = img.shape[:2]
        ang = rng.uniform(0, 2 * np.pi)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        ramp = (xx - w / 2) * np.cos(ang) + (yy - h / 2) * np.sin(ang)
        ramp = (ramp - ramp.min()) / max(float(np.ptp(ramp)), 1e-6)
        field = 1.0 - strength * ramp
        return _to_u8(img.astype(np.float32) * field[..., None])
    return f


def shadow(strength: float = 0.55) -> Transform:
    """Soft-edged elliptical shadow at a random position/size."""
    def f(img: np.ndarray, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        h, w = img.shape[:2]
        m = np.zeros((h, w), np.float32)
        centre = (int(rng.uniform(0.25, 0.75) * w), int(rng.uniform(0.25, 0.75) * h))
        axes = (int(rng.uniform(0.2, 0.4) * w), int(rng.uniform(0.2, 0.4) * h))
        cv2.ellipse(m, centre, axes, float(rng.uniform(0, 180)), 0, 360, 1.0, -1)
        m = cv2.GaussianBlur(m, (0, 0), 0.06 * min(h, w))
        return _to_u8(img.astype(np.float32) * (1.0 - strength * m)[..., None])
    return f


def gaussian_noise(sigma: float = 10.0) -> Transform:
    """Additive i.i.d. Gaussian sensor-like noise (sigma in 8-bit grey levels)."""
    def f(img: np.ndarray, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return _to_u8(img.astype(np.float32) + rng.normal(0, sigma, img.shape))
    return f


def default_transforms(include_noise: bool = True) -> dict[str, Transform]:
    """The standard robustness suite (name -> transform).  ``identity`` is the clean reference."""
    t: dict[str, Transform] = {
        "identity": identity,
        "dark_x0.5": brightness(0.5),
        "bright_x1.5": brightness(1.5),
        "gamma_0.5": gamma(0.5),
        "gamma_2.0": gamma(2.0),
        "uneven_gradient": gradient_light(0.6),
        "shadow": shadow(0.55),
    }
    if include_noise:
        t["noise_sigma10"] = gaussian_noise(10.0)
    return t


DELTA_METRICS = ("image_auroc", "pixel_auroc", "image_f1", "pixel_f1", "image_fpr")


def add_deltas(df: pd.DataFrame, reference: str = "identity") -> pd.DataFrame:
    """Add ``delta_<metric>`` = metric(transform) - metric(reference transform) per category/variant."""
    ref = df[df["transform"] == reference].set_index(["category", "variant"])
    out = df.copy()
    for m in DELTA_METRICS:
        if m not in df:
            continue
        base = ref[m]
        out[f"delta_{m}"] = [
            r[m] - base.get((r["category"], r["variant"]), np.nan) for _, r in df.iterrows()
        ]
    return out
