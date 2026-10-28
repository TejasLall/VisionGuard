"""Synthetic MVTec-style dataset generator.

IMPORTANT: this data exists *only* to smoke-test the software without
downloading MVTec AD.  Numbers measured on it say nothing about real
industrial performance.

Two categories are produced:

* ``synthetic_texture`` - a stationary fabric-like texture.
* ``synthetic_object``  - a round part on a plain background, randomly shifted.

Defect types: ``scratch`` (thin, small area), ``spot`` (small dark blob) and
``patch`` (large region with a different texture).  Masks follow the MVTec
naming convention ``<id>_mask.png``.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.data.image_io import save_image_rgb

SIZE = 256


def _texture(rng: np.random.Generator, size: int = SIZE) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    phase = rng.uniform(0, 2 * np.pi, 2)
    base = 0.5 + 0.12 * np.sin(xx * 0.55 + phase[0]) * np.sin(yy * 0.55 + phase[1])
    noise = cv2.GaussianBlur(rng.normal(0, 1, (size, size)).astype(np.float32), (0, 0), 1.2) * 0.08
    img = np.clip(base + noise, 0, 1)
    rgb = np.stack([img * 0.85, img * 0.75, img * 0.6], axis=-1)
    return rgb


def _object(rng: np.random.Generator, size: int = SIZE) -> np.ndarray:
    bg = np.full((size, size, 3), 0.25, np.float32)
    bg += cv2.GaussianBlur(rng.normal(0, 1, (size, size, 3)).astype(np.float32), (0, 0), 2) * 0.02
    cx, cy = size // 2 + rng.integers(-12, 13), size // 2 + rng.integers(-12, 13)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    disc = (dist < 80).astype(np.float32)
    shade = 0.62 + 0.15 * (1 - dist / 80) + cv2.GaussianBlur(
        rng.normal(0, 1, (size, size)).astype(np.float32), (0, 0), 1.5) * 0.03
    ring = ((dist > 60) & (dist < 66)).astype(np.float32) * 0.12
    obj = np.stack([shade * 0.9 - ring, shade * 0.8 - ring, shade * 0.7 - ring], axis=-1)
    img = bg * (1 - disc[..., None]) + obj * disc[..., None]
    return np.clip(img, 0, 1)


def _add_defect(img: np.ndarray, kind: str, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    size = img.shape[0]
    mask = np.zeros((size, size), np.uint8)
    out = img.copy()
    # keep defects well inside the frame (and inside the disc for objects)
    cx, cy = int(rng.integers(70, size - 70)), int(rng.integers(70, size - 70))
    if kind == "scratch":
        ang = rng.uniform(0, np.pi)
        length = int(rng.integers(25, 45))
        p1 = (int(cx - np.cos(ang) * length), int(cy - np.sin(ang) * length))
        p2 = (int(cx + np.cos(ang) * length), int(cy + np.sin(ang) * length))
        cv2.line(mask, p1, p2, 255, 2)
        out[mask > 0] = out[mask > 0] * 0.25 + 0.05
    elif kind == "spot":
        r = int(rng.integers(4, 9))
        cv2.circle(mask, (cx, cy), r, 255, -1)
        soft = cv2.GaussianBlur(mask.astype(np.float32) / 255, (0, 0), 1.0)
        out = out * (1 - 0.8 * soft[..., None]) + 0.05 * soft[..., None]
    elif kind == "patch":
        w, h = int(rng.integers(35, 60)), int(rng.integers(35, 60))
        x0, y0 = int(np.clip(cx - w // 2, 0, size - w)), int(np.clip(cy - h // 2, 0, size - h))
        mask[y0:y0 + h, x0:x0 + w] = 255
        fx = rng.uniform(0.2, 0.3)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        patt = 0.5 + 0.25 * np.sin(xx * fx * 2 + yy * fx)
        out[y0:y0 + h, x0:x0 + w] = np.stack([patt * 0.5, patt * 0.7, patt * 0.9], axis=-1)
    else:
        raise ValueError(kind)
    return np.clip(out, 0, 1), mask


def _finalize(img: np.ndarray, rng: np.random.Generator, brightness_jitter: float = 0.08) -> np.ndarray:
    gain = 1.0 + rng.uniform(-brightness_jitter, brightness_jitter)
    img = np.clip(img * gain + rng.normal(0, 0.008, img.shape), 0, 1)
    return (img * 255).astype(np.uint8)


def create_synthetic_dataset(
    root: str | Path,
    n_train: int = 24,
    n_test_good: int = 12,
    n_test_per_defect: int = 8,
    seed: int = 0,
    categories: tuple[str, ...] = ("synthetic_texture", "synthetic_object"),
) -> Path:
    """Write an MVTec-style synthetic dataset below ``root`` and return the path."""
    root = Path(root)
    rng = np.random.default_rng(seed)
    makers = {"synthetic_texture": _texture, "synthetic_object": _object}
    for cat in categories:
        make = makers[cat]
        for i in range(n_train):
            save_image_rgb(root / cat / "train" / "good" / f"{i:03d}.png", _finalize(make(rng), rng))
        for i in range(n_test_good):
            save_image_rgb(root / cat / "test" / "good" / f"{i:03d}.png", _finalize(make(rng), rng))
        for kind in ("scratch", "spot", "patch"):
            for i in range(n_test_per_defect):
                img, mask = _add_defect(make(rng), kind, rng)
                save_image_rgb(root / cat / "test" / kind / f"{i:03d}.png", _finalize(img, rng))
                save_image_rgb(root / cat / "ground_truth" / kind / f"{i:03d}_mask.png", mask)
    return root
