"""CPU-only handcrafted dense feature extractor.

Per-pixel channels (17):
    L, a, b (CIELAB / 255)                             -> 3
    |Sobel x|, |Sobel y|, gradient magnitude, |Laplacian| of L -> 4
    uniform LBP (P=8, R=1) one-hot histogram channels  -> 10

Each *scale* aggregates these channels over a square window (box filter, i.e.
the window mean of every channel; for the LBP one-hot channels this is the
local LBP histogram) and additionally the local standard deviation of L and of
the gradient magnitude, giving 19 dimensions per patch.  The feature grid is
sub-sampled with the scale's stride so that grid cell ``i`` is centred at
pixel ``i * stride + stride // 2``.

Fine scale   : window 9 px,  stride 4 -> 64 x 64 grid at 256 px (small defects)
Coarse scale : window 33 px, stride 8 -> 32 x 32 grid at 256 px (large defects)

The two scales therefore differ in receptive field (9 vs 33 px), feature
resolution and neighbourhood aggregation - they are not a resized copy of the
same map.

This backend is NOT a deep network and makes no claim to match DINOv2.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import uniform_filter
from skimage.feature import local_binary_pattern

from src.config import FeatureConfig
from src.features.base import FeatureExtractor

_LBP_P, _LBP_R = 8, 1
_N_LBP = _LBP_P + 2  # 'uniform' LBP has P + 2 labels


class HandcraftedFeatureExtractor(FeatureExtractor):
    name = "handcrafted"

    def __init__(self, cfg: FeatureConfig | None = None):
        self.cfg = cfg or FeatureConfig()
        self.scales = {
            "fine": (self.cfg.fine_window, self.cfg.fine_stride),
            "coarse": (self.cfg.coarse_window, self.cfg.coarse_stride),
        }

    @property
    def label(self) -> str:
        return "Handcrafted (colour + gradient + LBP) - CPU, not a deep model"

    # ------------------------------------------------------------- channels
    @staticmethod
    def pixel_channels(rgb: np.ndarray) -> np.ndarray:
        """(H, W, 17) float32 per-pixel descriptor channels."""
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        L8 = lab[..., 0]
        lab_f = lab.astype(np.float32) / 255.0
        Lf = lab_f[..., 0]
        gx = cv2.Sobel(Lf, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(Lf, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx * gx + gy * gy)
        lap = np.abs(cv2.Laplacian(Lf, cv2.CV_32F, ksize=3))
        lbp = local_binary_pattern(L8, _LBP_P, _LBP_R, method="uniform")
        onehot = (lbp[..., None] == np.arange(_N_LBP, dtype=np.float64)).astype(np.float32)
        return np.concatenate(
            [lab_f, np.abs(gx)[..., None], np.abs(gy)[..., None], mag[..., None], lap[..., None], onehot],
            axis=-1,
        ).astype(np.float32)

    @staticmethod
    def _pool(ch: np.ndarray, window: int, stride: int) -> np.ndarray:
        """Window mean of all channels + local std of L (ch 0) and of gradient magnitude (ch 5)."""
        size = (window, window, 1)
        mean = uniform_filter(ch, size=size, mode="reflect")
        sq = uniform_filter(ch[..., [0, 5]] ** 2, size=size, mode="reflect")
        std = np.sqrt(np.maximum(sq - mean[..., [0, 5]] ** 2, 0.0))
        feat = np.concatenate([mean, std], axis=-1)
        off = stride // 2
        return np.ascontiguousarray(feat[off::stride, off::stride], dtype=np.float32)

    # ------------------------------------------------------------------ API
    def extract(self, rgb: np.ndarray) -> dict[str, np.ndarray]:
        ch = self.pixel_channels(rgb)
        return {name: self._pool(ch, w, s) for name, (w, s) in self.scales.items()}

    def describe(self) -> dict:
        return {
            "backend": self.name,
            "dims_per_patch": 19,
            "channels": "Lab(3) + |Sobel x|,|Sobel y|,grad-mag,|Laplacian| (4) + uniform-LBP histogram (10) "
                        "+ local std of L and of grad-mag (2)",
            "scales": {
                n: {"window_px": w, "stride_px": s, "grid_at_image_size": self.cfg.image_size // s}
                for n, (w, s) in self.scales.items()
            },
        }
