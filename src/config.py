"""Typed configuration objects for the whole pipeline.

Every tunable of the system lives in one of these dataclasses so that an
experiment is fully described by a single JSON/YAML-serialisable dict.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

CLAHE_MODES = ("always", "auto", "off")
BACKENDS = ("handcrafted", "dinov2")
THRESHOLD_METHODS = ("fixed", "mean_std", "percentile")
CALIBRATION_MODES = ("crossfit", "resubstitution")
IMAGE_SCORE_MODES = ("max", "topk_mean")


@dataclass
class PreprocessConfig:
    """Contribution A: illumination-aware preprocessing."""

    enabled: bool = False
    clahe_mode: str = "always"          # always | auto | off
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: int = 8
    flatfield: bool = True               # divide L by a heavy Gaussian blur of L
    flatfield_sigma_frac: float = 0.2    # sigma = frac * min(H, W)
    lum_norm: bool = True                # match L mean/std to *training* statistics
    lum_gain_limits: tuple[float, float] = (0.4, 2.5)
    gray_world: bool = False             # per-channel gain so channel means are equal
    # thresholds used only when clahe_mode == "auto"
    auto_min_mean: float = 70.0
    auto_max_mean: float = 190.0
    auto_min_std: float = 25.0
    auto_max_lowfreq_std: float = 18.0


@dataclass
class FeatureConfig:
    backend: str = "handcrafted"         # handcrafted | dinov2
    image_size: int = 256                # analysis resolution (square)
    # handcrafted backend: window = receptive field, stride = feature-grid step
    fine_window: int = 9
    fine_stride: int = 4
    coarse_window: int = 33
    coarse_stride: int = 8
    # dinov2 backend (optional, needs requirements-deep.txt)
    dino_model: str = "facebook/dinov2-small"
    dino_input_size: int = 448           # multiple of 14
    dino_fine_layers: tuple[int, ...] = (6, 7, 8)   # hidden_states indices
    dino_coarse_pool: int = 3            # neighbourhood (odd) averaged for the coarse scale
    allow_fallback: bool = False         # fall back to handcrafted if DINOv2 is unavailable


@dataclass
class DetectorConfig:
    n_components: float | int = 0.95     # float in (0,1): variance ratio; int >= 1: count
    smooth_sigma: float = 4.0            # Gaussian smoothing of the anomaly map (px)
    max_patches: int = 150_000           # random sub-sampling cap for PCA fitting
    calibration: str = "crossfit"        # crossfit | resubstitution
    calib_folds: int = 5
    multiscale: bool = False             # Contribution B on/off
    alpha: float = 0.5                   # weight of the fine scale when multiscale
    image_score: str = "max"             # max | topk_mean
    topk_frac: float = 0.01


@dataclass
class ThresholdConfig:
    """Contribution C: thresholding strategy (fixed baseline vs adaptive)."""

    method: str = "fixed"                # fixed | mean_std | percentile
    fixed_value: float = 3.0             # in normalised (z-score) anomaly units
    k: float = 3.0                       # mean_std: T = mean + k * std
    percentile: float = 99.0             # percentile: T = P-th percentile
    min_area: int = 0                    # drop predicted blobs smaller than this (px)


@dataclass
class PipelineConfig:
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    threshold: ThresholdConfig = field(default_factory=ThresholdConfig)
    seed: int = 0

    # ---- (de)serialisation -------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "PipelineConfig":
        d = d or {}
        kwargs: dict[str, Any] = {}
        for f in fields(cls):
            if f.name == "seed":
                kwargs["seed"] = int(d.get("seed", 0))
                continue
            sub_cls = {
                "preprocess": PreprocessConfig,
                "features": FeatureConfig,
                "detector": DetectorConfig,
                "threshold": ThresholdConfig,
            }[f.name]
            kwargs[f.name] = _build(sub_cls, d.get(f.name, {}))
        cfg = cls(**kwargs)
        cfg.validate()
        return cfg

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(yaml.safe_load(fh) or {})

    def copy(self) -> "PipelineConfig":
        return copy.deepcopy(self)

    # ---- validation ----------------------------------------------------------
    def validate(self) -> None:
        if self.preprocess.clahe_mode not in CLAHE_MODES:
            raise ValueError(f"clahe_mode must be one of {CLAHE_MODES}")
        if self.features.backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}")
        if self.threshold.method not in THRESHOLD_METHODS:
            raise ValueError(f"threshold method must be one of {THRESHOLD_METHODS}")
        if self.detector.calibration not in CALIBRATION_MODES:
            raise ValueError(f"calibration must be one of {CALIBRATION_MODES}")
        if self.detector.image_score not in IMAGE_SCORE_MODES:
            raise ValueError(f"image_score must be one of {IMAGE_SCORE_MODES}")
        if not 0.0 <= self.detector.alpha <= 1.0:
            raise ValueError("alpha must lie in [0, 1]")
        if self.features.image_size < 32:
            raise ValueError("image_size must be >= 32")
        if self.features.backend == "dinov2" and self.features.dino_input_size % 14:
            raise ValueError("dino_input_size must be a multiple of 14")
        if not 0.0 < self.threshold.percentile <= 100.0:
            raise ValueError("percentile must be in (0, 100]")


def _build(cls, data: dict[str, Any] | None):
    """Instantiate a dataclass from a dict, ignoring unknown keys, coercing tuples."""
    data = dict(data or {})
    valid = {f.name: f for f in fields(cls)}
    kwargs = {}
    for key, value in data.items():
        if key not in valid:
            continue
        if isinstance(value, list):
            value = tuple(value)
        kwargs[key] = value
    return cls(**kwargs)
