"""End-to-end anomaly pipeline tying together every stage.

load -> resize -> (illumination preprocessing) -> feature extraction -> PCA
residual maps (fine/coarse) -> normalisation + fusion -> threshold -> mask.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from src.config import PipelineConfig, ThresholdConfig
from src.data.image_io import resize_square, validate_image_array
from src.features import FeatureExtractor, create_feature_extractor
from src.models.multiscale_detector import MultiScaleAnomalyDetector, image_score_from_map
from src.models.thresholding import AdaptiveThreshold, apply_threshold
from src.preprocessing.illumination import ImagePreprocessor

log = logging.getLogger(__name__)


@dataclass
class Analysis:
    """Intermediate result of running an image through the pipeline (before fusion/threshold)."""

    image: np.ndarray                      # resized RGB uint8
    processed: np.ndarray                  # after illumination preprocessing
    z_maps: dict[str, np.ndarray]          # normalised anomaly map per computed scale
    timings: dict[str, float] = field(default_factory=dict)


@dataclass
class Prediction:
    image: np.ndarray
    processed: np.ndarray
    heatmap: np.ndarray                    # fused normalised anomaly map (z-score units)
    mask: np.ndarray                       # bool, heatmap >= threshold (after optional blob filtering)
    image_score: float                     # max (or top-k mean) of heatmap
    mean_score: float
    max_score: float
    threshold: float
    threshold_description: str
    is_anomalous: bool
    alpha: float
    z_maps: dict[str, np.ndarray]
    timings: dict[str, float]


class AnomalyPipeline:
    def __init__(self, config: PipelineConfig | None = None, extractor: FeatureExtractor | None = None):
        self.config = config or PipelineConfig()
        self.config.validate()
        self.extractor = extractor or create_feature_extractor(self.config.features)
        # if a requested backend was replaced by the labelled fallback, record the truth
        if self.config.features.backend != self.extractor.name:
            self.config.features.backend = self.extractor.name
        self.preprocessor = ImagePreprocessor(self.config.preprocess)
        self.detector = MultiScaleAnomalyDetector(self.config.detector, self.config.features.image_size, self.config.seed)
        self.is_fitted = False
        self.fit_info: dict = {}
        self._thr_cache: dict[tuple, AdaptiveThreshold] = {}

    # --------------------------------------------------------------- helpers
    @property
    def size(self) -> int:
        return self.config.features.image_size

    def _prepare(self, rgb: np.ndarray) -> np.ndarray:
        return resize_square(validate_image_array(rgb), self.size)

    def effective_alpha(self, alpha: float | None = None) -> float:
        """alpha actually used: 1.0 (fine only) unless multi-scale mode is on."""
        if not self.config.detector.multiscale:
            return 1.0
        return self.config.detector.alpha if alpha is None else float(alpha)

    def needed_scales(self, alpha: float | None = None) -> tuple[str, ...]:
        a = self.effective_alpha(alpha)
        return ("fine",) if a >= 1.0 else (("coarse",) if a <= 0.0 else ("fine", "coarse"))

    # ------------------------------------------------------------------ fit
    def fit(self, images: Sequence[np.ndarray]) -> dict:
        """Fit using NORMAL images only (list of RGB arrays)."""
        if len(images) < 2:
            raise ValueError("Fitting needs at least 2 normal images (e.g. for PCA and calibration).")
        t0 = time.perf_counter()
        resized = [self._prepare(im) for im in images]
        self.preprocessor.fit(resized)
        feats = [self.extractor.extract(self.preprocessor(im)) for im in resized]
        t1 = time.perf_counter()
        self.detector.fit(feats)
        self._thr_cache.clear()
        self.is_fitted = True
        self.fit_info = {
            "n_train_images": len(images),
            "feature_dims": {s: int(f.shape[-1]) for s, f in feats[0].items()},
            "feature_grids": {s: list(f.shape[:2]) for s, f in feats[0].items()},
            "pca_components": {s: d.n_components_ for s, d in self.detector.detectors_.items()},
            "calibration_mode": self.detector.calibration_mode_,
            "backend_label": self.extractor.label,
            "feature_extraction_seconds": round(t1 - t0, 3),
            "pca_calibration_seconds": round(time.perf_counter() - t1, 3),
        }
        return self.fit_info

    def _require_fit(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("Pipeline is not fitted. Fit it on normal images first (see Dataset & Training).")

    # --------------------------------------------------------------- analyse
    def analyze(self, rgb: np.ndarray, scales: Sequence[str] | None = None) -> Analysis:
        """Run preprocessing, feature extraction and per-scale normalised maps for one image."""
        self._require_fit()
        scales = tuple(scales or self.needed_scales())
        t0 = time.perf_counter()
        img = self._prepare(rgb)
        proc = self.preprocessor(img)
        t1 = time.perf_counter()
        feats = self.extractor.extract(proc)
        t2 = time.perf_counter()
        z = self.detector.normalize(self.detector.raw_maps(feats, scales))
        t3 = time.perf_counter()
        return Analysis(img, proc, z, {"preprocess": t1 - t0, "features": t2 - t1, "scoring": t3 - t2})

    def fused_map(self, analysis: Analysis, alpha: float | None = None) -> np.ndarray:
        return self.detector.fuse(analysis.z_maps, self.effective_alpha(alpha))

    # ------------------------------------------------------------- threshold
    def threshold_for(self, thr_cfg: ThresholdConfig | None = None, alpha: float | None = None) -> AdaptiveThreshold:
        """Threshold fitted on held-out NORMAL calibration scores for the given alpha."""
        self._require_fit()
        c = thr_cfg or self.config.threshold
        a = self.effective_alpha(alpha)
        key = (c.method, c.fixed_value, c.k, c.percentile, round(a, 6))
        if key not in self._thr_cache:
            scores = None if c.method == "fixed" else self.detector.calibration_scores(a)
            self._thr_cache[key] = AdaptiveThreshold(c).fit(scores)
        return self._thr_cache[key]

    # --------------------------------------------------------------- predict
    def predict(
        self,
        rgb: np.ndarray,
        alpha: float | None = None,
        thr_cfg: ThresholdConfig | None = None,
        analysis: Analysis | None = None,
    ) -> Prediction:
        c = thr_cfg or self.config.threshold
        a = self.effective_alpha(alpha)
        an = analysis or self.analyze(rgb, self.needed_scales(a))
        t0 = time.perf_counter()
        heat = self.fused_map(an, a)
        thr = self.threshold_for(c, a)
        mask = apply_threshold(heat, thr.threshold, c.min_area)
        dcfg = self.config.detector
        score = image_score_from_map(heat, dcfg.image_score, dcfg.topk_frac)
        timings = dict(an.timings, fuse_threshold=time.perf_counter() - t0)
        return Prediction(
            image=an.image, processed=an.processed, heatmap=heat, mask=mask,
            image_score=score, mean_score=float(heat.mean()), max_score=float(heat.max()),
            threshold=thr.threshold, threshold_description=thr.describe(),
            is_anomalous=bool(mask.any()) if c.min_area > 0 else bool(score >= thr.threshold),
            alpha=a, z_maps=an.z_maps, timings=timings,
        )

    # ----------------------------------------------------------- persistence
    def get_state(self) -> tuple[dict[str, np.ndarray], dict]:
        self._require_fit()
        arrays = self.detector.get_state()
        pre = self.preprocessor.get_state()
        arrays["pre/target"] = np.array(
            [np.nan if pre["target_mean"] is None else pre["target_mean"],
             np.nan if pre["target_std"] is None else pre["target_std"]], np.float64)
        meta = {
            "config": self.config.to_dict(),
            "fit_info": self.fit_info,
            "calibration_mode": self.detector.calibration_mode_,
            "backend_label": self.extractor.label,
        }
        return arrays, meta

    @classmethod
    def from_state(cls, arrays: dict[str, np.ndarray], meta: dict) -> "AnomalyPipeline":
        cfg = PipelineConfig.from_dict(meta["config"])
        pipe = cls(cfg)
        pipe.detector.set_state(arrays, meta.get("calibration_mode", "unknown"))
        tm, ts = arrays["pre/target"]
        pipe.preprocessor.set_state({"target_mean": None if np.isnan(tm) else float(tm),
                                     "target_std": None if np.isnan(ts) else float(ts)})
        pipe.fit_info = meta.get("fit_info", {})
        pipe.is_fitted = True
        return pipe
