"""Reproducible experiment runner for the five required method variants.

Variants (ablation of the three contributions):

    A  Baseline                       no preprocessing, fine scale only, fixed threshold
    B  Baseline + illumination        preprocessing ON
    C  Baseline + multi-scale         fine+coarse fusion
    D  Baseline + adaptive threshold  mean + k*std of normal calibration scores
    E  Full system                    B + C + D

All variants share the same data split, resolution, feature backend, PCA
settings and evaluation rules.  Models are fitted on normal TRAIN images only;
thresholds are calibrated on held-out normal TRAIN images only (cross-fitting);
test images and their masks are used for evaluation only.
"""
from __future__ import annotations

import json
import logging
import platform
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from src.config import PipelineConfig, ThresholdConfig
from src.data.dataset_loader import CategoryData, discover_categories, load_category
from src.data.image_io import ImageLoadError, load_image_rgb, load_mask, resize_square
from src.evaluation.evaluator import AnomalyEvaluator
from src.experiments.robustness import Transform, identity
from src.models.multiscale_detector import image_score_from_map
from src.models.pipeline import AnomalyPipeline
from src.utils.paths import OUTPUTS_DIR, ensure_dir, iso_now, timestamp

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class VariantSpec:
    key: str
    name: str
    preprocess: bool
    multiscale: bool
    threshold: str      # fixed | mean_std | percentile
    pre_overrides: tuple[tuple[str, object], ...] = ()   # e.g. (("flatfield", False),) for op ablations


# Ablation of the individual illumination operations (single scale, adaptive threshold).
_OFF = (("clahe_mode", "off"), ("flatfield", False), ("lum_norm", False), ("gray_world", False))
ILLUMINATION_ABLATION_VARIANTS: tuple[VariantSpec, ...] = (
    VariantSpec("P0", "no preprocessing", False, False, "mean_std"),
    VariantSpec("P1", "CLAHE only", True, False, "mean_std", _OFF[1:3] + (("clahe_mode", "always"),)),
    VariantSpec("P2", "flat-field only", True, False, "mean_std", (("clahe_mode", "off"), ("lum_norm", False), ("flatfield", True))),
    VariantSpec("P3", "luminance-norm only", True, False, "mean_std", (("clahe_mode", "off"), ("flatfield", False), ("lum_norm", True))),
    VariantSpec("P4", "CLAHE (auto) only", True, False, "mean_std", _OFF[1:3] + (("clahe_mode", "auto"),)),
    VariantSpec("P5", "all (flat-field + lum-norm + CLAHE)", True, False, "mean_std", (("clahe_mode", "always"), ("flatfield", True), ("lum_norm", True))),
)

STANDARD_VARIANTS: tuple[VariantSpec, ...] = (
    VariantSpec("A", "Baseline", False, False, "fixed"),
    VariantSpec("B", "Baseline + illumination-aware", True, False, "fixed"),
    VariantSpec("C", "Baseline + multi-scale", False, True, "fixed"),
    VariantSpec("D", "Baseline + adaptive threshold", False, False, "mean_std"),
    VariantSpec("E", "Full proposed system", True, True, "mean_std"),
)

METRIC_COLUMNS = [
    "image_auroc", "image_ap", "image_precision", "image_recall", "image_f1", "image_fpr", "image_f1_oracle",
    "pixel_auroc", "pixel_ap", "pixel_precision", "pixel_recall", "pixel_f1", "pixel_f1_oracle",
]


@dataclass
class ExperimentResult:
    results: pd.DataFrame
    size_results: pd.DataFrame
    out_dir: Path | None = None
    meta: dict = field(default_factory=dict)


def select_train_paths(paths: Sequence[Path], n_train: int | None, seed: int) -> list[Path]:
    """Deterministic few-shot subset of the normal training images."""
    paths = list(paths)
    if n_train is None or n_train >= len(paths):
        return paths
    rng = np.random.default_rng(seed)
    return [paths[i] for i in sorted(rng.choice(len(paths), n_train, replace=False))]


def variant_config(base: PipelineConfig, v: VariantSpec) -> PipelineConfig:
    """Config of one variant derived from the shared base config."""
    cfg = base.copy()
    cfg.preprocess.enabled = v.preprocess
    for attr, value in v.pre_overrides:
        setattr(cfg.preprocess, attr, value)
    cfg.detector.multiscale = True            # alpha is passed explicitly per variant; see _alpha()
    cfg.threshold.method = v.threshold
    return cfg


class ExperimentRunner:
    def __init__(
        self,
        base_config: PipelineConfig,
        variants: Sequence[VariantSpec] = STANDARD_VARIANTS,
        n_train: int | None = None,
        seed: int = 0,
        experiment_name: str = "experiment",
        output_root: str | Path | None = None,
        evaluator: AnomalyEvaluator | None = None,
        latency_images: int = 10,
    ):
        self.base = base_config
        self.variants = tuple(variants)
        self.n_train = n_train
        self.seed = seed
        self.name = experiment_name
        self.output_root = Path(output_root) if output_root else OUTPUTS_DIR
        self.evaluator = evaluator or AnomalyEvaluator(seed=seed)
        self.latency_images = latency_images
        self.pipelines: dict[str, dict[str, AnomalyPipeline]] = {}      # category -> prekey -> pipeline
        self._mask_cache: dict[Path, np.ndarray] = {}

    # ------------------------------------------------------------- helpers
    def _alpha(self, v: VariantSpec) -> float:
        return self.base.detector.alpha if v.multiscale else 1.0

    def _thr_cfg(self, v: VariantSpec) -> ThresholdConfig:
        t = self.base.threshold
        return ThresholdConfig(method=v.threshold, fixed_value=t.fixed_value, k=t.k,
                               percentile=t.percentile, min_area=t.min_area)

    def _prekey(self, v: VariantSpec) -> str:
        return json.dumps(variant_config(self.base, v).preprocess.__dict__, sort_keys=True, default=str)

    # ----------------------------------------------------------------- fit
    def fit_pipelines(self, data: CategoryData) -> dict[str, AnomalyPipeline]:
        """Fit one pipeline per distinct preprocessing setting on normal train images."""
        paths = select_train_paths(data.train_paths, self.n_train, self.seed)
        size = self.base.features.image_size
        images = [resize_square(load_image_rgb(p), size) for p in paths]
        out: dict[str, AnomalyPipeline] = {}
        shared_extractor = None
        for v in self.variants:
            key = self._prekey(v)
            if key in out:
                continue
            pipe = AnomalyPipeline(variant_config(self.base, v), extractor=shared_extractor)
            shared_extractor = pipe.extractor       # load heavy backends only once
            log.info("[%s] fitting preprocess=%s on %d normal images", data.name, v.preprocess, len(images))
            pipe.fit(images)
            out[key] = pipe
        self.pipelines[data.name] = out
        return out

    # ------------------------------------------------------------ evaluate
    def _load_test(self, data: CategoryData):
        size = self.base.features.image_size
        imgs, labels, masks, types = [], [], [], []
        for s in data.test_samples:
            try:
                imgs.append(resize_square(load_image_rgb(s.image_path), size))
            except ImageLoadError as exc:
                log.warning("skipping unreadable test image: %s", exc)
                continue
            labels.append(s.label)
            types.append(s.defect_type)
            if data.has_masks:
                if s.mask_path is None:
                    masks.append(np.zeros((size, size), bool))
                else:
                    if s.mask_path not in self._mask_cache:
                        self._mask_cache[s.mask_path] = load_mask(s.mask_path, size)
                    masks.append(self._mask_cache[s.mask_path])
        return imgs, np.array(labels), (np.stack(masks) if masks else None), types

    def measure_latency(self, pipelines: dict[str, AnomalyPipeline], images: list[np.ndarray]) -> dict[str, float]:
        """Mean end-to-end ms/image for each variant using only the scales it needs."""
        imgs = images[: self.latency_images]
        out = {}
        for v in self.variants:
            pipe, a, thr = pipelines[self._prekey(v)], self._alpha(v), self._thr_cfg(v)
            pipe.predict(imgs[0], alpha=a, thr_cfg=thr)                  # warm-up / threshold cache
            t0 = time.perf_counter()
            for im in imgs:
                pipe.predict(im, alpha=a, thr_cfg=thr)
            out[v.key] = 1000 * (time.perf_counter() - t0) / len(imgs)
        return out

    def evaluate_category(
        self, data: CategoryData, pipelines: dict[str, AnomalyPipeline],
        transform_name: str = "identity", transform: Transform = identity,
        latency: dict[str, float] | None = None, test_cache=None,
    ) -> tuple[list[dict], list[dict]]:
        imgs, labels, gt, _ = test_cache if test_cache is not None else self._load_test(data)
        if len(imgs) == 0:
            raise ValueError(f"{data.name}: no readable test images")
        timg = [transform(im, self.seed * 100003 + i) for i, im in enumerate(imgs)]

        # group variants by (preprocessing, alpha): maps are computed once per group
        groups: dict[tuple, list[VariantSpec]] = {}
        for v in self.variants:
            groups.setdefault((self._prekey(v), self._alpha(v)), []).append(v)
        prekeys = {k for k, _ in groups}
        need_both = {pk: any(a < 1.0 for (p, a) in groups if p == pk) for pk in prekeys}
        scales = {pk: ("fine", "coarse") if need_both[pk] else ("fine",) for pk in prekeys}

        maps: dict[tuple, list[np.ndarray]] = {g: [] for g in groups}
        d = self.base.detector
        for im in timg:
            analyses = {pk: pipelines[pk].analyze(im, scales[pk]) for pk in prekeys}
            for (pk, a) in groups:
                maps[(pk, a)].append(pipelines[pk].fused_map(analyses[pk], a))

        rows, size_rows = [], []
        ts = iso_now()
        for (pk, a), vs in groups.items():
            M = np.stack(maps[(pk, a)])
            scores = np.array([image_score_from_map(m, d.image_score, d.topk_frac) for m in M])
            pipe = pipelines[pk]
            for v in vs:
                thr = pipe.threshold_for(self._thr_cfg(v), a)
                res = self.evaluator.evaluate(labels, scores, M, thr.threshold, gt, self.base.threshold.min_area)
                pc = pipe.config.preprocess
                rows.append({
                    "experiment": self.name, "timestamp": ts, "category": data.name,
                    "transform": transform_name, "variant": v.key, "method": v.name,
                    "feature_backend": pipe.extractor.label,
                    "preprocess_enabled": v.preprocess,
                    "clahe_mode": pc.clahe_mode if v.preprocess else "off",
                    "flatfield": pc.flatfield and v.preprocess, "lum_norm": pc.lum_norm and v.preprocess,
                    "multiscale": v.multiscale, "alpha": a, "image_score_mode": d.image_score,
                    "threshold_method": v.threshold, "threshold_value": thr.threshold,
                    "calibration_mode": pipe.detector.calibration_mode_,
                    "n_train": pipe.fit_info.get("n_train_images"),
                    "n_calibration": int(len(next(iter(pipe.detector.calib_raw_.values())))),
                    "n_test_normal": int((labels == 0).sum()), "n_test_defective": int((labels == 1).sum()),
                    "mean_inference_ms": (latency or {}).get(v.key, float("nan")),
                    **{k: res.get(k, float("nan")) for k in METRIC_COLUMNS},
                    "pixel_metrics_available": res["pixel_metrics_available"],
                })
                if gt is not None and (labels == 1).any():
                    for r in self.evaluator.evaluate_by_size(labels, scores, M, gt, thr.threshold, self.base.threshold.min_area):
                        size_rows.append({"experiment": self.name, "category": data.name, "transform": transform_name,
                                          "variant": v.key, "method": v.name, **r})
        return rows, size_rows

    # ----------------------------------------------------------------- run
    def run(
        self,
        dataset_root: str | Path,
        categories: Sequence[str] | None = None,
        transforms: dict[str, Transform] | None = None,
        save: bool = True,
        progress: Callable[[str, float], None] | None = None,
    ) -> ExperimentResult:
        root = Path(dataset_root)
        cats = list(categories) if categories else (discover_categories(root) or [None])
        transforms = transforms or {"identity": identity}
        all_rows, all_size, fit_infos = [], [], {}
        total = len(cats) * len(transforms)
        done = 0
        for cat in cats:
            data = load_category(root, cat)
            pipelines = self.fit_pipelines(data)
            fit_infos[data.name] = {f"pipeline_{i}": p.fit_info for i, p in enumerate(pipelines.values())}
            cache = self._load_test(data)
            latency = self.measure_latency(pipelines, cache[0])
            for tname, tfn in transforms.items():
                if progress:
                    progress(f"{data.name} / {tname}", done / total)
                rows, srows = self.evaluate_category(data, pipelines, tname, tfn, latency, cache)
                all_rows += rows
                all_size += srows
                done += 1
        df, sdf = pd.DataFrame(all_rows), pd.DataFrame(all_size)
        meta = {
            "experiment": self.name, "timestamp": iso_now(), "dataset_root": str(root), "categories": cats,
            "seed": self.seed, "n_train_requested": self.n_train,
            "base_config": self.base.to_dict(), "variants": [asdict(v) for v in self.variants],
            "transforms": list(transforms), "fit_info": fit_infos,
            "environment": {"python": platform.python_version(), "platform": platform.platform(),
                            "numpy": np.__version__, "pandas": pd.__version__},
            "note": "All numbers come from this run; thresholds were calibrated on held-out normal TRAIN images only.",
        }
        res = ExperimentResult(df, sdf, None, meta)
        if save:
            res.out_dir = self.save(res)
        return res

    def save(self, res: ExperimentResult) -> Path:
        out = ensure_dir(self.output_root / f"{self.name}_{timestamp()}")
        res.results.to_csv(out / "results.csv", index=False)
        if not res.size_results.empty:
            res.size_results.to_csv(out / "size_analysis.csv", index=False)
        (out / "config.json").write_text(json.dumps(res.meta, indent=2, default=str), encoding="utf-8")
        summarize(res.results).to_csv(out / "summary.csv")
        return out


def summarize(df: pd.DataFrame, transform: str = "identity") -> pd.DataFrame:
    """Macro-average over categories of every metric, per variant (NaNs ignored)."""
    if df.empty:
        return df
    sub = df[df["transform"] == transform] if "transform" in df else df
    cols = [c for c in METRIC_COLUMNS + ["mean_inference_ms", "threshold_value"] if c in sub]
    g = sub.groupby(["variant", "method"], sort=True)[cols].mean(numeric_only=True)
    g["n_categories"] = sub.groupby(["variant", "method"])["category"].nunique()
    return g
