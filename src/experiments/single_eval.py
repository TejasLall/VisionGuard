"""Evaluate one fitted pipeline on a category's test set with a chosen operating point."""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from src.config import ThresholdConfig
from src.data.dataset_loader import CategoryData
from src.data.image_io import ImageLoadError, load_image_rgb, load_mask, resize_square
from src.evaluation.evaluator import AnomalyEvaluator
from src.experiments.robustness import Transform, identity
from src.models.multiscale_detector import image_score_from_map
from src.models.pipeline import AnomalyPipeline


@dataclass
class EvalOutput:
    metrics: dict
    labels: np.ndarray
    scores: np.ndarray
    maps: np.ndarray
    gt: np.ndarray | None
    size_rows: list[dict]
    defect_types: list[str]
    mean_inference_ms: float


def evaluate_pipeline(
    pipe: AnomalyPipeline, data: CategoryData, alpha: float | None = None,
    thr_cfg: ThresholdConfig | None = None, transform: Transform = identity,
    max_images: int | None = None, seed: int = 0,
) -> EvalOutput:
    """Run the pipeline on test images (labels/masks are used only for scoring the result)."""
    size = pipe.size
    samples = data.test_samples[:max_images] if max_images else data.test_samples
    a = pipe.effective_alpha(alpha)
    thr = pipe.threshold_for(thr_cfg, a)
    c, d = thr_cfg or pipe.config.threshold, pipe.config.detector
    maps, labels, types, gts = [], [], [], []
    t_total = 0.0
    for i, s in enumerate(samples):
        try:
            img = resize_square(load_image_rgb(s.image_path), size)
        except ImageLoadError:
            continue
        img = transform(img, seed * 100003 + i)
        t0 = time.perf_counter()
        an = pipe.analyze(img, pipe.needed_scales(a))
        maps.append(pipe.fused_map(an, a))
        t_total += time.perf_counter() - t0
        labels.append(s.label)
        types.append(s.defect_type)
        if data.has_masks:
            gts.append(np.zeros((size, size), bool) if s.mask_path is None else load_mask(s.mask_path, size))
    if not maps:
        raise ValueError("no readable test images")
    M = np.stack(maps)
    scores = np.array([image_score_from_map(m, d.image_score, d.topk_frac) for m in M])
    gt = np.stack(gts) if gts else None
    labels_a = np.array(labels)
    ev = AnomalyEvaluator(seed=seed)
    metrics = ev.evaluate(labels_a, scores, M, thr.threshold, gt, c.min_area)
    metrics["threshold_description"] = thr.describe()
    size_rows = []
    if gt is not None and (labels_a == 1).any():
        size_rows = ev.evaluate_by_size(labels_a, scores, M, gt, thr.threshold, c.min_area)
    return EvalOutput(metrics, labels_a, scores, M, gt, size_rows, types, 1000 * t_total / len(maps))
