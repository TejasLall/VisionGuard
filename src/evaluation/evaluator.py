"""AnomalyEvaluator: turns per-image scores/maps into the reported metrics.

Ground-truth masks are consumed here and only here - never by fitting or
threshold calibration.

Pixel-level metrics are micro-averaged over all pixels of all test images
(normal images contribute all-negative pixels) at the analysis resolution.

Defect size bins (fraction of image area covered by the ground-truth mask),
fixed a priori and independent of any result:
    small  : < 1 %
    medium : 1 % - 5 %
    large  : >= 5 %
"""
from __future__ import annotations

import numpy as np

from src.evaluation import metrics as M
from src.models.thresholding import apply_threshold

SIZE_BINS: tuple[tuple[str, float, float], ...] = (
    ("small", 0.0, 0.01),
    ("medium", 0.01, 0.05),
    ("large", 0.05, float("inf")),
)


def size_bin_of(fraction: float) -> str:
    for name, lo, hi in SIZE_BINS:
        if lo <= fraction < hi:
            return name
    return SIZE_BINS[-1][0]


class AnomalyEvaluator:
    def __init__(self, max_pixels: int = 4_000_000, seed: int = 0):
        self.max_pixels = max_pixels
        self.seed = seed

    # ---------------------------------------------------------------- core
    def evaluate(
        self,
        labels: np.ndarray,
        image_scores: np.ndarray,
        maps: np.ndarray,
        threshold: float,
        gt_masks: np.ndarray | None = None,
        min_area: int = 0,
    ) -> dict:
        """Compute image-level (always) and pixel-level (if ``gt_masks``) metrics.

        labels      (N,)       0/1 image labels
        image_scores(N,)       image-level anomaly scores
        maps        (N,H,W)    anomaly maps
        gt_masks    (N,H,W)    bool ground truth (None -> pixel metrics are NaN / not reported)
        """
        labels = np.asarray(labels).astype(int)
        image_scores = np.asarray(image_scores, dtype=np.float64)
        out: dict = {f"image_{k}": v for k, v in M.class_distribution(labels).items() if k != "n"}
        out["n_test"] = int(labels.size)
        out["threshold"] = float(threshold)

        out["image_auroc"] = M.safe_auroc(labels, image_scores)
        out["image_ap"] = M.safe_average_precision(labels, image_scores)
        # same decision rule as AnomalyPipeline.predict: with blob filtering, "anomalous" = non-empty cleaned mask
        if min_area > 0:
            pred = np.array([apply_threshold(m, threshold, min_area).any() for m in maps])
        else:
            pred = image_scores >= threshold
        prf = M.precision_recall_f1(labels, pred)
        out.update({f"image_{k}": v for k, v in prf.items()})
        out["image_f1_oracle"], _ = M.best_f1(labels, image_scores)   # reference only

        out["pixel_metrics_available"] = gt_masks is not None
        pix_keys = ["pixel_auroc", "pixel_ap", "pixel_precision", "pixel_recall", "pixel_f1", "pixel_f1_oracle"]
        if gt_masks is None:
            out.update({k: M.NAN for k in pix_keys})
            return out

        gt = np.asarray(gt_masks).astype(bool)
        flat_gt, flat_s = gt.ravel(), np.asarray(maps, dtype=np.float32).ravel()
        sub_y, sub_s = M.subsample_pixels(flat_gt, flat_s, self.max_pixels, self.seed)
        rk = M.ranking_metrics(sub_y, sub_s)
        out["pixel_auroc"], out["pixel_ap"], out["pixel_f1_oracle"] = rk["auroc"], rk["ap"], rk["best_f1"]
        pm = np.stack([apply_threshold(m, threshold, min_area) for m in maps])
        pprf = M.precision_recall_f1(flat_gt, pm.ravel())
        out.update({f"pixel_{k}": pprf[k] for k in ("precision", "recall", "f1")})
        out["pixel_anomalous_fraction"] = float(flat_gt.mean())
        return out

    # ------------------------------------------------------------ size study
    def evaluate_by_size(
        self,
        labels: np.ndarray,
        image_scores: np.ndarray,
        maps: np.ndarray,
        gt_masks: np.ndarray,
        threshold: float,
        min_area: int = 0,
    ) -> list[dict]:
        """Metrics per defect-size bin: the bin's defective images plus ALL normal test images."""
        labels = np.asarray(labels).astype(int)
        gt = np.asarray(gt_masks).astype(bool)
        frac = gt.reshape(len(gt), -1).mean(axis=1)
        normal = np.flatnonzero(labels == 0)
        rows = []
        for name, lo, hi in SIZE_BINS:
            idx_def = np.flatnonzero((labels == 1) & (frac >= lo) & (frac < hi))
            row = {"size_bin": name, "area_range": f"[{lo:g}, {hi:g})", "n_defective": int(len(idx_def)),
                   "n_normal": int(len(normal))}
            if len(idx_def) == 0:
                row.update({"image_auroc": M.NAN, "image_recall": M.NAN, "pixel_auroc": M.NAN, "pixel_recall": M.NAN})
                rows.append(row)
                continue
            idx = np.concatenate([normal, idx_def])
            lab = labels[idx]
            row["image_auroc"] = M.safe_auroc(lab, image_scores[idx])
            row["image_recall"] = float(np.mean(image_scores[idx_def] >= threshold))
            y, s = M.subsample_pixels(gt[idx].ravel(), np.asarray(maps[idx], np.float32).ravel(), self.max_pixels, self.seed)
            row["pixel_auroc"] = M.ranking_metrics(y, s)["auroc"]
            pm = np.stack([apply_threshold(maps[i], threshold, min_area) for i in idx_def])
            if min_area > 0:
                row["image_recall"] = float(pm.reshape(len(pm), -1).any(axis=1).mean())
            row["pixel_recall"] = float(pm[gt[idx_def]].mean()) if gt[idx_def].any() else M.NAN
            rows.append(row)
        return rows
