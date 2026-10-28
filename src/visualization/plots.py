"""Matplotlib / OpenCV rendering helpers.  All figures are saved with labelled axes."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluation import metrics as M
from src.utils.paths import ensure_dir

# Okabe-Ito colour-blind-safe palette, one colour per method variant.
VARIANT_COLORS = {"A": "#6B6B6B", "B": "#E69F00", "C": "#56B4E9", "D": "#009E73", "E": "#D55E00"}
_DEFAULT_COLORS = ["#6B6B6B", "#E69F00", "#56B4E9", "#009E73", "#D55E00", "#CC79A7", "#0072B2", "#F0E442"]


def _color(key: str, i: int) -> str:
    return VARIANT_COLORS.get(key, _DEFAULT_COLORS[i % len(_DEFAULT_COLORS)])


# ----------------------------------------------------------------- array renderers
def display_range(heat: np.ndarray, threshold: float | None) -> tuple[float, float]:
    """Fixed display scale: 0 (typical normal, z-units) to max(1.5*T, observed max)."""
    hi = float(heat.max())
    if threshold is not None and np.isfinite(threshold):
        hi = max(hi, 1.5 * abs(threshold))
    return 0.0, max(hi, 1e-6)


def render_heatmap(heat: np.ndarray, threshold: float | None = None) -> np.ndarray:
    """Heat map as RGB uint8 (inferno: dark = normal, bright yellow = most anomalous)."""
    lo, hi = display_range(heat, threshold)
    g = np.clip((heat - lo) / (hi - lo), 0, 1)
    bgr = cv2.applyColorMap((g * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def render_mask(mask: np.ndarray) -> np.ndarray:
    """Binary mask as a single-channel uint8 image (255 = predicted defect)."""
    return (mask.astype(np.uint8)) * 255


def render_overlay(image: np.ndarray, heat: np.ndarray, mask: np.ndarray, threshold: float | None = None,
                   alpha: float = 0.45) -> np.ndarray:
    """Heat map blended over the image with the predicted-defect outline in cyan."""
    hm = render_heatmap(heat, threshold)
    lo, hi = display_range(heat, threshold)
    w = (np.clip((heat - lo) / (hi - lo), 0, 1) ** 0.8)[..., None] * alpha * 2   # more heat -> more opaque
    w = np.clip(w, 0, 0.85)
    out = (image.astype(np.float32) * (1 - w) + hm.astype(np.float32) * w).astype(np.uint8)
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, (0, 255, 255), 1)
    return out


def render_legend(threshold: float | None, heat: np.ndarray, width: int = 360, height: int = 54) -> np.ndarray:
    """Colour-bar legend image: normal -> anomalous with the decision threshold marked."""
    lo, hi = display_range(heat, threshold)
    bar = np.tile(np.linspace(0, 255, width, dtype=np.uint8), (22, 1))
    bar = cv2.cvtColor(cv2.applyColorMap(bar, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)
    canvas = np.full((height, width, 3), 255, np.uint8)
    canvas[4:26] = bar
    if threshold is not None:
        x = int(np.clip((threshold - lo) / (hi - lo), 0, 1) * (width - 1))
        cv2.line(canvas, (x, 2), (x, 28), (0, 200, 255), 2)
        cv2.putText(canvas, f"T={threshold:.2f}", (min(max(x - 24, 0), width - 62), 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 120, 160), 1, cv2.LINE_AA)
    cv2.putText(canvas, "normal", (2, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 60, 60), 1, cv2.LINE_AA)
    cv2.putText(canvas, "anomalous", (width - 66, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 60, 60), 1, cv2.LINE_AA)
    return canvas


def _save(fig, path: str | Path) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return p


# ------------------------------------------------------------------ figures
def save_before_after(original: np.ndarray, processed: np.ndarray, path: str | Path, title: str = "") -> Path:
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.8))
    ax[0].imshow(original); ax[0].set_title("Original")
    ax[1].imshow(processed); ax[1].set_title("After illumination-aware preprocessing")
    for a, im, c in ((ax[2], original, "tab:blue"), (ax[2], processed, "tab:orange")):
        L = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        a.hist(L.ravel(), bins=64, range=(0, 255), alpha=0.5, color=c, label="original" if c == "tab:blue" else "processed")
    ax[2].set_title("Luminance histogram"); ax[2].set_xlabel("grey level (0-255)"); ax[2].set_ylabel("pixel count"); ax[2].legend(loc="upper right")
    for a in ax[:2]:
        a.axis("off")
    if title:
        fig.suptitle(title)
    return _save(fig, path)


def save_inspection_figure(pred, path: str | Path, gt_mask: np.ndarray | None = None, title: str = "") -> Path:
    """Original | preprocessed | heat map | ground truth (if any) | predicted mask | overlay."""
    panels = [("Original", pred.image), ("Preprocessed", pred.processed)]
    hm = render_heatmap(pred.heatmap, pred.threshold)
    panels.append(("Anomaly heat map", hm))
    if gt_mask is not None:
        panels.append(("Ground-truth mask (eval only)", gt_mask.astype(np.uint8) * 255))
    panels += [(f"Predicted mask (T={pred.threshold:.2f})", render_mask(pred.mask)),
               ("Overlay", render_overlay(pred.image, pred.heatmap, pred.mask, pred.threshold))]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.0 * len(panels), 3.6))
    for ax, (name, im) in zip(axes, panels):
        ax.imshow(im, cmap="gray" if im.ndim == 2 else None, vmin=0, vmax=255 if im.ndim == 2 else None)
        ax.set_title(name, fontsize=9); ax.axis("off")
    verdict = "ANOMALOUS" if pred.is_anomalous else "normal"
    fig.suptitle(title or f"image score {pred.image_score:.2f} (max) -> {verdict}", fontsize=10)
    return _save(fig, path)


def save_roc_pr(labels, scores, path: str | Path, title: str = "Image-level") -> Path:
    labels = np.asarray(labels).astype(int)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.8))
    if len(np.unique(labels)) < 2:
        for a in ax:
            a.text(0.5, 0.5, "needs both normal and defective samples", ha="center"); a.axis("off")
        return _save(fig, path)
    fpr, tpr = M.roc_points(labels, scores)
    ax[0].plot(fpr, tpr, color="#D55E00", label=f"AUROC = {M.safe_auroc(labels, scores):.3f}")
    ax[0].plot([0, 1], [0, 1], "--", color="gray", label="chance")
    ax[0].set_xlabel("False positive rate"); ax[0].set_ylabel("True positive rate"); ax[0].set_title(f"ROC - {title}"); ax[0].legend(loc="lower right")
    p, r = M.pr_points(labels, scores)
    ax[1].plot(r, p, color="#0072B2", label=f"AP = {M.safe_average_precision(labels, scores):.3f}")
    ax[1].axhline(labels.mean(), ls="--", color="gray", label=f"prevalence = {labels.mean():.2f}")
    ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precision"); ax[1].set_title(f"Precision-recall - {title}"); ax[1].legend(loc="lower left")
    return _save(fig, path)


def _variant_order(df: pd.DataFrame) -> list[tuple[str, str]]:
    return list(df[["variant", "method"]].drop_duplicates().sort_values("variant").itertuples(index=False, name=None))


def save_method_comparison(df: pd.DataFrame, path: str | Path, metrics=("image_auroc", "pixel_auroc", "image_f1", "pixel_f1"),
                           transform: str = "identity") -> Path:
    """Grouped bars: macro-average over categories for each metric and method variant."""
    sub = df[df["transform"] == transform] if "transform" in df else df
    variants = _variant_order(sub)
    fig, ax = plt.subplots(figsize=(1.9 * len(metrics) + 2, 4.2))
    width = 0.8 / max(len(variants), 1)
    for i, (key, name) in enumerate(variants):
        vals = [sub[sub["variant"] == key][m].mean() for m in metrics]
        ax.bar(np.arange(len(metrics)) + i * width, vals, width, label=f"{key}: {name}", color=_color(key, i))
    ax.set_xticks(np.arange(len(metrics)) + width * (len(variants) - 1) / 2)
    ax.set_xticklabels([m.replace("_", " ") for m in metrics])
    ax.set_ylabel("score (mean over categories)"); ax.set_ylim(0, 1.05)
    ax.set_title(f"Method comparison ({sub['category'].nunique()} categor{'y' if sub['category'].nunique()==1 else 'ies'})")
    ax.legend(fontsize=7, loc="lower center", bbox_to_anchor=(0.5, -0.45), ncol=2)
    return _save(fig, path)


def save_robustness_plot(df: pd.DataFrame, metric: str, path: str | Path) -> Path:
    """Metric of every variant under every synthetic transform (mean over categories)."""
    transforms = list(dict.fromkeys(df["transform"]))
    variants = _variant_order(df)
    fig, ax = plt.subplots(figsize=(min(1.4 * len(transforms) + 2, 14), 4.4))
    width = 0.8 / max(len(variants), 1)
    for i, (key, name) in enumerate(variants):
        vals = [df[(df["variant"] == key) & (df["transform"] == t)][metric].mean() for t in transforms]
        ax.bar(np.arange(len(transforms)) + i * width, vals, width, label=f"{key}: {name}", color=_color(key, i))
    ax.set_xticks(np.arange(len(transforms)) + width * (len(variants) - 1) / 2)
    ax.set_xticklabels(transforms, rotation=30, ha="right")
    ax.set_ylabel(metric.replace("_", " ")); ax.set_xlabel("synthetic illumination / noise transform (test images only)")
    ax.set_ylim(0, 1.05); ax.set_title(f"Robustness: {metric.replace('_', ' ')} (synthetic perturbations)")
    ax.legend(fontsize=7, loc="lower center", bbox_to_anchor=(0.5, -0.6), ncol=2)
    return _save(fig, path)


def save_size_plot(size_df: pd.DataFrame, path: str | Path, metric: str = "pixel_recall") -> Path:
    """Metric per defect-size bin and method variant (identity transform only)."""
    sub = size_df[size_df.get("transform", "identity") == "identity"] if "transform" in size_df else size_df
    bins = ["small", "medium", "large"]
    variants = _variant_order(sub)
    fig, ax = plt.subplots(figsize=(7.5, 4))
    width = 0.8 / max(len(variants), 1)
    for i, (key, name) in enumerate(variants):
        vals = [sub[(sub["variant"] == key) & (sub["size_bin"] == b)][metric].mean() for b in bins]
        ax.bar(np.arange(3) + i * width, vals, width, label=f"{key}: {name}", color=_color(key, i))
    counts = sub.groupby("size_bin")["n_defective"].sum() / max(sub["variant"].nunique(), 1)
    ax.set_xticks(np.arange(3) + width * (len(variants) - 1) / 2)
    ax.set_xticklabels([f"{b}\n(n={int(counts.get(b, 0))})" for b in bins])
    ax.set_xlabel("defect size (fraction of image area: small <1%, medium 1-5%, large >=5%)")
    ax.set_ylabel(metric.replace("_", " ")); ax.set_ylim(0, 1.05)
    ax.set_title(f"{metric.replace('_', ' ')} by defect size")
    ax.legend(fontsize=7, loc="lower center", bbox_to_anchor=(0.5, -0.5), ncol=2)
    return _save(fig, path)


def save_score_distribution(normal_scores, defect_scores, threshold: float, path: str | Path, title: str = "") -> Path:
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.hist(normal_scores, bins=30, alpha=0.6, label="normal test images", color="#0072B2")
    if len(defect_scores):
        ax.hist(defect_scores, bins=30, alpha=0.6, label="defective test images", color="#D55E00")
    ax.axvline(threshold, color="k", ls="--", label=f"threshold T = {threshold:.2f}")
    ax.set_xlabel("image anomaly score (max of fused z-score map)"); ax.set_ylabel("number of images")
    ax.set_title(title or "Image score distribution"); ax.legend(loc="upper right")
    return _save(fig, path)
