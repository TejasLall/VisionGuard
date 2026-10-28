"""Metric primitives (thin, explicit wrappers around scikit-learn).

Conventions: label 1 = anomalous/defective, label 0 = normal.  Metrics that are
undefined (e.g. AUROC with a single class) return ``nan`` instead of a made-up value.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

NAN = float("nan")


def safe_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int).ravel()
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return NAN
    return float(roc_auc_score(y_true, np.asarray(y_score, dtype=np.float64).ravel()))


def safe_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int).ravel()
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return NAN
    return float(average_precision_score(y_true, np.asarray(y_score, dtype=np.float64).ravel()))


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    t = np.asarray(y_true).astype(bool).ravel()
    p = np.asarray(y_pred).astype(bool).ravel()
    return {
        "tp": int(np.sum(t & p)), "fp": int(np.sum(~t & p)),
        "fn": int(np.sum(t & ~p)), "tn": int(np.sum(~t & ~p)),
    }


def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Precision, recall, F1 and false-positive rate; 0/0 cases are reported as nan (precision)
    or 0 (recall/F1 follow the usual convention)."""
    c = confusion_counts(y_true, y_pred)
    tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
    precision = tp / (tp + fp) if (tp + fp) else NAN
    recall = tp / (tp + fn) if (tp + fn) else NAN
    if np.isnan(precision) or np.isnan(recall) or (precision + recall) == 0:
        f1 = 0.0 if (tp + fp + fn) else NAN
    else:
        f1 = 2 * precision * recall / (precision + recall)
    fpr = fp / (fp + tn) if (fp + tn) else NAN
    return {"precision": precision, "recall": recall, "f1": f1, "fpr": fpr, **c}


def best_f1(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    """ORACLE: maximum F1 over all thresholds and the threshold achieving it.

    Uses the labels to pick the threshold, so it is only an upper-bound reference and
    must never be presented as a deployable result.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    if y_true.sum() == 0 or y_true.sum() == y_true.size:
        return NAN, NAN
    p, r, t = precision_recall_curve(y_true, np.asarray(y_score, dtype=np.float64).ravel())
    f = 2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-12)
    i = int(np.argmax(f))
    return float(f[i]), float(t[i])


def ranking_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    """AUROC, average precision and (oracle) best F1 from ONE sort - for millions of pixels.

    Uses the same definitions as scikit-learn (ROC: trapezoidal rule over distinct thresholds;
    AP: sum over recall steps of precision), so results match ``roc_auc_score`` /
    ``average_precision_score`` up to floating-point error while sorting only once.
    """
    y = np.asarray(y_true).astype(bool).ravel()
    s = np.asarray(y_score, dtype=np.float64).ravel()
    n_pos = int(y.sum())
    if y.size == 0 or n_pos == 0 or n_pos == y.size:
        return {"auroc": NAN, "ap": NAN, "best_f1": NAN}
    order = np.argsort(-s)
    s_sorted, y_sorted = s[order], y[order]
    last = np.r_[np.flatnonzero(np.diff(s_sorted)), y.size - 1]   # last index of each distinct score
    tps = np.cumsum(y_sorted)[last].astype(np.float64)
    fps = (last + 1) - tps
    tpr, fpr = tps / n_pos, fps / (y.size - n_pos)
    auroc = float(np.trapezoid(np.r_[0.0, tpr], np.r_[0.0, fpr]))
    precision = tps / (tps + fps)
    ap = float(np.sum(np.diff(np.r_[0.0, tpr]) * precision))
    f1 = 2 * precision * tpr / np.maximum(precision + tpr, 1e-12)
    return {"auroc": auroc, "ap": ap, "best_f1": float(f1.max())}


def roc_points(y_true, y_score):
    fpr, tpr, _ = roc_curve(np.asarray(y_true).astype(int), np.asarray(y_score, dtype=np.float64))
    return fpr, tpr


def pr_points(y_true, y_score):
    p, r, _ = precision_recall_curve(np.asarray(y_true).astype(int), np.asarray(y_score, dtype=np.float64))
    return p, r


def class_distribution(labels: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels).astype(int)
    n = int(labels.size)
    pos = int(labels.sum())
    return {"n": n, "n_anomalous": pos, "n_normal": n - pos, "anomalous_fraction": pos / n if n else NAN}


def subsample_pixels(y: np.ndarray, s: np.ndarray, max_pixels: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Seeded, class-stratified sub-sampling used only when there are more than ``max_pixels``."""
    if y.size <= max_pixels:
        return y, s
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y), np.flatnonzero(~y)
    frac = max_pixels / y.size
    keep = np.concatenate([
        rng.choice(pos, max(1, int(len(pos) * frac)), replace=False) if len(pos) else pos,
        rng.choice(neg, max(1, int(len(neg) * frac)), replace=False) if len(neg) else neg,
    ])
    return y[keep], s[keep]
