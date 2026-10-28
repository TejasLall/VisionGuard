"""Turn experiment DataFrames into plots and markdown tables (real numbers only)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.experiments.robustness import add_deltas
from src.experiments.runner import summarize
from src.visualization import plots


def df_to_markdown(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    """Minimal GitHub-markdown table writer (no extra dependency)."""
    d = df.reset_index() if df.index.name or isinstance(df.index, pd.MultiIndex) else df
    cols = [str(c) for c in d.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in d.iterrows():
        cells = []
        for v in row:
            if isinstance(v, (float, np.floating)):
                cells.append("n/a" if np.isnan(v) else floatfmt.format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def make_experiment_plots(results: pd.DataFrame, size_results: pd.DataFrame, out_dir: Path) -> list[Path]:
    paths = [plots.save_method_comparison(results, out_dir / "method_comparison.png")]
    has_pixel = results["pixel_metrics_available"].any() if "pixel_metrics_available" in results else False
    if has_pixel and not size_results.empty:
        paths.append(plots.save_size_plot(size_results, out_dir / "size_pixel_recall.png", "pixel_recall"))
        paths.append(plots.save_size_plot(size_results, out_dir / "size_image_recall.png", "image_recall"))
    return paths


def make_robustness_plots(results: pd.DataFrame, out_dir: Path) -> list[Path]:
    return [
        plots.save_robustness_plot(results, "image_auroc", out_dir / "robustness_image_auroc.png"),
        plots.save_robustness_plot(results, "pixel_auroc", out_dir / "robustness_pixel_auroc.png"),
        plots.save_robustness_plot(results, "image_f1", out_dir / "robustness_image_f1.png"),
        plots.save_robustness_plot(results, "image_fpr", out_dir / "robustness_image_fpr.png"),
    ]


def write_summary_markdown(results: pd.DataFrame, size_results: pd.DataFrame, out_dir: Path, title: str,
                           robustness: bool = False) -> Path:
    """Write summary.md containing the per-variant means (and per-category tables)."""
    parts = [f"# {title}", "", "All values below were produced by this run (see config.json for settings).", ""]
    if robustness:
        res = add_deltas(results)
        piv = res.groupby(["transform", "variant"])[["image_auroc", "pixel_auroc", "image_f1", "image_fpr",
                                                       "delta_image_auroc", "delta_pixel_auroc"]].mean()
        parts += ["## Mean over categories per transform and variant", "", df_to_markdown(piv), ""]
    else:
        parts += ["## Mean over categories (identity transform)", "", df_to_markdown(summarize(results)), ""]
        cols = ["category", "variant", "image_auroc", "pixel_auroc", "image_f1", "pixel_f1", "threshold_value"]
        parts += ["## Per category", "", df_to_markdown(results[results["transform"] == "identity"][cols]), ""]
    if not size_results.empty:
        s = size_results[size_results["transform"] == "identity"]
        g = s.groupby(["size_bin", "variant"])[["n_defective", "image_auroc", "image_recall", "pixel_auroc", "pixel_recall"]].agg(
            {"n_defective": "sum", "image_auroc": "mean", "image_recall": "mean", "pixel_auroc": "mean", "pixel_recall": "mean"})
        parts += ["## By defect size (small <1% / medium 1-5% / large >=5% of image area)", "", df_to_markdown(g), ""]
    p = out_dir / "summary.md"
    p.write_text("\n".join(parts), encoding="utf-8")
    return p
