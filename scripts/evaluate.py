"""Evaluate a saved model on a category's TEST set and save ROC/PR curves, score histogram and examples.

Labels and ground-truth masks are used only to score the result.

Example:
    python scripts/evaluate.py --dataset data/synthetic --category synthetic_texture \
        --model synthetic_texture__handcrafted__illum --multiscale --threshold mean_std
"""
import _bootstrap  # noqa: F401
import argparse
import json
import math
from pathlib import Path

import numpy as np

from src.config import ThresholdConfig
from src.data.dataset_loader import load_category
from src.data.image_io import load_image_rgb
from src.experiments.single_eval import evaluate_pipeline
from src.models.model_manager import ModelManager
from src.utils.logging_config import setup_logging
from src.utils.paths import OUTPUTS_DIR, ensure_dir, timestamp
from src.visualization import plots


def main() -> int:
    setup_logging()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--category")
    ap.add_argument("--model", required=True)
    ap.add_argument("--multiscale", action="store_true")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--threshold", choices=["fixed", "mean_std", "percentile"], default="mean_std")
    ap.add_argument("--k", type=float, default=3.0)
    ap.add_argument("--percentile", type=float, default=99.0)
    ap.add_argument("--fixed-value", type=float, default=3.0)
    ap.add_argument("--n-examples", type=int, default=4)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()

    pipe = ModelManager().load(a.model)
    data = load_category(a.dataset, a.category)
    thr = ThresholdConfig(method=a.threshold, fixed_value=a.fixed_value, k=a.k, percentile=a.percentile)
    alpha = a.alpha if a.multiscale else 1.0
    res = evaluate_pipeline(pipe, data, alpha, thr)
    m = res.metrics
    out = ensure_dir(a.out or OUTPUTS_DIR / f"eval_{data.name}_{timestamp()}")

    print(f"Category {data.name}: {m['n_test']} test images "
          f"({m['image_n_normal']} normal / {m['image_n_anomalous']} defective; "
          f"anomalous fraction {m['image_anomalous_fraction']:.2f})")
    print(f"Backend   : {pipe.extractor.label}")
    print(f"Threshold : {m['threshold_description']}")
    keys = ["image_auroc", "image_ap", "image_precision", "image_recall", "image_f1", "image_fpr"]
    if m["pixel_metrics_available"]:
        keys += ["pixel_auroc", "pixel_ap", "pixel_precision", "pixel_recall", "pixel_f1"]
    else:
        print("Pixel-level metrics: not reported (no ground-truth masks for every defective image).")
    for k in keys:
        v = m[k]
        print(f"  {k:16s} {'n/a' if isinstance(v, float) and math.isnan(v) else f'{v:.4f}'}")
    print(f"  mean inference   {res.mean_inference_ms:.1f} ms / image (analysis only)")
    print("  (accuracy is deliberately not reported: the test set is class-imbalanced)")

    (out / "metrics.json").write_text(json.dumps({k: (None if isinstance(v, float) and math.isnan(v) else v)
                                                  for k, v in m.items()}, indent=2, default=float), encoding="utf-8")
    plots.save_roc_pr(res.labels, res.scores, out / "roc_pr_image.png", f"{data.name} image-level")
    if res.gt is not None:
        idx = np.random.default_rng(0).choice(res.gt.size, min(res.gt.size, 2_000_000), replace=False)
        plots.save_roc_pr(res.gt.ravel()[idx], res.maps.ravel()[idx], out / "roc_pr_pixel.png", f"{data.name} pixel-level")
    plots.save_score_distribution(res.scores[res.labels == 0], res.scores[res.labels == 1], m["threshold"],
                                  out / "score_distribution.png", f"{data.name}: image score distribution")
    defect_idx = np.flatnonzero(res.labels == 1)[:: max(1, int(res.labels.sum()) // max(a.n_examples, 1))][: a.n_examples]
    samples = data.test_samples
    for i in defect_idx:
        pred = pipe.predict(load_image_rgb(samples[i].image_path), alpha=alpha, thr_cfg=thr)
        stem = f"{samples[i].image_path.stem}_{samples[i].defect_type}"
        plots.save_inspection_figure(pred, out / f"example_{stem}.png", gt_mask=None if res.gt is None else res.gt[i])
        if pipe.config.preprocess.enabled:
            plots.save_before_after(pred.image, pred.processed, out / f"preprocessing_{stem}.png",
                                    "Illumination-aware preprocessing (same transform is applied to training images)")
    print(f"Saved results to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
