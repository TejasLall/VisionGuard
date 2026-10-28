"""Run a saved model on one image and write heat map, binary mask and overlay.

Examples:
    python scripts/predict.py --model synthetic_texture__handcrafted__illum --image some.png
    python scripts/predict.py --model bottle__handcrafted__raw --image x.png --multiscale --alpha 0.5 --threshold mean_std --k 3
"""
import _bootstrap  # noqa: F401
import argparse
from pathlib import Path

from src.config import ThresholdConfig
from src.data.image_io import load_image_rgb, save_image_rgb
from src.models.model_manager import ModelManager
from src.utils.logging_config import setup_logging
from src.utils.paths import OUTPUTS_DIR
from src.visualization import plots


def main() -> int:
    setup_logging()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="saved model name (see --list)")
    ap.add_argument("--image", type=Path)
    ap.add_argument("--list", action="store_true", help="list saved models")
    ap.add_argument("--multiscale", action="store_true", help="fuse fine+coarse scales")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--threshold", choices=["fixed", "mean_std", "percentile"], default="mean_std")
    ap.add_argument("--k", type=float, default=3.0)
    ap.add_argument("--percentile", type=float, default=99.0)
    ap.add_argument("--fixed-value", type=float, default=3.0)
    ap.add_argument("--min-area", type=int, default=0)
    ap.add_argument("--out", type=Path, default=OUTPUTS_DIR / "predictions")
    a = ap.parse_args()

    mgr = ModelManager()
    if a.list:
        print("\n".join(mgr.list_models()) or "no saved models in models/")
        return 0
    if a.image is None:
        ap.error("--image is required")
    try:
        pipe = mgr.load(a.model)
        img = load_image_rgb(a.image)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1
    thr = ThresholdConfig(method=a.threshold, fixed_value=a.fixed_value, k=a.k, percentile=a.percentile,
                          min_area=a.min_area)
    pred = pipe.predict(img, alpha=a.alpha if a.multiscale else 1.0, thr_cfg=thr)
    stem = a.image.stem
    out = a.out
    save_image_rgb(out / f"{stem}_heatmap.png", plots.render_heatmap(pred.heatmap, pred.threshold))
    save_image_rgb(out / f"{stem}_mask.png", plots.render_mask(pred.mask))
    save_image_rgb(out / f"{stem}_overlay.png", plots.render_overlay(pred.image, pred.heatmap, pred.mask, pred.threshold))
    plots.save_inspection_figure(pred, out / f"{stem}_panel.png")
    print(f"model     : {a.model}  ({pipe.extractor.label})")
    print(f"scales    : {'fine+coarse (alpha=%.2f)' % pred.alpha if a.multiscale else 'fine only'}")
    print(f"threshold : {pred.threshold_description}")
    print(f"max score : {pred.max_score:.3f}   mean score: {pred.mean_score:.3f}")
    print(f"verdict   : {'ANOMALOUS' if pred.is_anomalous else 'normal'}  "
          f"({100 * pred.mask.mean():.2f}% of pixels flagged)")
    print("note      : scores are z-score-normalised anomaly scores, not calibrated probabilities.")
    print(f"outputs   : {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
