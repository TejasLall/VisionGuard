"""Fit VisionGuard models on NORMAL training images only and save them.

Each category yields two models: `<cat>__<backend>__raw` (no preprocessing) and
`<cat>__<backend>__illum` (illumination-aware preprocessing), so that the
dashboard can compare the variants.  Both contain the fine and coarse scales;
single- vs multi-scale and the threshold method are chosen at prediction time.

Examples:
    python scripts/train.py --dataset data/synthetic --category synthetic_texture
    python scripts/train.py --dataset data/mvtec_ad --category bottle --n-train 20
    python scripts/train.py --dataset path/to/custom_dataset --name my_part
"""
import _bootstrap  # noqa: F401
import argparse
import logging
import time
from pathlib import Path

from _cli import add_config_args, build_config

from src.data.dataset_loader import discover_categories, load_category
from src.data.dataset_validator import validate_dataset
from src.data.image_io import load_image_rgb, resize_square
from src.experiments.runner import select_train_paths
from src.models.model_manager import ModelManager, model_name
from src.models.pipeline import AnomalyPipeline
from src.utils.logging_config import setup_logging

log = logging.getLogger("train")


def main() -> int:
    setup_logging()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--category", nargs="*", help="categories to train (default: all discovered)")
    ap.add_argument("--name", help="model/category name for custom datasets")
    ap.add_argument("--n-train", type=int, help="use a random subset of N normal images (few-shot)")
    ap.add_argument("--only", choices=["raw", "illum"], help="train only one preprocessing variant")
    ap.add_argument("--models-dir", type=Path)
    add_config_args(ap)
    a = ap.parse_args()

    rep = validate_dataset(a.dataset)
    print(rep.summary())
    if not rep.ok:
        return 1
    base = build_config(a)
    base.detector.multiscale = True          # keep both scales; alpha chosen at inference
    cats = a.category or ([None] if rep.layout == "custom" else discover_categories(a.dataset))
    mgr = ModelManager(a.models_dir)
    extractor = None
    for cat in cats:
        data = load_category(a.dataset, cat)
        cname = a.name or data.name
        paths = select_train_paths(data.train_paths, a.n_train, base.seed)
        images = [resize_square(load_image_rgb(p), base.features.image_size) for p in paths]
        for pre in (False, True):
            if a.only and (a.only == "illum") != pre:
                continue
            cfg = base.copy()
            cfg.preprocess.enabled = pre
            pipe = AnomalyPipeline(cfg, extractor=extractor)
            extractor = pipe.extractor
            t0 = time.perf_counter()
            info = pipe.fit(images)
            name = model_name(cname, pipe.config.features.backend, pre)
            folder = mgr.save(pipe, name, {"category": cname, "train_files": [p.name for p in paths],
                                           "dataset": str(a.dataset)})
            print(f"[{cname}] preprocess={pre}: {len(images)} normal images, PCA components {info['pca_components']}, "
                  f"calibration={info['calibration_mode']}, backend='{info['backend_label']}', "
                  f"{time.perf_counter() - t0:.1f}s -> {folder}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
