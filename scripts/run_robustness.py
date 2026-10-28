"""Illumination robustness study with SYNTHETIC perturbations of the test images.

Models are fitted/calibrated once on clean normal images; every transform is then applied to the test
images only (labels and masks unchanged).  Synthetic perturbations are not a substitute for real lighting
variation.  Use --op-ablation to compare the individual illumination operations (CLAHE, flat-field, ...).

Examples:
    python scripts/run_robustness.py --dataset data/mvtec_ad --categories bottle --n-train 20
    python scripts/run_robustness.py --synthetic --op-ablation
"""
import _bootstrap  # noqa: F401
import argparse
from pathlib import Path

from _cli import add_config_args, build_config

from src.data.dataset_validator import validate_dataset
from src.data.synthetic import create_synthetic_dataset
from src.experiments.reporting import df_to_markdown, make_experiment_plots, make_robustness_plots, write_summary_markdown
from src.experiments.robustness import add_deltas, default_transforms
from src.experiments.runner import ILLUMINATION_ABLATION_VARIANTS, STANDARD_VARIANTS, ExperimentRunner
from src.utils.logging_config import setup_logging
from src.utils.paths import DATA_DIR, OUTPUTS_DIR


def main() -> int:
    setup_logging()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--categories", nargs="*")
    ap.add_argument("--n-train", type=int)
    ap.add_argument("--no-noise", action="store_true", help="skip the sensor-noise transform")
    ap.add_argument("--op-ablation", action="store_true", help="compare individual illumination operations")
    ap.add_argument("--name", default="robustness")
    ap.add_argument("--out", type=Path, default=OUTPUTS_DIR)
    add_config_args(ap)
    a = ap.parse_args()

    if a.synthetic:
        a.dataset = a.dataset or DATA_DIR / "synthetic"
        if not a.dataset.exists():
            create_synthetic_dataset(a.dataset)
        print("NOTE: synthetic data is for software testing only; results say nothing about real performance.")
    if a.dataset is None:
        ap.error("--dataset or --synthetic is required")
    rep = validate_dataset(a.dataset)
    print(rep.summary())
    if not rep.ok:
        return 1

    cfg = build_config(a)
    variants = ILLUMINATION_ABLATION_VARIANTS if a.op_ablation else STANDARD_VARIANTS
    name = a.name + ("_opablation" if a.op_ablation else "")
    runner = ExperimentRunner(cfg, variants=variants, n_train=a.n_train, seed=cfg.seed,
                              experiment_name=name, output_root=a.out)
    res = runner.run(a.dataset, a.categories, transforms=default_transforms(not a.no_noise),
                     progress=lambda msg, f: print(f"  [{100 * f:3.0f}%] {msg}"))
    df = add_deltas(res.results)
    df.to_csv(res.out_dir / "results_with_deltas.csv", index=False)
    make_robustness_plots(df, res.out_dir)
    make_experiment_plots(res.results, res.size_results, res.out_dir)
    write_summary_markdown(res.results, res.size_results, res.out_dir, f"Robustness experiment {name}", robustness=True)
    cols = ["image_auroc", "pixel_auroc", "image_f1", "image_fpr", "delta_image_auroc", "delta_pixel_auroc"]
    print("\nMean over categories per transform/variant:")
    print(df_to_markdown(df.groupby(["transform", "variant"])[cols].mean()))
    print(f"\nResults saved to {res.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
