"""Run the five-variant comparison (A baseline ... E full system) on one or more categories.

Models are fitted on normal TRAIN images; thresholds are calibrated on held-out normal TRAIN images;
the TEST split and its masks are used for evaluation only.

Examples:
    python scripts/run_experiments.py --dataset data/mvtec_ad --categories bottle hazelnut --n-train 20
    python scripts/run_experiments.py --synthetic            (software smoke test, NOT a real benchmark)
"""
import _bootstrap  # noqa: F401
import argparse
from pathlib import Path

from _cli import add_config_args, build_config

from src.data.dataset_validator import validate_dataset
from src.data.synthetic import create_synthetic_dataset
from src.experiments.reporting import df_to_markdown, make_experiment_plots, write_summary_markdown
from src.experiments.runner import ExperimentRunner, summarize
from src.utils.logging_config import setup_logging
from src.utils.paths import DATA_DIR, OUTPUTS_DIR


def main() -> int:
    setup_logging()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path)
    ap.add_argument("--synthetic", action="store_true", help="use (and create if needed) data/synthetic")
    ap.add_argument("--categories", nargs="*")
    ap.add_argument("--n-train", type=int, help="few-shot: random subset of N normal images (default: all)")
    ap.add_argument("--name", default="experiment")
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
    runner = ExperimentRunner(cfg, n_train=a.n_train, seed=cfg.seed, experiment_name=a.name, output_root=a.out)
    res = runner.run(a.dataset, a.categories, progress=lambda msg, f: print(f"  [{100 * f:3.0f}%] {msg}"))
    make_experiment_plots(res.results, res.size_results, res.out_dir)
    write_summary_markdown(res.results, res.size_results, res.out_dir, f"Experiment {a.name}")
    print("\nMean over categories:")
    print(df_to_markdown(summarize(res.results)))
    print(f"\nResults saved to {res.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
