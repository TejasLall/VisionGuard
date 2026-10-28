"""Dataset setup helper: validate a dataset, extract a manually downloaded archive, or create synthetic data.

MVTec AD is NOT downloaded automatically: the official download is behind a form and its licence
(CC BY-NC-SA 4.0, non-commercial use) must be accepted by you.  Run `help-download` for the steps.
"""
import _bootstrap  # noqa: F401
import argparse
import shutil
import tarfile
import zipfile
from pathlib import Path

from src.data.dataset_validator import MVTEC_HELP, validate_dataset
from src.data.synthetic import create_synthetic_dataset
from src.utils.logging_config import setup_logging
from src.utils.paths import DATA_DIR


def _check_members(names, dest: Path) -> None:
    dest = dest.resolve()
    for n in names:
        if not (dest / n).resolve().is_relative_to(dest):
            raise ValueError(f"unsafe path in archive: {n}")


def extract(archive: Path, dest: Path) -> None:
    size = archive.stat().st_size
    probe = dest if dest.exists() else dest.parent if dest.parent.exists() else Path.cwd()
    free = shutil.disk_usage(probe).free
    if free < 2.2 * size:
        raise SystemExit(f"Not enough disk space: archive {size / 1e9:.1f} GB, free {free / 1e9:.1f} GB")
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {archive} -> {dest} (this can take several minutes)")
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as z:
            _check_members(z.namelist(), dest)
            z.extractall(dest)
    else:
        with tarfile.open(archive) as t:
            _check_members(t.getnames(), dest)     # reject path traversal before extracting
            t.extractall(dest)


def main() -> int:
    setup_logging()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate", help="check a dataset folder (MVTec-style or custom)")
    v.add_argument("path", type=Path)
    s = sub.add_parser("synthetic", help="generate the SYNTHETIC smoke-test dataset (software testing only)")
    s.add_argument("--out", type=Path, default=DATA_DIR / "synthetic")
    s.add_argument("--seed", type=int, default=0)
    e = sub.add_parser("extract", help="extract a manually downloaded MVTec AD archive (.tar.xz/.tar/.zip)")
    e.add_argument("archive", type=Path)
    e.add_argument("--out", type=Path, default=DATA_DIR / "mvtec_ad")
    sub.add_parser("help-download", help="print manual download instructions")
    a = ap.parse_args()

    if a.cmd == "help-download":
        print(MVTEC_HELP)
        print("\nSteps:\n"
              " 1. Open the dataset page and fill in the download form.\n"
              " 2. Save mvtec_anomaly_detection.tar.xz (about 4.9 GB) anywhere.\n"
              " 3. python scripts/prepare_dataset.py extract <archive>      (or extract with 7-Zip)\n"
              " 4. python scripts/prepare_dataset.py validate data/mvtec_ad")
        return 0
    if a.cmd == "synthetic":
        create_synthetic_dataset(a.out, seed=a.seed)
        print(f"Synthetic dataset written to {a.out} (for software testing only).")
        a.path = a.out
    elif a.cmd == "extract":
        extract(a.archive, a.out)
        a.path = a.out
    rep = validate_dataset(a.path)
    print(rep.summary())
    return 0 if rep.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
