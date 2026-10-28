"""Report Python/package versions, optional deep backend availability, disk space and datasets."""
import _bootstrap  # noqa: F401
import importlib
import platform
import shutil
import sys

from src.utils.paths import DATA_DIR, MODELS_DIR, OUTPUTS_DIR, PROJECT_ROOT

CORE = ["numpy", "scipy", "cv2", "sklearn", "skimage", "PIL", "matplotlib", "pandas", "yaml", "streamlit", "pytest"]
OPTIONAL = ["torch", "transformers"]


def version(mod: str):
    try:
        return getattr(importlib.import_module(mod), "__version__", "installed")
    except Exception:
        return None


def main() -> int:
    print(f"Python      : {sys.version.split()[0]} ({platform.platform()})")
    print(f"Project root: {PROJECT_ROOT}")
    missing = []
    print("\nCore packages (required):")
    for m in CORE:
        v = version(m)
        print(f"  {'OK     ' if v else 'MISSING'} {m:14s} {v or ''}")
        if not v:
            missing.append(m)
    print("\nOptional deep backend (DINOv2):")
    for m in OPTIONAL:
        v = version(m)
        print(f"  {'OK           ' if v else 'not installed'} {m:14s} {v or ''}")
    if version("torch"):
        import torch

        print(f"  CUDA available: {torch.cuda.is_available()}")
    free = shutil.disk_usage(PROJECT_ROOT).free / 1e9
    print(f"\nFree disk space: {free:.1f} GB (the MVTec AD archive is ~5 GB compressed and ~5 GB extracted)")
    for name, p in (("data", DATA_DIR), ("models", MODELS_DIR), ("outputs", OUTPUTS_DIR)):
        print(f"  {name:8s}: {p} {'(exists)' if p.exists() else '(missing)'}")
    if missing:
        print(f"\nMissing core packages: {missing}. Run: pip install -r requirements.txt")
        return 1
    print("\nEnvironment OK for the CPU handcrafted backend.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
