"""Filesystem helpers (all path handling goes through pathlib)."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CONFIG_DIR = PROJECT_ROOT / "config"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "default.yaml"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def ensure_dir(path: str | Path) -> Path:
    """Create ``path`` (and parents) if needed and return it as a Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def timestamp() -> str:
    """Filesystem-safe timestamp, e.g. ``20261029_141530``."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def iso_now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def list_images(folder: str | Path) -> list[Path]:
    """Sorted list of supported image files directly inside ``folder``."""
    p = Path(folder)
    if not p.is_dir():
        return []
    return sorted(f for f in p.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS)
