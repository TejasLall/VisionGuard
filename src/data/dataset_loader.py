"""Dataset discovery for MVTec-AD-style and custom folder layouts.

Category and defect-type names are discovered from the filesystem, never
hard-coded.  Ground-truth masks are exposed only through ``Sample.mask_path``
and are used for *evaluation only* (never for fitting or calibration).

MVTec layout::

    root/<category>/train/good/*.png
    root/<category>/test/<good|defect_type>/*.png
    root/<category>/ground_truth/<defect_type>/<stem>_mask.png

Custom layout::

    root/normal/*.png                  (normal training / calibration images)
    root/test/normal/*.png
    root/test/defective/*.png
    root/ground_truth/<stem>.png       (optional; or <stem>_mask.png)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.utils.logging_config import get_logger
from src.utils.paths import list_images

log = get_logger(__name__)

NORMAL_NAMES = {"good", "normal"}


@dataclass(frozen=True)
class Sample:
    image_path: Path
    label: int                      # 0 = normal, 1 = defective
    defect_type: str                # "good" for normal images
    category: str
    mask_path: Path | None = None   # evaluation only


@dataclass
class CategoryData:
    name: str
    layout: str                     # "mvtec" | "custom"
    train_paths: list[Path]
    test_samples: list[Sample] = field(default_factory=list)

    @property
    def has_masks(self) -> bool:
        defect = [s for s in self.test_samples if s.label == 1]
        return bool(defect) and all(s.mask_path is not None for s in defect)

    @property
    def n_test_normal(self) -> int:
        return sum(1 for s in self.test_samples if s.label == 0)

    @property
    def n_test_defective(self) -> int:
        return sum(1 for s in self.test_samples if s.label == 1)

    def defect_types(self) -> list[str]:
        return sorted({s.defect_type for s in self.test_samples if s.label == 1})


def is_mvtec_category(path: Path) -> bool:
    return (path / "train").is_dir() and (path / "test").is_dir()


def is_custom_dataset(path: Path) -> bool:
    return (path / "normal").is_dir() and (path / "test").is_dir() and not (path / "train").is_dir()


def discover_categories(root: str | Path) -> list[str]:
    """Names of MVTec-style categories directly under ``root`` (sorted)."""
    r = Path(root)
    if not r.is_dir():
        return []
    return sorted(d.name for d in r.iterdir() if d.is_dir() and is_mvtec_category(d))


def _find_mask(gt_dir: Path, defect_type: str, stem: str) -> Path | None:
    for name in (f"{stem}_mask", stem):
        for ext in (".png", ".bmp", ".jpg", ".jpeg"):
            cand = gt_dir / defect_type / f"{name}{ext}"
            if cand.is_file():
                return cand
    return None


def load_mvtec_category(root: str | Path, category: str) -> CategoryData:
    cat_dir = Path(root) / category
    if not is_mvtec_category(cat_dir):
        raise FileNotFoundError(f"{cat_dir} is not an MVTec-style category (needs train/ and test/)")
    train_dir = cat_dir / "train"
    normal_dirs = [train_dir / n for n in NORMAL_NAMES if (train_dir / n).is_dir()]
    if not normal_dirs:
        raise FileNotFoundError(f"{train_dir} has no 'good' sub-folder of normal images")
    train_paths = sorted(p for d in normal_dirs for p in list_images(d))

    samples: list[Sample] = []
    test_dir, gt_dir = cat_dir / "test", cat_dir / "ground_truth"
    for sub in sorted(d for d in test_dir.iterdir() if d.is_dir()):
        is_normal = sub.name.lower() in NORMAL_NAMES
        for img in list_images(sub):
            mask = None if is_normal else _find_mask(gt_dir, sub.name, img.stem)
            samples.append(Sample(img, 0 if is_normal else 1, "good" if is_normal else sub.name, category, mask))
    return CategoryData(category, "mvtec", train_paths, samples)


def load_custom_dataset(root: str | Path, name: str | None = None) -> CategoryData:
    r = Path(root)
    if not is_custom_dataset(r):
        raise FileNotFoundError(f"{r} is not a custom dataset (needs normal/ and test/)")
    cname = name or r.name
    train_paths = list_images(r / "normal")
    samples: list[Sample] = []
    gt_dir = r / "ground_truth"
    for sub_name, label in (("normal", 0), ("defective", 1)):
        for img in list_images(r / "test" / sub_name):
            mask = None
            if label == 1:
                for cand in (gt_dir / f"{img.stem}.png", gt_dir / f"{img.stem}_mask.png"):
                    if cand.is_file():
                        mask = cand
                        break
            samples.append(Sample(img, label, "good" if label == 0 else "defective", cname, mask))
    return CategoryData(cname, "custom", train_paths, samples)


def load_category(root: str | Path, category: str | None = None) -> CategoryData:
    """Load either layout.  For MVTec roots ``category`` is required."""
    r = Path(root)
    if is_custom_dataset(r):
        return load_custom_dataset(r)
    if is_mvtec_category(r):                     # root points directly at one category
        return load_mvtec_category(r.parent, r.name)
    if category is None:
        raise ValueError("category is required when root contains several categories")
    return load_mvtec_category(r, category)
