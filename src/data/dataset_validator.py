"""Validation of dataset directories before any training or evaluation."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.data.dataset_loader import (
    discover_categories,
    is_custom_dataset,
    is_mvtec_category,
    load_category,
)

MVTEC_HELP = (
    "Download MVTec AD manually from https://www.mvtec.com/company/research/datasets/mvtec-ad "
    "(a registration form is required; licence CC BY-NC-SA 4.0, non-commercial use only), "
    "extract it, and point VisionGuard at the folder that contains the category folders "
    "(bottle/, cable/, ...)."
)


@dataclass
class ValidationReport:
    root: Path
    layout: str = "unknown"            # mvtec | custom | unknown
    ok: bool = False
    categories: dict[str, dict] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"Dataset root: {self.root}", f"Layout: {self.layout}", f"Valid: {self.ok}"]
        for name, info in self.categories.items():
            lines.append(
                f"  - {name}: train={info['n_train']}, test normal={info['n_test_normal']}, "
                f"test defective={info['n_test_defective']}, masks={'yes' if info['has_masks'] else 'no'}"
            )
        lines += [f"ERROR: {e}" for e in self.errors] + [f"WARNING: {w}" for w in self.warnings]
        return "\n".join(lines)


def validate_dataset(root: str | Path) -> ValidationReport:
    """Detect the layout under ``root`` and verify it contains usable data."""
    r = Path(root)
    rep = ValidationReport(root=r)
    if not r.exists():
        rep.errors.append(f"Path does not exist: {r}")
        return rep
    if not r.is_dir():
        rep.errors.append(f"Path is not a directory: {r}")
        return rep

    if is_custom_dataset(r):
        rep.layout, names = "custom", [None]
    elif is_mvtec_category(r):
        rep.layout, names = "mvtec", [r.name]
    else:
        names = discover_categories(r)
        rep.layout = "mvtec" if names else "unknown"
        if not names:
            rep.errors.append("No MVTec-style categories (<cat>/train + <cat>/test) and no custom layout "
                              "(normal/ + test/) found. " + MVTEC_HELP)
            return rep

    for n in names:
        try:
            data = load_category(r, n)
        except (FileNotFoundError, ValueError) as exc:
            rep.errors.append(str(exc))
            continue
        info = {
            "n_train": len(data.train_paths),
            "n_test_normal": data.n_test_normal,
            "n_test_defective": data.n_test_defective,
            "has_masks": data.has_masks,
            "defect_types": data.defect_types(),
        }
        rep.categories[data.name] = info
        if info["n_train"] < 2:
            rep.errors.append(f"{data.name}: need at least 2 normal training images, found {info['n_train']}")
        if info["n_test_defective"] == 0:
            rep.warnings.append(f"{data.name}: no defective test images; only fitting/inspection possible")
        elif info["n_test_normal"] == 0:
            rep.warnings.append(f"{data.name}: no normal test images; image-level AUROC is undefined")
        if info["n_test_defective"] and not info["has_masks"]:
            rep.warnings.append(f"{data.name}: ground-truth masks missing; pixel-level metrics will not be reported")

    rep.ok = bool(rep.categories) and not rep.errors
    return rep
