"""Saving and loading fitted pipelines.

A saved model is a folder ``models/<name>/`` containing

* ``state.npz``  - PCA bases, normalisation statistics, calibration maps (NumPy arrays only,
                   no pickle, so loading cannot execute code);
* ``meta.json``  - full configuration, fit information and provenance.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

from src.models.pipeline import AnomalyPipeline
from src.utils.paths import MODELS_DIR, ensure_dir, iso_now

_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def sanitize_name(name: str) -> str:
    out = _SAFE.sub("_", name.strip()).strip("._")
    if not out:
        raise ValueError("model name is empty after sanitising")
    return out


class ModelManager:
    def __init__(self, root: str | Path | None = None):
        self.root = Path(root) if root else MODELS_DIR

    def path_for(self, name: str) -> Path:
        return self.root / sanitize_name(name)

    def save(self, pipeline: AnomalyPipeline, name: str, extra_meta: dict | None = None) -> Path:
        arrays, meta = pipeline.get_state()
        meta = dict(meta, saved_at=iso_now(), name=name, python=sys.version.split()[0],
                    numpy=np.__version__, **(extra_meta or {}))
        folder = ensure_dir(self.path_for(name))
        np.savez_compressed(folder / "state.npz", **{k.replace("/", "__"): v for k, v in arrays.items()})
        (folder / "meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
        return folder

    def load(self, name: str) -> AnomalyPipeline:
        folder = self.path_for(name)
        if not (folder / "state.npz").is_file() or not (folder / "meta.json").is_file():
            raise FileNotFoundError(f"No saved model named '{name}' in {self.root}")
        meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
        with np.load(folder / "state.npz", allow_pickle=False) as npz:
            arrays = {k.replace("__", "/"): npz[k] for k in npz.files}
        return AnomalyPipeline.from_state(arrays, meta)

    def read_meta(self, name: str) -> dict:
        return json.loads((self.path_for(name) / "meta.json").read_text(encoding="utf-8"))

    def list_models(self) -> list[str]:
        if not self.root.is_dir():
            return []
        return sorted(d.name for d in self.root.iterdir() if (d / "state.npz").is_file())


def model_name(category: str, backend: str, preprocess: bool) -> str:
    """Canonical model folder name: ``<category>__<backend>__<raw|illum>``."""
    return sanitize_name(f"{category}__{backend}__{'illum' if preprocess else 'raw'}")
