"""Shared helpers for the Streamlit pages (caching, model discovery, widgets, image export)."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from src.config import ThresholdConfig
from src.data.dataset_loader import CategoryData, load_category
from src.data.dataset_validator import ValidationReport, validate_dataset
from src.data.image_io import load_image_from_bytes, load_image_rgb, load_mask
from src.models.model_manager import ModelManager
from src.models.pipeline import AnomalyPipeline
from src.utils.paths import DATA_DIR, MODELS_DIR, OUTPUTS_DIR

SCORE_NOTE = ("Scores are z-score-normalised anomaly scores relative to normal calibration images. "
              "They are **not** calibrated probabilities.")


# ------------------------------------------------------------------ session
def init_state() -> None:
    if "dataset_root" not in st.session_state:
        default = DATA_DIR / "mvtec_ad"
        if not default.exists():
            default = DATA_DIR / "synthetic"
        st.session_state.dataset_root = str(default)


def deep_backend_installed() -> bool:
    return all(importlib.util.find_spec(m) is not None for m in ("torch", "transformers"))


# ------------------------------------------------------------------ datasets
@st.cache_data(ttl="10m", max_entries=8, show_spinner="Validating dataset…")
def cached_validation(root: str) -> ValidationReport:
    return validate_dataset(root)


@st.cache_data(ttl="10m", max_entries=16, show_spinner=False)
def cached_category(root: str, category: str | None) -> CategoryData:
    return load_category(root, category)


# ------------------------------------------------------------------ models
def manager() -> ModelManager:
    return ModelManager(MODELS_DIR)


def _mtime(name: str) -> float:
    p = manager().path_for(name) / "state.npz"
    return p.stat().st_mtime if p.exists() else 0.0


@st.cache_resource(max_entries=12, show_spinner="Loading model…")
def _load(name: str, mtime: float) -> AnomalyPipeline:
    return manager().load(name)


def load_model(name: str) -> AnomalyPipeline:
    """Cached load; re-loads automatically when the model file changes."""
    return _load(name, _mtime(name))


def model_table() -> pd.DataFrame:
    """One row per saved model with metadata from meta.json."""
    rows = []
    for name in manager().list_models():
        try:
            meta = manager().read_meta(name)
        except (OSError, json.JSONDecodeError):
            continue
        cfg, info = meta.get("config", {}), meta.get("fit_info", {})
        rows.append({
            "model": name,
            "category": meta.get("category", name.split("__")[0]),
            "backend": meta.get("backend_label", cfg.get("features", {}).get("backend")),
            "preprocessing": "on" if cfg.get("preprocess", {}).get("enabled") else "off",
            "n_train": info.get("n_train_images"),
            "pca_components": str(info.get("pca_components")),
            "calibration": meta.get("calibration_mode"),
            "image_size": cfg.get("features", {}).get("image_size"),
            "saved_at": meta.get("saved_at"),
        })
    return pd.DataFrame(rows)


def model_pairs() -> dict[str, dict[str, str]]:
    """``{"<category> (<backend>)": {"raw": name, "illum": name}}`` from canonical model names."""
    out: dict[str, dict[str, str]] = {}
    for name in manager().list_models():
        parts = name.split("__")
        if len(parts) == 3 and parts[2] in ("raw", "illum"):
            out.setdefault(f"{parts[0]} ({parts[1]})", {})[parts[2]] = name
        else:
            out.setdefault(name, {})["raw"] = name
    return out


# ------------------------------------------------------------------ outputs
def experiment_dirs() -> list[Path]:
    if not OUTPUTS_DIR.is_dir():
        return []
    dirs = [d for d in OUTPUTS_DIR.iterdir() if (d / "results.csv").is_file()]
    return sorted(dirs, key=lambda d: (d / "results.csv").stat().st_mtime, reverse=True)


@st.cache_data(ttl="5m", max_entries=16, show_spinner=False)
def read_experiment(path: str, mtime: float) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    p = Path(path)
    res = pd.read_csv(p / "results.csv")
    size = pd.read_csv(p / "size_analysis.csv") if (p / "size_analysis.csv").is_file() else pd.DataFrame()
    meta = json.loads((p / "config.json").read_text(encoding="utf-8")) if (p / "config.json").is_file() else {}
    return res, size, meta


def load_experiment(path: Path):
    return read_experiment(str(path), (path / "results.csv").stat().st_mtime)


# ------------------------------------------------------------------ widgets
def threshold_controls(key: str, default_method: str = "mean_std") -> ThresholdConfig:
    """Threshold method + its parameter, with explanations."""
    labels = {"fixed": "Fixed", "mean_std": "Mean + k·std", "percentile": "Percentile"}
    method = st.segmented_control(
        "Threshold method", list(labels), format_func=labels.get, default=default_method, required=True,
        key=f"{key}_thr", help="Fixed is the baseline. The adaptive methods are estimated from held-out NORMAL "
                               "calibration images only (never from test images or masks).")
    cfg = ThresholdConfig(method=method)
    if method == "fixed":
        cfg.fixed_value = st.number_input("Fixed threshold T (z-units)", 0.0, 50.0, 3.0, 0.5, key=f"{key}_fix",
                                          help="Same constant for every category and lighting condition.")
    elif method == "mean_std":
        cfg.k = st.slider("k (T = mean + k·std of normal image scores)", 0.0, 6.0, 3.0, 0.25, key=f"{key}_k",
                          help="Larger k → fewer false alarms but more missed defects.")
    else:
        cfg.percentile = st.slider("Percentile P of normal image scores", 80.0, 100.0, 99.0, 0.5, key=f"{key}_p",
                                   help="With few calibration images a high percentile is close to their maximum.")
    cfg.min_area = st.number_input("Minimum defect area (px, 0 = off)", 0, 5000, 0, 10, key=f"{key}_area",
                                   help="Removes predicted blobs smaller than this before deciding.")
    return cfg


def image_source(key: str, root: str | None, categories: list[str] | None = None):
    """Upload or pick a dataset test image. Returns (rgb, name, gt_mask_or_None, label_or_None)."""
    src = st.segmented_control("Image source", ["Upload", "Dataset test image"], default="Upload", required=True,
                               key=f"{key}_src")
    if src == "Upload":
        up = st.file_uploader("Upload an image (JPG, JPEG, PNG, BMP)", type=["jpg", "jpeg", "png", "bmp"], key=f"{key}_up")
        if up is None:
            return None, None, None, None
        return load_image_from_bytes(up.getvalue(), up.name), up.name, None, None
    if not root:
        st.info("Select a dataset on the *Dataset & training* page first.")
        return None, None, None, None
    rep = cached_validation(root)
    cats = categories or list(rep.categories)
    if not cats:
        st.warning("The current dataset has no usable categories.")
        return None, None, None, None
    cat = st.selectbox("Category", cats, key=f"{key}_cat")
    data = cached_category(root, None if rep.layout == "custom" else cat)
    if not data.test_samples:
        st.warning("This category has no test images.")
        return None, None, None, None
    names = [f"{s.defect_type}/{s.image_path.name}" for s in data.test_samples]
    i = st.selectbox("Test image", range(len(names)), format_func=names.__getitem__, key=f"{key}_img")
    s = data.test_samples[i]
    return load_image_rgb(s.image_path), s.image_path.name, s.mask_path, s.label


def png_bytes(arr: np.ndarray) -> bytes:
    a = arr if arr.ndim == 2 else cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    ok, enc = cv2.imencode(".png", a)
    return enc.tobytes() if ok else b""


def gt_mask_for(mask_path, size: int):
    return None if mask_path is None else load_mask(mask_path, size)


def fmt(v, digits: int = 3) -> str:
    try:
        return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{digits}f}"
    except (TypeError, ValueError):
        return str(v)
