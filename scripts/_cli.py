"""Shared command-line options for the VisionGuard scripts."""
import _bootstrap  # noqa: F401
import argparse

from src.config import PipelineConfig
from src.utils.paths import DEFAULT_CONFIG_PATH


def add_config_args(ap: argparse.ArgumentParser) -> None:
    g = ap.add_argument_group("pipeline configuration (override config/default.yaml)")
    g.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="YAML configuration file")
    g.add_argument("--backend", choices=["handcrafted", "dinov2"], help="feature backend")
    g.add_argument("--allow-fallback", action="store_true",
                   help="if DINOv2 is unavailable use the clearly-labelled handcrafted fallback")
    g.add_argument("--image-size", type=int)
    g.add_argument("--n-components", type=float, help="PCA: <1 = explained variance ratio, >=1 = component count")
    g.add_argument("--alpha", type=float, help="fine-scale weight in S = alpha*fine + (1-alpha)*coarse")
    g.add_argument("--k", type=float, help="mean_std threshold: T = mean + k*std")
    g.add_argument("--percentile", type=float, help="percentile threshold (0-100]")
    g.add_argument("--fixed-value", type=float, help="fixed threshold (z-score units)")
    g.add_argument("--seed", type=int)


def build_config(a: argparse.Namespace) -> PipelineConfig:
    cfg = PipelineConfig.from_yaml(a.config)
    if a.backend:
        cfg.features.backend = a.backend
    if getattr(a, "allow_fallback", False):
        cfg.features.allow_fallback = True
    if a.image_size:
        cfg.features.image_size = a.image_size
    if a.n_components is not None:
        cfg.detector.n_components = int(a.n_components) if a.n_components >= 1 else a.n_components
    if a.alpha is not None:
        cfg.detector.alpha = a.alpha
    if a.k is not None:
        cfg.threshold.k = a.k
    if a.percentile is not None:
        cfg.threshold.percentile = a.percentile
    if a.fixed_value is not None:
        cfg.threshold.fixed_value = a.fixed_value
    if a.seed is not None:
        cfg.seed = a.seed
    cfg.validate()
    return cfg
