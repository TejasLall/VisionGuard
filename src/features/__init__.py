"""Feature extraction backends and a factory that never misrepresents the backend in use."""
from __future__ import annotations

from src.config import FeatureConfig
from src.features.base import SCALE_NAMES, BackendUnavailableError, FeatureExtractor
from src.features.handcrafted import HandcraftedFeatureExtractor
from src.utils.logging_config import get_logger

log = get_logger(__name__)

__all__ = [
    "SCALE_NAMES",
    "BackendUnavailableError",
    "FeatureExtractor",
    "HandcraftedFeatureExtractor",
    "create_feature_extractor",
]


class _FallbackExtractor(HandcraftedFeatureExtractor):
    """Handcrafted extractor used because DINOv2 was requested but is unavailable."""

    @property
    def label(self) -> str:
        return "Handcrafted FALLBACK (DINOv2 was requested but is unavailable - NOT DINOv2)"


def create_feature_extractor(cfg: FeatureConfig) -> FeatureExtractor:
    """Create the extractor for ``cfg.backend``.

    If DINOv2 is requested but cannot be created, either raise
    :class:`BackendUnavailableError` or (``allow_fallback=True``) return a
    clearly labelled handcrafted fallback.
    """
    if cfg.backend == "handcrafted":
        return HandcraftedFeatureExtractor(cfg)
    if cfg.backend == "dinov2":
        try:
            from src.features.deep_features import DINOv2FeatureExtractor

            return DINOv2FeatureExtractor(cfg)
        except BackendUnavailableError as exc:
            if not cfg.allow_fallback:
                raise
            log.warning("DINOv2 unavailable (%s). Using clearly-labelled handcrafted fallback.", exc)
            return _FallbackExtractor(cfg)
    raise ValueError(f"Unknown backend '{cfg.backend}'")
