"""Common interface for feature extraction backends."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

SCALE_NAMES = ("fine", "coarse")


class BackendUnavailableError(RuntimeError):
    """Raised when an optional backend cannot be created (missing package or weights)."""


class FeatureExtractor(ABC):
    """Maps an RGB uint8 image to one dense feature map per spatial scale.

    ``extract`` returns ``{"fine": (h1, w1, C1), "coarse": (h2, w2, C2)}``.
    Feature maps keep their spatial layout so that patch-level residuals can
    be mapped back to image coordinates.
    """

    #: short identifier stored with results
    name: str = "base"

    @abstractmethod
    def extract(self, rgb: np.ndarray) -> dict[str, np.ndarray]:
        ...

    @abstractmethod
    def describe(self) -> dict:
        """Human-readable description of what each scale means (used in docs/UI)."""

    @property
    def label(self) -> str:
        """Backend label shown to users (must never misrepresent what is used)."""
        return self.name
