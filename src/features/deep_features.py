"""Optional pretrained DINOv2 patch-feature backend (requires requirements-deep.txt).

Uses the official DINOv2 weights published on the Hugging Face hub
(``facebook/dinov2-small`` by default) through ``transformers``.  Weights are
downloaded on first use and cached by Hugging Face; nothing is bundled here.

Spatial tokens (not the global CLS embedding) are used so that patch-level
residuals can localise defects.  With a 448 px input and patch size 14 the
token grid is 32 x 32.

Fine scale   : mean of the hidden states of ``dino_fine_layers`` (intermediate
               layers, more local/textural) at the native 32 x 32 token grid.
Coarse scale : final-layer tokens averaged over a ``dino_coarse_pool`` x
               ``dino_coarse_pool`` token neighbourhood (larger receptive field,
               more semantic / context-aware).

These layer choices are OUR documented design decisions; they are not claimed
to be the layers used by the SubspaceAD paper.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import uniform_filter

from src.config import FeatureConfig
from src.features.base import BackendUnavailableError, FeatureExtractor
from src.utils.logging_config import get_logger

log = get_logger(__name__)

_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)
PATCH = 14


class DINOv2FeatureExtractor(FeatureExtractor):
    name = "dinov2"

    def __init__(self, cfg: FeatureConfig | None = None, device: str | None = None):
        self.cfg = cfg or FeatureConfig(backend="dinov2")
        try:
            import torch
            from transformers import AutoModel
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise BackendUnavailableError(
                "DINOv2 backend needs 'torch' and 'transformers'. Install with: "
                "pip install -r requirements-deep.txt"
            ) from exc
        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.model = AutoModel.from_pretrained(self.cfg.dino_model).to(self.device).eval()
        except Exception as exc:  # pragma: no cover - network/weights dependent
            raise BackendUnavailableError(
                f"Could not load pretrained weights '{self.cfg.dino_model}' ({exc}). "
                "Check your internet connection or pre-download the model."
            ) from exc
        n_layers = self.model.config.num_hidden_layers
        bad = [i for i in self.cfg.dino_fine_layers if not 0 <= i <= n_layers]
        if bad:
            raise ValueError(f"dino_fine_layers {bad} out of range for a {n_layers}-layer model")

    @property
    def label(self) -> str:
        return f"DINOv2 ({self.cfg.dino_model}, pretrained) on {self.device}"

    def extract(self, rgb: np.ndarray) -> dict[str, np.ndarray]:
        torch = self._torch
        s = self.cfg.dino_input_size
        img = cv2.resize(rgb, (s, s), interpolation=cv2.INTER_AREA if rgb.shape[0] > s else cv2.INTER_CUBIC)
        x = ((img.astype(np.float32) / 255.0 - _MEAN) / _STD).transpose(2, 0, 1)[None]
        with torch.no_grad():
            out = self.model(pixel_values=torch.from_numpy(x).to(self.device), output_hidden_states=True)
        g = s // PATCH
        hs = out.hidden_states

        def grid(t):  # drop CLS token -> (g, g, C)
            return t[0, 1:1 + g * g].reshape(g, g, -1).float().cpu().numpy()

        fine = np.mean([grid(hs[i]) for i in self.cfg.dino_fine_layers], axis=0)
        last = grid(out.last_hidden_state)
        p = self.cfg.dino_coarse_pool
        coarse = uniform_filter(last, size=(p, p, 1), mode="nearest")
        return {"fine": fine.astype(np.float32), "coarse": coarse.astype(np.float32)}

    def describe(self) -> dict:
        g = self.cfg.dino_input_size // PATCH
        return {
            "backend": self.name,
            "model": self.cfg.dino_model,
            "token_grid": f"{g}x{g}",
            "scales": {
                "fine": f"mean of hidden_states{list(self.cfg.dino_fine_layers)} at {g}x{g} tokens",
                "coarse": f"final-layer tokens averaged over {self.cfg.dino_coarse_pool}x{self.cfg.dino_coarse_pool} neighbourhood",
            },
        }
