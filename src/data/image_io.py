"""Image and mask loading with validation.

OpenCV decodes images through ``np.fromfile`` + ``cv2.imdecode`` so that
Windows paths containing non-ASCII characters or spaces work reliably.
All images leave this module as ``uint8`` RGB arrays of shape (H, W, 3).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.utils.paths import IMAGE_EXTENSIONS

MIN_SIDE = 16
MAX_SIDE = 20_000


class ImageLoadError(ValueError):
    """Raised when a file is missing, has an unsupported format or is corrupt."""


def validate_image_array(arr: np.ndarray, name: str = "image") -> np.ndarray:
    """Check dtype/shape/size of an in-memory image and return it as RGB uint8."""
    if not isinstance(arr, np.ndarray):
        raise ImageLoadError(f"{name}: expected a numpy array, got {type(arr).__name__}")
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[..., :3]
    elif arr.ndim != 3 or arr.shape[2] != 3:
        raise ImageLoadError(f"{name}: unsupported shape {arr.shape}; need (H, W) or (H, W, 1/3/4)")
    h, w = arr.shape[:2]
    if min(h, w) < MIN_SIDE or max(h, w) > MAX_SIDE:
        raise ImageLoadError(f"{name}: image side must be within [{MIN_SIDE}, {MAX_SIDE}] px, got {w}x{h}")
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating) and arr.max() <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _decode(buf: np.ndarray, name: str) -> np.ndarray:
    img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ImageLoadError(f"{name}: file is corrupt or not a decodable image")
    if img.dtype == np.uint16:
        img = (img / 257.0).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return validate_image_array(img, name)


def load_image_rgb(path: str | Path) -> np.ndarray:
    """Load an image file as RGB uint8, validating extension, content and size."""
    p = Path(path)
    if not p.is_file():
        raise ImageLoadError(f"{p}: file does not exist")
    if p.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ImageLoadError(f"{p.name}: unsupported format '{p.suffix}'; use {sorted(IMAGE_EXTENSIONS)}")
    try:
        buf = np.fromfile(str(p), dtype=np.uint8)
    except OSError as exc:
        raise ImageLoadError(f"{p}: cannot read file ({exc})") from exc
    if buf.size == 0:
        raise ImageLoadError(f"{p.name}: file is empty")
    return _decode(buf, p.name)


def load_image_from_bytes(data: bytes, name: str = "upload") -> np.ndarray:
    """Decode an uploaded image (e.g. from Streamlit) with the same validation."""
    suffix = Path(name).suffix.lower()
    if suffix and suffix not in IMAGE_EXTENSIONS:
        raise ImageLoadError(f"{name}: unsupported format '{suffix}'; use {sorted(IMAGE_EXTENSIONS)}")
    if not data:
        raise ImageLoadError(f"{name}: file is empty")
    return _decode(np.frombuffer(data, dtype=np.uint8), name)


def resize_square(rgb: np.ndarray, size: int) -> np.ndarray:
    """Resize to ``size`` x ``size`` (aspect ratio is NOT preserved; documented)."""
    h, w = rgb.shape[:2]
    if (h, w) == (size, size):
        return rgb
    interp = cv2.INTER_AREA if max(h, w) > size else cv2.INTER_CUBIC
    return cv2.resize(rgb, (size, size), interpolation=interp)


def load_mask(path: str | Path, size: int) -> np.ndarray:
    """Load a ground-truth mask as a boolean (size, size) array (nearest resize)."""
    p = Path(path)
    if not p.is_file():
        raise ImageLoadError(f"{p}: mask does not exist")
    buf = np.fromfile(str(p), dtype=np.uint8)
    m = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise ImageLoadError(f"{p.name}: mask is corrupt")
    if m.shape != (size, size):
        m = cv2.resize(m, (size, size), interpolation=cv2.INTER_NEAREST)
    return m > 127


def save_image_rgb(path: str | Path, rgb: np.ndarray) -> Path:
    """Save an RGB (or single channel) uint8 array; creates parent folders."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    arr = rgb if rgb.ndim == 2 else cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, enc = cv2.imencode(p.suffix or ".png", arr)
    if not ok:
        raise ImageLoadError(f"cannot encode image for {p}")
    enc.tofile(str(p))
    return p
