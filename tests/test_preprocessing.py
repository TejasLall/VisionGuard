import numpy as np
import pytest

from src.config import PreprocessConfig
from src.experiments.robustness import brightness, default_transforms, gamma, gaussian_noise, gradient_light, shadow
from src.preprocessing.illumination import ImagePreprocessor


@pytest.fixture()
def img():
    rng = np.random.default_rng(0)
    base = rng.integers(80, 180, (96, 96, 1), dtype=np.uint8)
    return np.repeat(base, 3, axis=2) + rng.integers(0, 20, (96, 96, 3), dtype=np.uint8)


def test_disabled_is_identity(img):
    out = ImagePreprocessor(PreprocessConfig(enabled=False))(img)
    assert out is img


def test_enabled_keeps_shape_dtype_and_changes_pixels(img):
    p = ImagePreprocessor(PreprocessConfig(enabled=True)).fit([img])
    out = p(img)
    assert out.shape == img.shape and out.dtype == np.uint8
    assert not np.array_equal(out, img)


def test_clahe_modes(img):
    always = ImagePreprocessor(PreprocessConfig(enabled=True, clahe_mode="always", flatfield=False, lum_norm=False))
    off = ImagePreprocessor(PreprocessConfig(enabled=True, clahe_mode="off", flatfield=False, lum_norm=False))
    assert not np.array_equal(always(img), img)
    # with every stage off the Lab round trip is (almost) lossless
    assert np.abs(off(img).astype(int) - img.astype(int)).max() <= 3


def test_auto_mode_only_corrects_badly_lit_images(img):
    p = ImagePreprocessor(PreprocessConfig(enabled=True, clahe_mode="auto"))
    dark = (img * 0.2).astype(np.uint8)
    assert p.clahe_recommended(dark)
    well_lit = np.clip(img.astype(int) * 0 + 128 + (img.astype(int) - img.mean()) * 1.5, 0, 255).astype(np.uint8)
    assert not p.clahe_recommended(well_lit)


def test_luminance_normalisation_reduces_brightness_shift(img):
    cfg = PreprocessConfig(enabled=True, clahe_mode="off", flatfield=False, lum_norm=True)
    p = ImagePreprocessor(cfg).fit([img])
    dark = brightness(0.5)(img)
    d_before = abs(float(dark.mean()) - float(img.mean()))
    out = p(dark)
    # luminance stats are pulled back to the training statistics
    ref = p(img)
    assert abs(float(out.mean()) - float(ref.mean())) < d_before * 0.4


def test_flatfield_flattens_gradient(img):
    cfg = PreprocessConfig(enabled=True, clahe_mode="off", flatfield=True, lum_norm=False)
    p = ImagePreprocessor(cfg)
    graded = gradient_light(0.6)(np.full((96, 96, 3), 150, np.uint8), seed=3)
    before = ImagePreprocessor.diagnostics(graded)["lowfreq_std"]
    after = ImagePreprocessor.diagnostics(p(graded))["lowfreq_std"]
    assert after < before * 0.5


def test_gray_world_equalises_channel_means(img):
    tinted = img.copy()
    tinted[..., 0] = np.clip(tinted[..., 0].astype(int) * 0.5, 0, 255)
    cfg = PreprocessConfig(enabled=True, clahe_mode="off", flatfield=False, lum_norm=False, gray_world=True)
    out = ImagePreprocessor(cfg)(tinted)
    m = out.reshape(-1, 3).mean(0)
    assert m.max() - m.min() < 12


def test_state_roundtrip(img):
    p = ImagePreprocessor(PreprocessConfig(enabled=True)).fit([img])
    q = ImagePreprocessor(PreprocessConfig(enabled=True))
    q.set_state(p.get_state())
    assert np.array_equal(p(img), q(img))


def test_robustness_transforms_are_valid_and_deterministic(img):
    for name, t in default_transforms().items():
        a, b = t(img, 5), t(img, 5)
        assert a.shape == img.shape and a.dtype == np.uint8, name
        assert np.array_equal(a, b), name
    assert brightness(0.5)(img).mean() < img.mean() < brightness(1.5)(img).mean()
    assert gamma(2.0)(img).mean() < img.mean() < gamma(0.5)(img).mean()
    assert shadow()(img, 1).mean() < img.mean()
    assert not np.array_equal(gaussian_noise(10)(img, 1), gaussian_noise(10)(img, 2))
