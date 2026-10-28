import cv2
import numpy as np
import pytest

from src.data.dataset_loader import discover_categories, load_category, load_custom_dataset, load_mvtec_category
from src.data.dataset_validator import validate_dataset
from src.data.image_io import (ImageLoadError, load_image_from_bytes, load_image_rgb, load_mask, resize_square,
                               save_image_rgb, validate_image_array)


def test_load_valid_image_roundtrip(tmp_path):
    img = np.random.default_rng(0).integers(0, 255, (40, 50, 3), dtype=np.uint8)
    p = save_image_rgb(tmp_path / "a b.png", img)       # space in name on purpose
    out = load_image_rgb(p)
    assert out.shape == (40, 50, 3) and out.dtype == np.uint8
    assert np.array_equal(out, img)


def test_grayscale_and_rgba_are_converted(tmp_path):
    gray = np.full((32, 32), 100, np.uint8)
    cv2.imencode(".png", gray)[1].tofile(str(tmp_path / "g.png"))
    assert load_image_rgb(tmp_path / "g.png").shape == (32, 32, 3)
    rgba = np.zeros((32, 32, 4), np.uint8)
    cv2.imencode(".png", rgba)[1].tofile(str(tmp_path / "r.png"))
    assert load_image_rgb(tmp_path / "r.png").shape == (32, 32, 3)


@pytest.mark.parametrize("content,name", [(b"", "empty.png"), (b"not an image at all", "bad.png")])
def test_invalid_files_raise(tmp_path, content, name):
    p = tmp_path / name
    p.write_bytes(content)
    with pytest.raises(ImageLoadError):
        load_image_rgb(p)


def test_missing_and_unsupported_extension(tmp_path):
    with pytest.raises(ImageLoadError):
        load_image_rgb(tmp_path / "nope.png")
    (tmp_path / "x.gif").write_bytes(b"GIF89a")
    with pytest.raises(ImageLoadError, match="unsupported"):
        load_image_rgb(tmp_path / "x.gif")
    with pytest.raises(ImageLoadError):
        load_image_from_bytes(b"junk", "upload.png")


def test_validate_array_rejects_bad_shapes():
    with pytest.raises(ImageLoadError):
        validate_image_array(np.zeros((8, 8, 3), np.uint8))          # too small
    with pytest.raises(ImageLoadError):
        validate_image_array(np.zeros((32, 32, 2), np.uint8))
    with pytest.raises(ImageLoadError):
        validate_image_array([[1, 2, 3]])


def test_resize_and_mask(tmp_path):
    img = np.zeros((100, 60, 3), np.uint8)
    assert resize_square(img, 64).shape == (64, 64, 3)
    m = np.zeros((100, 100), np.uint8)
    m[:50] = 255
    cv2.imencode(".png", m)[1].tofile(str(tmp_path / "m.png"))
    mask = load_mask(tmp_path / "m.png", 50)
    assert mask.shape == (50, 50) and mask.dtype == bool and mask[:25].all() and not mask[25:].any()


def test_mvtec_layout_discovery_and_masks(synthetic_root):
    assert discover_categories(synthetic_root) == ["synthetic_texture"]
    d = load_mvtec_category(synthetic_root, "synthetic_texture")
    assert len(d.train_paths) == 10
    assert d.n_test_normal == 4 and d.n_test_defective == 9
    assert d.defect_types() == ["patch", "scratch", "spot"]          # discovered, not hard-coded
    assert d.has_masks
    assert all(s.mask_path is None for s in d.test_samples if s.label == 0)


def test_validator_reports_ok_and_errors(synthetic_root, tmp_path):
    rep = validate_dataset(synthetic_root)
    assert rep.ok and rep.layout == "mvtec" and "synthetic_texture" in rep.categories
    bad = validate_dataset(tmp_path / "does_not_exist")
    assert not bad.ok and bad.errors
    (tmp_path / "empty").mkdir()
    rep2 = validate_dataset(tmp_path / "empty")
    assert not rep2.ok and any("MVTec" in e for e in rep2.errors)


def test_custom_dataset_without_masks(tmp_path):
    rng = np.random.default_rng(0)
    for sub in ("normal", "test/normal", "test/defective"):
        for i in range(3):
            save_image_rgb(tmp_path / sub / f"{i}.png", rng.integers(0, 255, (32, 32, 3), dtype=np.uint8))
    d = load_custom_dataset(tmp_path)
    assert len(d.train_paths) == 3 and d.n_test_defective == 3 and not d.has_masks
    rep = validate_dataset(tmp_path)
    assert rep.ok and rep.layout == "custom"
    assert any("masks" in w for w in rep.warnings)          # pixel metrics must not be implied
    assert load_category(tmp_path).name == tmp_path.name


def test_custom_dataset_with_masks(tmp_path):
    for sub in ("normal", "test/normal"):
        save_image_rgb(tmp_path / sub / "0.png", np.zeros((32, 32, 3), np.uint8))
    save_image_rgb(tmp_path / "test/defective/0.png", np.zeros((32, 32, 3), np.uint8))
    save_image_rgb(tmp_path / "ground_truth/0.png", np.full((32, 32), 255, np.uint8))
    assert load_custom_dataset(tmp_path).has_masks
