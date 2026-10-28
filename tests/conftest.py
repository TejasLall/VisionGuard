import pytest

from src.config import PipelineConfig
from src.data.synthetic import create_synthetic_dataset


@pytest.fixture(scope="session")
def synthetic_root(tmp_path_factory):
    """Tiny SYNTHETIC MVTec-style dataset (software testing only)."""
    root = tmp_path_factory.mktemp("synthetic")
    create_synthetic_dataset(root, n_train=10, n_test_good=4, n_test_per_defect=3, seed=1,
                             categories=("synthetic_texture",))
    return root


@pytest.fixture()
def small_config():
    cfg = PipelineConfig()
    cfg.features.image_size = 128          # keep tests fast
    cfg.features.fine_window, cfg.features.fine_stride = 7, 4
    cfg.features.coarse_window, cfg.features.coarse_stride = 25, 8
    cfg.detector.smooth_sigma = 2.0
    cfg.detector.multiscale = True
    return cfg
