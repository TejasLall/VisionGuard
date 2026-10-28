"""Headless Streamlit tests: every page renders without exceptions (uses st.testing AppTest)."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.data.synthetic import create_synthetic_dataset
from src.utils.paths import PROJECT_ROOT

APP = str(PROJECT_ROOT / "app.py")
PAGES = ["app_pages/dashboard.py", "app_pages/inspect.py", "app_pages/compare.py", "app_pages/training.py",
         "app_pages/evaluation.py"]


@pytest.fixture(scope="module")
def dataset_root(tmp_path_factory) -> str:
    root = tmp_path_factory.mktemp("appdata")
    create_synthetic_dataset(root, n_train=6, n_test_good=2, n_test_per_defect=1, seed=3,
                             categories=("synthetic_texture",))
    return str(root)


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_without_exception(page, dataset_root):
    at = AppTest.from_file(APP, default_timeout=120)
    at.session_state["dataset_root"] = dataset_root
    at.run()
    at.switch_page(page).run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.title[0].value                                     # page title rendered


def test_inspect_dataset_image_runs_inference(dataset_root):
    if not any((PROJECT_ROOT / "models").glob("*/state.npz")):
        pytest.skip("no saved model in models/ (run scripts/train.py first)")
    at = AppTest.from_file(APP, default_timeout=180)
    at.session_state["dataset_root"] = dataset_root
    at.run()
    at.switch_page("app_pages/inspect.py").run()
    src = next(w for w in at.segmented_control if w.label == "Image source")
    src.set_value("Dataset test image").run()
    assert not at.exception, [e.value for e in at.exception]
    labels = [m.label for m in at.metric]
    assert "Verdict" in labels and "Threshold T" in labels


def test_compare_page_runs_all_five_variants(dataset_root):
    if not any((PROJECT_ROOT / "models").glob("*__illum/state.npz")):
        pytest.skip("needs a __raw/__illum model pair in models/")
    at = AppTest.from_file(APP, default_timeout=180)
    at.session_state["dataset_root"] = dataset_root
    at.run()
    at.switch_page("app_pages/compare.py").run()
    next(w for w in at.segmented_control if w.label == "Image source").set_value("Dataset test image").run()
    assert not at.exception, [e.value for e in at.exception]
    text = " ".join(m.value for m in at.markdown)
    for key in "ABCDE":
        assert f"**{key} ·" in text
