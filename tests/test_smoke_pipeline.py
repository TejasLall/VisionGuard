"""End-to-end checks on the SYNTHETIC dataset (software tests, not performance claims)."""
import json
import shutil

import numpy as np
import pandas as pd
import pytest

from src.config import PipelineConfig, ThresholdConfig
from src.data.dataset_loader import load_mvtec_category
from src.data.image_io import load_image_rgb, resize_square
from src.experiments.reporting import make_experiment_plots, write_summary_markdown
from src.experiments.robustness import add_deltas, default_transforms
from src.experiments.runner import STANDARD_VARIANTS, ExperimentRunner, select_train_paths
from src.features import create_feature_extractor
from src.features.base import BackendUnavailableError
from src.models.model_manager import ModelManager, model_name
from src.models.pipeline import AnomalyPipeline


@pytest.fixture(scope="module")
def data(synthetic_root):
    return load_mvtec_category(synthetic_root, "synthetic_texture")


def _images(data, size, n=None):
    return [resize_square(load_image_rgb(p), size) for p in data.train_paths[:n]]


def test_handcrafted_scales_differ_in_resolution_and_receptive_field(small_config, data):
    ext = create_feature_extractor(small_config.features)
    f = ext.extract(_images(data, 128, 1)[0])
    assert f["fine"].shape[:2] == (32, 32) and f["coarse"].shape[:2] == (16, 16)
    assert f["fine"].shape[-1] == f["coarse"].shape[-1] == 19
    assert np.isfinite(f["fine"]).all()
    d = ext.describe()["scales"]
    assert d["fine"]["window_px"] != d["coarse"]["window_px"]


def test_dinov2_unavailable_is_reported_not_faked(small_config):
    cfg = small_config.copy()
    cfg.features.backend = "dinov2"
    cfg.features.dino_model = "definitely/not-a-real-model"
    with pytest.raises(BackendUnavailableError):
        create_feature_extractor(cfg.features)
    cfg.features.allow_fallback = True
    ext = create_feature_extractor(cfg.features)
    assert ext.name == "handcrafted" and "FALLBACK" in ext.label and "NOT DINOv2" in ext.label
    pipe = AnomalyPipeline(cfg)
    assert pipe.config.features.backend == "handcrafted"      # configuration records the truth


def test_fit_predict_shapes_and_masks(small_config, data):
    pipe = AnomalyPipeline(small_config)
    info = pipe.fit(_images(data, 128))
    assert info["n_train_images"] == 10 and info["calibration_mode"].startswith("crossfit")
    img = load_image_rgb(data.test_samples[-1].image_path)       # a defective image
    for alpha in (1.0, 0.5):
        pred = pipe.predict(img, alpha=alpha, thr_cfg=ThresholdConfig(method="mean_std", k=3))
        assert pred.heatmap.shape == (128, 128) and pred.mask.shape == (128, 128)
        assert pred.mask.dtype == bool and pred.image.shape == (128, 128, 3)
        assert pred.max_score >= pred.mean_score
        assert pred.is_anomalous == bool(pred.image_score >= pred.threshold)
    assert set(pipe.analyze(img, ("fine",)).z_maps) == {"fine"}


def test_fit_requires_enough_images_and_prediction_requires_fit(small_config, data):
    with pytest.raises(ValueError):
        AnomalyPipeline(small_config).fit(_images(data, 128, 1))
    with pytest.raises(RuntimeError):
        AnomalyPipeline(small_config).predict(_images(data, 128, 1)[0])


def test_detects_synthetic_defects_better_than_chance(small_config, data):
    pipe = AnomalyPipeline(small_config)
    pipe.fit(_images(data, 128))
    scores, labels = [], []
    for s in data.test_samples:
        p = pipe.predict(load_image_rgb(s.image_path), alpha=0.5)
        scores.append(p.image_score); labels.append(s.label)
    from src.evaluation.metrics import safe_auroc
    assert safe_auroc(labels, scores) > 0.8        # sanity bound for a trivially separable synthetic set


def test_fitting_never_touches_test_data_or_masks(small_config, data, tmp_path):
    """Delete every test image and mask: fitting and threshold calibration must still work and be identical."""
    ref = AnomalyPipeline(small_config); ref.fit(_images(data, 128))
    t_ref = ref.threshold_for(ThresholdConfig(method="mean_std", k=3), 0.5).threshold
    root = data.train_paths[0].parents[3]          # dataset root (…/<category>/train/good/<img>)
    clone = tmp_path / "train_only"
    shutil.copytree(root, clone, ignore=shutil.ignore_patterns("test", "ground_truth"))
    paths = sorted((clone / "synthetic_texture" / "train" / "good").glob("*.png"))
    imgs = [resize_square(load_image_rgb(p), 128) for p in paths]
    other = AnomalyPipeline(small_config); other.fit(imgs)
    assert other.threshold_for(ThresholdConfig(method="mean_std", k=3), 0.5).threshold == pytest.approx(t_ref)


def test_adaptive_threshold_depends_only_on_normal_calibration(small_config, data):
    pipe = AnomalyPipeline(small_config)
    pipe.fit(_images(data, 128))
    t = pipe.threshold_for(ThresholdConfig(method="mean_std", k=3), 0.5)
    pipe.predict(load_image_rgb(data.test_samples[-1].image_path), alpha=0.5)
    assert pipe.threshold_for(ThresholdConfig(method="mean_std", k=3), 0.5).threshold == t.threshold
    cal = pipe.detector.calibration_scores(0.5)
    assert t.threshold == pytest.approx(cal.mean() + 3 * cal.std(ddof=1))


def test_enhancements_toggle_independently(small_config, data):
    imgs = _images(data, 128)
    plain = AnomalyPipeline(small_config); plain.fit(imgs)
    cfg = small_config.copy(); cfg.preprocess.enabled = True
    illum = AnomalyPipeline(cfg); illum.fit(imgs)
    img = load_image_rgb(data.test_samples[-1].image_path)
    assert np.array_equal(plain.analyze(img).processed, resize_square(img, 128))     # preprocessing off
    assert not np.array_equal(illum.analyze(img).processed, resize_square(img, 128))  # preprocessing on
    single = plain.predict(img, alpha=1.0).heatmap
    multi = plain.predict(img, alpha=0.5).heatmap
    assert not np.allclose(single, multi)
    assert plain.threshold_for(ThresholdConfig(method="fixed", fixed_value=3.0), 1.0).threshold == 3.0
    assert plain.threshold_for(ThresholdConfig(method="mean_std"), 1.0).threshold != 3.0


def test_model_save_load_roundtrip(small_config, data, tmp_path):
    cfg = small_config.copy(); cfg.preprocess.enabled = True
    pipe = AnomalyPipeline(cfg); pipe.fit(_images(data, 128))
    mgr = ModelManager(tmp_path)
    name = model_name("synthetic texture", "handcrafted", True)
    assert name == "synthetic_texture__handcrafted__illum"
    folder = mgr.save(pipe, name, {"category": "synthetic_texture"})
    assert (folder / "state.npz").is_file() and json.loads((folder / "meta.json").read_text())["name"] == name
    assert mgr.list_models() == [name]
    loaded = mgr.load(name)
    img = load_image_rgb(data.test_samples[-1].image_path)
    thr = ThresholdConfig(method="percentile", percentile=95)
    a, b = pipe.predict(img, alpha=0.5, thr_cfg=thr), loaded.predict(img, alpha=0.5, thr_cfg=thr)
    assert np.allclose(a.heatmap, b.heatmap, atol=1e-4)
    assert a.threshold == pytest.approx(b.threshold, rel=1e-3)
    assert np.array_equal(a.mask, b.mask) or (a.mask != b.mask).mean() < 0.001
    with pytest.raises(FileNotFoundError):
        mgr.load("nope")


def test_select_train_paths_is_deterministic(data):
    a = select_train_paths(data.train_paths, 4, seed=1)
    b = select_train_paths(data.train_paths, 4, seed=1)
    assert a == b and len(a) == 4 and set(a) <= set(data.train_paths)
    assert select_train_paths(data.train_paths, None, 1) == data.train_paths


@pytest.fixture(scope="module")
def experiment(synthetic_root, tmp_path_factory):
    cfg = PipelineConfig()
    cfg.features.image_size = 128
    cfg.features.fine_window, cfg.features.fine_stride = 7, 4
    cfg.features.coarse_window, cfg.features.coarse_stride = 25, 8
    cfg.detector.smooth_sigma = 2.0
    out = tmp_path_factory.mktemp("out")
    runner = ExperimentRunner(cfg, n_train=8, experiment_name="smoke", output_root=out, latency_images=2)
    transforms = {k: v for k, v in default_transforms().items() if k in ("identity", "dark_x0.5")}
    return runner.run(synthetic_root, ["synthetic_texture"], transforms=transforms)


def test_experiment_outputs_are_generated(experiment):
    out = experiment.out_dir
    for f in ("results.csv", "size_analysis.csv", "config.json", "summary.csv"):
        assert (out / f).is_file(), f
    df = pd.read_csv(out / "results.csv")
    assert set(df["variant"]) == {"A", "B", "C", "D", "E"}
    required = {"experiment", "timestamp", "category", "feature_backend", "preprocess_enabled", "alpha",
                "multiscale", "threshold_method", "image_auroc", "pixel_auroc", "image_precision", "image_recall",
                "image_f1", "mean_inference_ms", "n_train", "n_calibration"}
    assert required <= set(df.columns)
    assert (df["n_train"] == 8).all() and set(df["transform"]) == {"identity", "dark_x0.5"}
    meta = json.loads((out / "config.json").read_text())
    assert meta["seed"] == 0 and len(meta["variants"]) == 5


def test_variants_match_their_definition(experiment):
    df = experiment.results
    row = lambda k: df[(df["variant"] == k) & (df["transform"] == "identity")].iloc[0]
    assert not row("A").preprocess_enabled and not row("A").multiscale and row("A").threshold_method == "fixed"
    assert row("B").preprocess_enabled and not row("B").multiscale
    assert row("C").multiscale and row("C").alpha == 0.5 and not row("C").preprocess_enabled
    assert row("D").threshold_method == "mean_std" and not row("D").multiscale
    assert row("E").preprocess_enabled and row("E").multiscale and row("E").threshold_method == "mean_std"
    assert row("A").image_auroc == pytest.approx(row("D").image_auroc)       # threshold does not change AUROC
    assert row("A").threshold_value == 3.0 and row("D").threshold_value != 3.0
    assert 0.0 <= row("E").image_auroc <= 1.0 and 0.0 <= row("E").pixel_auroc <= 1.0
    assert (df["mean_inference_ms"] > 0).all()


def test_plots_summary_and_deltas(experiment):
    out = experiment.out_dir
    paths = make_experiment_plots(experiment.results, experiment.size_results, out)
    assert all(p.is_file() and p.stat().st_size > 1000 for p in paths)
    md = write_summary_markdown(experiment.results, experiment.size_results, out, "smoke")
    assert "Mean over categories" in md.read_text(encoding="utf-8")
    d = add_deltas(experiment.results)
    ident = d[d["transform"] == "identity"]
    assert np.allclose(ident["delta_image_auroc"].dropna(), 0.0)
