import numpy as np
import pytest

from src.evaluation import metrics as M
from src.evaluation.evaluator import AnomalyEvaluator, size_bin_of


def test_auroc_known_values():
    y = np.array([0, 0, 1, 1])
    assert M.safe_auroc(y, [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert M.safe_auroc(y, [0.9, 0.8, 0.2, 0.1]) == 0.0
    assert M.safe_auroc(y, [0.5, 0.5, 0.5, 0.5]) == 0.5


def test_auroc_undefined_with_single_class():
    assert np.isnan(M.safe_auroc([0, 0, 0], [0.1, 0.2, 0.3]))
    assert np.isnan(M.safe_average_precision([1, 1], [0.1, 0.2]))


def test_precision_recall_f1_known_values():
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    p = np.array([1, 1, 0, 1, 0, 0, 0, 0])
    r = M.precision_recall_f1(y, p)
    assert (r["tp"], r["fp"], r["fn"], r["tn"]) == (2, 1, 1, 4)
    assert r["precision"] == pytest.approx(2 / 3) and r["recall"] == pytest.approx(2 / 3)
    assert r["f1"] == pytest.approx(2 / 3) and r["fpr"] == pytest.approx(1 / 5)


def test_prf_edge_cases():
    r = M.precision_recall_f1(np.array([1, 1, 0]), np.array([0, 0, 0]))     # nothing predicted
    assert np.isnan(r["precision"]) and r["recall"] == 0.0 and r["f1"] == 0.0
    r = M.precision_recall_f1(np.array([0, 0]), np.array([0, 0]))           # nothing to find
    assert np.isnan(r["f1"])


def test_best_f1_is_oracle_upper_bound():
    y = np.array([0, 0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    f, t = M.best_f1(y, s)
    assert f == 1.0 and 0.3 < t <= 0.8
    assert np.isnan(M.best_f1(np.zeros(3), np.arange(3))[0])


def test_class_distribution_reports_imbalance():
    d = M.class_distribution(np.array([0] * 9 + [1]))
    assert d["n_anomalous"] == 1 and d["anomalous_fraction"] == pytest.approx(0.1)


def _toy(n_norm=6, n_def=4, size=32, seed=0):
    rng = np.random.default_rng(seed)
    maps = rng.normal(0, 0.3, (n_norm + n_def, size, size)).astype(np.float32)
    gt = np.zeros((n_norm + n_def, size, size), bool)
    for i in range(n_norm, n_norm + n_def):
        gt[i, 10:16, 10:16] = True
        maps[i, 10:16, 10:16] += 3.0
    labels = np.array([0] * n_norm + [1] * n_def)
    scores = maps.reshape(len(maps), -1).max(1)
    return labels, scores, maps, gt


def test_evaluator_perfect_separation():
    labels, scores, maps, gt = _toy()
    res = AnomalyEvaluator().evaluate(labels, scores, maps, threshold=1.5, gt_masks=gt)
    assert res["image_auroc"] == 1.0 and res["pixel_auroc"] > 0.99
    assert res["image_recall"] == 1.0 and res["image_precision"] == 1.0 and res["image_fpr"] == 0.0
    assert res["pixel_f1"] > 0.9 and res["pixel_metrics_available"]
    assert res["image_n_anomalous"] == 4 and res["image_anomalous_fraction"] == pytest.approx(0.4)


def test_pixel_metrics_not_reported_without_masks():
    labels, scores, maps, _ = _toy()
    res = AnomalyEvaluator().evaluate(labels, scores, maps, threshold=1.5, gt_masks=None)
    assert not res["pixel_metrics_available"]
    assert np.isnan(res["pixel_auroc"]) and np.isnan(res["pixel_f1"])
    assert res["image_auroc"] == 1.0                     # image-level still available


def test_threshold_changes_prf_but_not_auroc():
    labels, scores, maps, gt = _toy()
    ev = AnomalyEvaluator()
    lo = ev.evaluate(labels, scores, maps, -5.0, gt)
    hi = ev.evaluate(labels, scores, maps, 100.0, gt)
    assert lo["image_auroc"] == hi["image_auroc"]
    assert lo["image_fpr"] == 1.0 and hi["image_recall"] == 0.0


def test_min_area_decision_matches_pipeline_rule():
    labels, scores, maps, gt = _toy()
    maps = maps.copy()
    maps[0, 0, 0] = 10.0                                 # single hot pixel in a NORMAL image
    scores = maps.reshape(len(maps), -1).max(1)
    ev = AnomalyEvaluator()
    assert ev.evaluate(labels, scores, maps, 1.5, gt)["image_fpr"] > 0           # flagged without blob filter
    assert ev.evaluate(labels, scores, maps, 1.5, gt, min_area=5)["image_fpr"] == 0.0


def test_size_bins_and_group_metrics():
    assert size_bin_of(0.004) == "small" and size_bin_of(0.02) == "medium" and size_bin_of(0.2) == "large"
    labels, scores, maps, gt = _toy()
    rows = AnomalyEvaluator().evaluate_by_size(labels, scores, maps, gt, 1.5)
    by = {r["size_bin"]: r for r in rows}
    # each defect covers 36/1024 = 3.5% of the image -> 'medium'
    assert by["medium"]["n_defective"] == 4 and by["small"]["n_defective"] == 0 and by["large"]["n_defective"] == 0
    assert np.isnan(by["small"]["image_auroc"]) and by["medium"]["image_auroc"] == 1.0
    assert by["medium"]["pixel_recall"] > 0.9


def test_fast_ranking_metrics_match_sklearn():
    rng = np.random.default_rng(0)
    y = rng.random(20_000) < 0.05
    s = rng.normal(size=y.size) + 1.5 * y
    s[:500] = np.round(s[:500], 1)                      # include ties
    rk = M.ranking_metrics(y, s)
    assert rk["auroc"] == pytest.approx(M.safe_auroc(y, s), abs=1e-9)
    assert rk["ap"] == pytest.approx(M.safe_average_precision(y, s), abs=1e-9)
    assert rk["best_f1"] == pytest.approx(M.best_f1(y, s)[0], abs=1e-9)
    assert np.isnan(M.ranking_metrics(np.zeros(5, bool), np.arange(5))["auroc"])


def test_pixel_subsampling_is_seeded():
    y = np.zeros(10_000, bool); y[:100] = True
    s = np.random.default_rng(0).normal(size=y.size)
    a = M.subsample_pixels(y, s, 1000, seed=3)
    b = M.subsample_pixels(y, s, 1000, seed=3)
    assert np.array_equal(a[0], b[0]) and len(a[0]) < 2000 and a[0].any()
