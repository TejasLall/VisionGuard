import numpy as np
import pytest

from src.config import ThresholdConfig
from src.models.thresholding import AdaptiveThreshold, apply_threshold


def test_fixed_threshold_ignores_data():
    t = AdaptiveThreshold(ThresholdConfig(method="fixed", fixed_value=4.2)).fit(None)
    assert t.threshold == 4.2


def test_mean_std_formula():
    s = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    t = AdaptiveThreshold(ThresholdConfig(method="mean_std", k=2.0)).fit(s)
    assert t.threshold == pytest.approx(s.mean() + 2.0 * s.std(ddof=1))


def test_percentile_formula_and_configurable():
    s = np.arange(101, dtype=float)
    assert AdaptiveThreshold(ThresholdConfig(method="percentile", percentile=90)).fit(s).threshold == pytest.approx(90.0)
    assert AdaptiveThreshold(ThresholdConfig(method="percentile", percentile=50)).fit(s).threshold == pytest.approx(50.0)


def test_adaptive_requires_normal_scores():
    with pytest.raises(ValueError):
        AdaptiveThreshold(ThresholdConfig(method="mean_std")).fit(np.array([]))
    with pytest.raises(ValueError):
        AdaptiveThreshold(ThresholdConfig(method="mean_std")).fit(np.array([np.nan]))
    with pytest.raises(RuntimeError):
        _ = AdaptiveThreshold(ThresholdConfig(method="mean_std")).threshold


def test_invalid_method_rejected():
    with pytest.raises(ValueError):
        AdaptiveThreshold(ThresholdConfig(method="magic"))


def test_larger_k_gives_higher_threshold():
    s = np.random.default_rng(0).normal(5, 1, 40)
    lo = AdaptiveThreshold(ThresholdConfig(method="mean_std", k=1)).fit(s).threshold
    hi = AdaptiveThreshold(ThresholdConfig(method="mean_std", k=4)).fit(s).threshold
    assert hi > lo


def test_apply_threshold_mask_and_min_area():
    m = np.zeros((20, 20)); m[2:4, 2:4] = 5.0; m[10:15, 10:15] = 5.0
    mask = apply_threshold(m, 3.0)
    assert mask.dtype == bool and mask.sum() == 4 + 25
    cleaned = apply_threshold(m, 3.0, min_area=10)
    assert cleaned.sum() == 25 and not cleaned[2:4, 2:4].any()
    assert not apply_threshold(m, 6.0).any()
