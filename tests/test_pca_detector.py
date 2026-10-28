import numpy as np
import pytest

from src.config import DetectorConfig
from src.models.multiscale_detector import MultiScaleAnomalyDetector, image_score_from_map
from src.models.pca_detector import PCAAnomalyDetector


def _low_rank(n=500, d=12, rank=3, noise=0.01, seed=0):
    rng = np.random.default_rng(seed)
    basis = np.linalg.qr(rng.normal(size=(d, rank)))[0]
    return (rng.normal(size=(n, rank)) * [5, 3, 2][:rank]) @ basis.T + rng.normal(scale=noise, size=(n, d)), basis


def test_fit_recovers_rank_and_small_residual_inside_subspace():
    X, _ = _low_rank()
    det = PCAAnomalyDetector(n_components=0.99).fit(X)
    assert det.n_components_ == 3
    assert det.residual_norm(X).mean() < 0.5


def test_off_subspace_point_has_large_residual():
    X, basis = _low_rank(noise=0.05)
    det = PCAAnomalyDetector(n_components=3).fit(X)
    rng = np.random.default_rng(1)
    ortho = np.linalg.qr(np.hstack([basis, rng.normal(size=(12, 1))]))[0][:, -1]
    anomalous = X[:50] + 4.0 * ortho * X.std()
    assert det.residual_norm(anomalous).mean() > 5 * det.residual_norm(X[:50]).mean()


def test_reconstruction_equals_formula():
    """x_hat = mu + V_k V_k^T (x - mu) when features already have unit scale."""
    X, _ = _low_rank()
    X = (X - X.mean(0)) / X.std(0)
    det = PCAAnomalyDetector(n_components=3).fit(X)
    V = det.components_.T                   # (D, k)
    manual = det.mean_ + (X[:5] - det.mean_) @ V @ V.T
    assert np.allclose(det.reconstruct(X[:5]), manual, atol=1e-4)
    r = X[:5] - det.reconstruct(X[:5])
    assert np.allclose(np.linalg.norm(r, axis=1), det.residual_norm(X[:5]), atol=1e-3)


def test_variance_ratio_vs_integer_components():
    X, _ = _low_rank()
    assert PCAAnomalyDetector(2).fit(X).n_components_ == 2
    assert PCAAnomalyDetector(0.5).fit(X).n_components_ <= 2


def test_guards_degenerate_inputs():
    with pytest.raises(ValueError):
        PCAAnomalyDetector().fit(np.zeros((1, 5)))            # single sample
    with pytest.raises(ValueError):
        PCAAnomalyDetector().fit(np.array([[np.nan, 1.0], [1.0, 2.0]]))
    with pytest.raises(ValueError):
        PCAAnomalyDetector(0)
    const = np.ones((20, 4))
    det = PCAAnomalyDetector(0.9).fit(const)                  # zero variance everywhere
    assert np.isfinite(det.residual_norm(const)).all()
    few = np.random.default_rng(0).normal(size=(3, 50))       # N < D
    det = PCAAnomalyDetector(100).fit(few)
    assert det.n_components_ <= 2
    with pytest.raises(RuntimeError):
        PCAAnomalyDetector().residual_norm(few)


def test_score_map_preserves_spatial_layout():
    X, _ = _low_rank(d=6, rank=2)
    det = PCAAnomalyDetector(2).fit(X)
    fmap = X[:20].reshape(4, 5, 6)
    sm = det.score_map(fmap)
    assert sm.shape == (4, 5)
    assert np.allclose(sm.ravel(), det.residual_norm(X[:20]))


def test_state_roundtrip():
    X, _ = _low_rank()
    a = PCAAnomalyDetector(0.95).fit(X)
    b = PCAAnomalyDetector(0.95).set_state(a.get_state())
    assert np.allclose(a.residual_norm(X[:30]), b.residual_norm(X[:30]))


def _feats(rng, n=6):
    return [{"fine": rng.normal(size=(16, 16, 5)).astype(np.float32),
             "coarse": rng.normal(size=(8, 8, 5)).astype(np.float32)} for _ in range(n)]


def test_multiscale_maps_have_image_size_and_fusion_is_convex():
    rng = np.random.default_rng(0)
    det = MultiScaleAnomalyDetector(DetectorConfig(n_components=3, smooth_sigma=1.0), map_size=64).fit(_feats(rng))
    f = _feats(rng, 1)[0]
    z = det.normalize(det.raw_maps(f))
    assert z["fine"].shape == z["coarse"].shape == (64, 64)
    fused = det.fuse(z, 0.3)
    assert np.allclose(fused, 0.3 * z["fine"] + 0.7 * z["coarse"], atol=1e-5)
    assert np.array_equal(det.fuse(z, 1.0), z["fine"]) and np.array_equal(det.fuse(z, 0.0), z["coarse"])
    with pytest.raises(ValueError):
        det.fuse(z, 1.5)


def test_crossfit_calibration_uses_heldout_scores_and_is_less_optimistic():
    rng = np.random.default_rng(0)
    feats = _feats(rng, 8)
    cross = MultiScaleAnomalyDetector(DetectorConfig(n_components=3, calibration="crossfit", calib_folds=4), 32).fit(feats)
    resub = MultiScaleAnomalyDetector(DetectorConfig(n_components=3, calibration="resubstitution"), 32).fit(feats)
    assert cross.calibration_mode_.startswith("crossfit") and resub.calibration_mode_ == "resubstitution"
    assert cross.calib_raw_["fine"].shape[0] == 8
    # held-out residuals are on average larger than residuals on the fitting images
    assert cross.calib_raw_["fine"].mean() >= resub.calib_raw_["fine"].mean()
    assert len(cross.calibration_scores(0.5)) == 8


def test_image_score_modes():
    m = np.zeros((10, 10)); m[0, 0] = 5.0
    assert image_score_from_map(m, "max") == 5.0
    assert image_score_from_map(m, "topk_mean", 0.01) == 5.0
    assert image_score_from_map(m, "topk_mean", 0.1) < 5.0
