"""PCA normal-subspace anomaly detector (our own implementation).

Mathematics
-----------
Let ``x`` be a patch feature vector.  Features are standardised with the
mean ``mu`` and scale ``s`` of the NORMAL training patches::

    z = (x - mu) / s

The covariance ``C = Z^T Z / (N - 1)`` of the standardised training patches is
eigen-decomposed, ``C = V diag(lambda) V^T`` and the ``k`` leading eigenvectors
``V_k`` (columns) span the *normal subspace*.  A feature is projected and
reconstructed::

    z_hat = V_k V_k^T z                (equivalently x_hat = mu + S V_k V_k^T S^-1 (x - mu))
    r     = z - z_hat                  (reconstruction residual)
    score = || r ||_2                  (anomaly score of the patch)

Patches whose appearance is not explained by the principal directions of
normal variation leave a large residual.  Only normal images are ever passed
to :meth:`fit`.

``k`` is either a fixed integer or the smallest number of components that
explains a requested fraction of the variance (``0 < n_components < 1``).
"""
from __future__ import annotations

import numpy as np


class PCAAnomalyDetector:
    def __init__(self, n_components: float | int = 0.95, scale_floor_rel: float = 0.05):
        if isinstance(n_components, bool) or n_components <= 0:
            raise ValueError("n_components must be a positive int or a float in (0, 1)")
        self.n_components = n_components
        self.scale_floor_rel = scale_floor_rel
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.components_: np.ndarray | None = None          # (k, D)
        self.explained_variance_: np.ndarray | None = None   # all eigenvalues, descending
        self.n_components_: int = 0

    # ------------------------------------------------------------------ fit
    def fit(self, X: np.ndarray) -> "PCAAnomalyDetector":
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"X must be (N, D), got {X.shape}")
        n, d = X.shape
        if n < 2:
            raise ValueError("PCA needs at least 2 feature vectors (is the training set empty?)")
        if not np.isfinite(X).all():
            raise ValueError("training features contain NaN/inf")

        mean = X.mean(axis=0, dtype=np.float64)
        std = X.std(axis=0, dtype=np.float64)
        # Constant (or nearly constant) dimensions: floor the scale relative to the median
        # so a tiny test-time deviation cannot explode after division.
        floor = max(self.scale_floor_rel * float(np.median(std)), 1e-8)
        scale = np.maximum(std, floor)
        Z = ((X - mean) / scale).astype(np.float32)

        cov = (Z.T @ Z).astype(np.float64) / (n - 1)
        evals, evecs = np.linalg.eigh(cov)                 # ascending
        order = np.argsort(evals)[::-1]
        evals = np.clip(evals[order], 0.0, None)
        evecs = evecs[:, order]

        max_rank = max(1, min(d, n - 1))
        total = float(evals.sum())
        if isinstance(self.n_components, float) and self.n_components < 1.0:
            if total <= 0:
                k = 1
            else:
                k = int(np.searchsorted(np.cumsum(evals) / total, self.n_components) + 1)
        else:
            k = int(self.n_components)
        k = int(np.clip(k, 1, max_rank))

        self.mean_, self.scale_ = mean, scale
        self.components_ = evecs[:, :k].T.astype(np.float32)   # (k, D)
        self.explained_variance_ = evals
        self.n_components_ = k
        return self

    # --------------------------------------------------------------- scoring
    def _check_fitted(self) -> None:
        if self.components_ is None:
            raise RuntimeError("detector is not fitted")

    def standardize(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return ((np.asarray(X) - self.mean_) / self.scale_).astype(np.float32)

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        """x_hat = mu + S V_k V_k^T S^-1 (x - mu) in the original feature space."""
        Z = self.standardize(X)
        Zh = (Z @ self.components_.T) @ self.components_
        return Zh * self.scale_ + self.mean_

    def residual_norm(self, X: np.ndarray) -> np.ndarray:
        """L2 norm of the reconstruction residual (in standardised space), shape (N,)."""
        Z = self.standardize(X)
        R = Z - (Z @ self.components_.T) @ self.components_
        return np.sqrt(np.einsum("ij,ij->i", R, R))

    def score_map(self, fmap: np.ndarray) -> np.ndarray:
        """(h, w, D) feature map -> (h, w) patch anomaly scores (spatial layout preserved)."""
        h, w, d = fmap.shape
        return self.residual_norm(fmap.reshape(-1, d)).reshape(h, w)

    @property
    def explained_variance_ratio_(self) -> np.ndarray:
        self._check_fitted()
        tot = self.explained_variance_.sum()
        return self.explained_variance_ / tot if tot > 0 else np.zeros_like(self.explained_variance_)

    # ----------------------------------------------------------- persistence
    def get_state(self) -> dict[str, np.ndarray]:
        self._check_fitted()
        return {
            "mean": self.mean_,
            "scale": self.scale_,
            "components": self.components_,
            "eigenvalues": self.explained_variance_,
        }

    def set_state(self, state: dict[str, np.ndarray]) -> "PCAAnomalyDetector":
        self.mean_ = np.asarray(state["mean"], np.float64)
        self.scale_ = np.asarray(state["scale"], np.float64)
        self.components_ = np.asarray(state["components"], np.float32)
        self.explained_variance_ = np.asarray(state["eigenvalues"], np.float64)
        self.n_components_ = int(self.components_.shape[0])
        return self
