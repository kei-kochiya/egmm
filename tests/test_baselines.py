import numpy as np
import pytest
import torch
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture

from egmm.baselines import (
    CappeMoulinesOnlineGMM,
    EMeansInitGMM,
    NaiveMinibatchGMM,
    StandardizedSklearnGMM,
    TorchBatchGMM,
)


@pytest.mark.parametrize("cov_type", ["diag", "full"])
def test_torch_batch_gmm_matches_sklearn_on_blobs(cov_type):
    X, _ = make_blobs(n_samples=800, n_features=5, centers=4, cluster_std=0.8, random_state=7)
    X_train, X_test = X[:600], X[600:]

    tb = TorchBatchGMM(
        n_components=4, covariance_type=cov_type, max_iter=100, tol=1e-4, device="cpu", random_state=7
    ).fit(X_train)
    sk = GaussianMixture(
        n_components=4, covariance_type=cov_type, max_iter=100, tol=1e-4, random_state=7
    ).fit(X_train)

    assert np.isfinite(tb.score(X_test))
    np.testing.assert_allclose(tb.score(X_test), sk.score(X_test), atol=0.15)


def test_all_baselines_fit_and_score_cleanly():
    X, _ = make_blobs(n_samples=600, n_features=6, centers=3, random_state=11)
    X_train, X_test = X[:450], X[450:]

    models = [
        TorchBatchGMM(n_components=3, max_iter=20, device="cpu", random_state=0),
        CappeMoulinesOnlineGMM(
            n_components=3, max_epochs=5, batch_size=128, rho=0.75, device="cpu", random_state=0
        ),
        NaiveMinibatchGMM(
            n_components=3, max_epochs=5, batch_size=128, alpha=0.2, device="cpu", random_state=0
        ),
        EMeansInitGMM(
            n_components=3, kmeans_epochs=2, em_iters=5, batch_size=128, device="cpu", random_state=0
        ),
    ]
    for m in models:
        m.fit(X_train)
        s = m.score(X_test)
        p = m.predict_proba(X_test[:10])
        assert np.isfinite(s)
        assert p.shape == (10, 3)
        np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-5)


@pytest.mark.parametrize("cov_type", ["diag", "full"])
def test_standardized_sklearn_gmm_scale_invariance_and_polish_ablation(cov_type):
    X, _ = make_blobs(n_samples=600, n_features=6, centers=4, cluster_std=0.6, random_state=21)
    # Rescale coordinates drastically (1e-4 to 1e4) so raw reg_covar=1e-6 distorts tiny coordinates
    scales = np.array([1e-4, 1e-3, 1e-1, 10.0, 1e3, 1e4], dtype=np.float64)
    X_scaled = X * scales[None, :]
    log_jac_scales = float(-np.log(scales).sum())

    m_base = StandardizedSklearnGMM(n_components=4, covariance_type=cov_type, random_state=21).fit(X)
    m_scaled = StandardizedSklearnGMM(n_components=4, covariance_type=cov_type, random_state=21).fit(X_scaled)

    # Under exact standardization + log-Jacobian, log-density transforms by log_jac_scales
    np.testing.assert_allclose(m_scaled.score(X_scaled), m_base.score(X) + log_jac_scales, atol=1e-5)
    np.testing.assert_allclose(m_scaled.means_, m_base.means_ * scales[None, :], rtol=1e-4, atol=1e-5)

    # Also test CappeMoulinesOnlineGMM with matched polish_steps enabled
    cm_unpolished = CappeMoulinesOnlineGMM(
        n_components=4, covariance_type=cov_type, max_epochs=4, batch_size=128, polish_steps=0, device="cpu", random_state=21
    ).fit(X_scaled)
    cm_polished = CappeMoulinesOnlineGMM(
        n_components=4, covariance_type=cov_type, max_epochs=4, batch_size=128, polish_steps=2, init_usage_scale="auto", device="cpu", random_state=21
    ).fit(X_scaled)
    assert cm_polished.score(X_scaled) >= cm_unpolished.score(X_scaled) - 0.05

