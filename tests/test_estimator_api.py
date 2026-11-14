import numpy as np
import pytest
import torch
from sklearn.base import clone
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture

from egmm.estimator import EGMM


def test_egmm_sklearn_api_compatibility():
    X, _ = make_blobs(n_samples=600, n_features=6, centers=4, cluster_std=0.8, random_state=42)
    est = EGMM(n_components=4, covariance_type="full", random_state=42, device="cpu")
    cloned = clone(est)
    assert cloned.get_params() == est.get_params()

    labels = est.fit_predict(X)
    assert labels.shape == (600,)
    assert set(np.unique(labels)).issubset({0, 1, 2, 3})

    probs = est.predict_proba(X[:20])
    assert probs.shape == (20, 4)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    pred = est.predict(X[:20])
    np.testing.assert_array_equal(pred, probs.argmax(axis=1))

    scores = est.score_samples(X[:20])
    assert scores.shape == (20,)
    assert np.isfinite(scores).all()
    np.testing.assert_allclose(est.score(X[:20]), scores.mean(), rtol=1e-5)

    assert np.isfinite(est.bic(X))
    assert np.isfinite(est.aic(X))
    assert est.aic(X) <= est.bic(X)

    samples, y_samples = est.sample(50)
    assert samples.shape == (50, 6)
    assert y_samples.shape == (50,)
    assert est.weights_.shape == (4,)
    assert est.means_.shape == (4, 6)
    assert est.covariances_.shape == (4, 6, 6)
    assert est.precisions_.shape == (4, 6, 6)
    assert est.precisions_cholesky_.shape == (4, 6, 6)


@pytest.mark.parametrize("cov_type", ["diag", "full"])
def test_egmm_unscaling_invariance_matches_sklearn_scoring(cov_type):
    rng = np.random.default_rng(0)
    X_base, _ = make_blobs(
        n_samples=1200, n_features=4, centers=3, cluster_std=0.7, random_state=0
    )
    # Huge shift and anisotropic feature scales to stress numerical stability
    scale = np.array([1e4, 1e-3, 50.0, 1.0])
    offset = np.array([1e6, -500.0, 2e3, 0.0])
    X = X_base * scale + offset
    X_train, X_test = X[:900], X[900:]

    egmm = EGMM(
        n_components=3,
        covariance_type=cov_type,
        max_epochs=15,
        batch_size=256,
        random_state=0,
        device="cpu",
    ).fit(X_train)

    sk = GaussianMixture(
        n_components=3, covariance_type=cov_type, random_state=0
    ).fit(X_train)

    # Check score_samples matches direct Gaussian logpdf from unscaled public attributes
    from egmm.kernels import compute_log_gaussian_prob

    w = torch.from_numpy(egmm.weights_).double()
    m = torch.from_numpy(egmm.means_).double()
    c = torch.from_numpy(egmm.covariances_).double()
    lp = compute_log_gaussian_prob(
        torch.from_numpy(X_test).double(), m, c, covariance_type=cov_type
    )
    direct_scores = torch.logsumexp(lp + torch.log(w).unsqueeze(0), dim=1).numpy()
    np.testing.assert_allclose(egmm.score_samples(X_test), direct_scores, rtol=1e-4, atol=1e-3)

    # Should achieve test log-likelihood competitive with sklearn
    assert egmm.score(X_test) >= sk.score(X_test) - 0.5


@pytest.mark.parametrize("cov_type", ["diag", "full"])
def test_egmm_cuda_and_partial_fit(cov_type):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    X, _ = make_blobs(n_samples=2000, n_features=8, centers=5, random_state=12)
    est = EGMM(
        n_components=5,
        covariance_type=cov_type,
        max_epochs=6,
        batch_size=512,
        device="cuda",
        random_state=12,
    ).fit(X)
    assert np.isfinite(est.score(X))
    assert (est.weights_ > 0.01).all()

    # partial_fit streaming
    est_pf = EGMM(
        n_components=5,
        covariance_type=cov_type,
        batch_size=500,
        device="cuda",
        random_state=12,
    )
    for chunk in np.array_split(X, 4):
        est_pf.partial_fit(chunk)
    assert np.isfinite(est_pf.score(X))


def test_egmm_mixed_precision_moments_prevent_cancellation():
    rng = np.random.default_rng(7)
    # 90% wide cluster around 0, 10% tight cluster at +25: standardized E[x^2] - E[x]^2
    # would suffer catastrophic cancellation if M, S, Q were accumulated in float32.
    x_wide = rng.normal(loc=0.0, scale=1.0, size=(1800, 3))
    x_tight = rng.normal(loc=25.0, scale=1e-2, size=(200, 3))
    X = np.vstack([x_wide, x_tight])
    rng.shuffle(X)

    est = EGMM(
        n_components=2,
        covariance_type="full",
        dtype="float32",
        reg_covar=1e-9,
        max_epochs=8,
        random_state=7,
        device="cpu",
    ).fit(X)

    assert est._M_t.dtype == torch.float64
    assert est._S_t.dtype == torch.float64
    assert est._Q_t.dtype == torch.float64
    assert est._means_std_t.dtype == torch.float32
    assert est._prec_factor_std_t.dtype == torch.float32

    tight_k = int(np.argmax(est.means_[:, 0]))
    recovered_var = float(np.diag(est.covariances_[tight_k]).mean())
    # True variance of tight cluster is (1e-2)^2 = 1e-4; float64 moments should preserve scale accurately.
    assert 3e-5 < recovered_var < 3e-4


@pytest.mark.parametrize("cov_type", ["diag", "full"])
def test_egmm_explicit_ablation_params_and_chunked_scoring(cov_type):
    X, _ = make_blobs(n_samples=1200, n_features=6, centers=4, random_state=21)
    est = EGMM(
        n_components=4,
        covariance_type=cov_type,
        polish_steps=0,
        init_usage_scale=1.0,
        device="cpu",
        random_state=21,
    ).fit(X)
    assert est.polish_steps == 0
    assert est.init_usage_scale == 1.0
    assert est.n_iter_ == est._hp.max_epochs

    # Verify chunked scoring produces identical results to unchunked scoring
    s_full = est.score_samples(X)
    p_full = est.predict_proba(X)
    s_chunk = est._score_samples_chunked(X, max_work_elements=200)
    p_chunk = est._predict_proba_chunked(X, max_work_elements=200)
    np.testing.assert_allclose(s_chunk, s_full, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(p_chunk, p_full, rtol=1e-5, atol=1e-5)


