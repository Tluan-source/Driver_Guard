import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("scipy")
pytest.importorskip("sklearn")

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from driverguard.learning.features import transform_features
from driverguard.learning.model import VigilancePredictor


def test_relative_power_preserves_ratios_and_removes_channel_gain():
    x = np.log10(np.arange(1., 31.)).reshape(3, 10).astype(np.float32)
    relative = transform_features(x, "relative_log_power")
    np.testing.assert_allclose(np.power(10., relative).reshape(3, 2, 5).sum(axis=-1), 1., rtol=1e-6)
    shifted = (x.reshape(3, 2, 5) + np.array([3., -2.])[None, :, None]).reshape(3, 10)
    np.testing.assert_allclose(transform_features(shifted, "relative_log_power"), relative, atol=1e-6)
    with pytest.raises(ValueError, match="five bands"):
        transform_features(np.ones((2, 4)), "relative_log_power")


def test_portable_svm_matches_sklearn_decision(tmp_path):
    rng = np.random.default_rng(42)
    x = rng.normal(size=(50, 10)).astype(np.float32)
    mean, std = x.mean(axis=0), x.std(axis=0)
    z = (x - mean) / std
    labels = x[:, 0] > 0
    model = SVC(C=2, gamma="scale").fit(z, labels)
    path = tmp_path / "svm.pt"
    torch.save({"format_version": 1, "architecture": "rbf_svm", "context": 1,
                "feature_key": "de_movingAve", "feature_transform": "absolute", "mean": mean.tolist(),
                "std": std.tolist(), "decision_threshold": .5, "gamma": float(model._gamma),
                "support_vectors": model.support_vectors_.tolist(), "dual_coef": model.dual_coef_[0].tolist(),
                "intercept": float(model.intercept_[0])}, path)
    expected = 1 / (1 + np.exp(-model.decision_function(z)))
    np.testing.assert_allclose(VigilancePredictor(path).predict(x), expected, rtol=1e-5, atol=1e-6)


def test_relative_logistic_checkpoint_matches_sklearn(tmp_path):
    rng = np.random.default_rng(9)
    x = rng.normal(size=(60, 10)).astype(np.float32)
    z = transform_features(x, "relative_log_power")
    mean, std = z.mean(axis=0), z.std(axis=0)
    normalized = (z - mean) / std
    model = LogisticRegression(C=1., max_iter=2000).fit(normalized, x[:, 0] > 0)
    path = tmp_path / "logistic.pt"
    torch.save({"format_version": 1, "architecture": "logistic", "context": 1,
                "feature_key": "de_movingAve", "feature_transform": "relative_log_power",
                "mean": mean.tolist(), "std": std.tolist(), "decision_threshold": .5,
                "coef": model.coef_[0].tolist(), "intercept": float(model.intercept_[0])}, path)
    np.testing.assert_allclose(VigilancePredictor(path).predict(x), model.predict_proba(normalized)[:, 1],
                               rtol=1e-5, atol=1e-6)
