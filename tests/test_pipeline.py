import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.train import build_models, evaluate_model, preprocess


@pytest.fixture
def sample_df():
    """Minimal fraud dataset — imbalanced like the real data."""
    rng = np.random.default_rng(42)
    n_normal, n_fraud = 200, 10
    n = n_normal + n_fraud

    data = {f"V{i}": rng.standard_normal(n) for i in range(1, 29)}
    data["Time"] = rng.uniform(0, 172792, n)
    data["Amount"] = rng.exponential(scale=88, size=n)
    data["Class"] = [0] * n_normal + [1] * n_fraud
    return pd.DataFrame(data)


def test_preprocess_shape(sample_df):
    X, y = preprocess(sample_df)
    assert X.shape[0] == len(sample_df)
    assert len(y) == len(sample_df)
    assert X.shape[1] == 30  # 28 V-features + Amount + Time


def test_preprocess_no_nan(sample_df):
    X, y = preprocess(sample_df)
    assert not np.isnan(X).any()
    assert not np.isnan(y).any()


def test_preprocess_target_binary(sample_df):
    _, y = preprocess(sample_df)
    assert set(y).issubset({0, 1})


def test_build_models_returns_three():
    models = build_models()
    assert len(models) == 3


def test_evaluate_model_returns_expected_keys(sample_df):
    X, y = preprocess(sample_df)
    model = LogisticRegression(max_iter=1000, random_state=42)
    metrics = evaluate_model("LR", model, X, y, X, y)
    assert "Precision (Fraud)" in metrics
    assert "Recall (Fraud)" in metrics
    assert "F1 (Fraud)" in metrics
    assert "ROC-AUC" in metrics
    assert "Train Time (s)" in metrics


def test_metrics_in_valid_range(sample_df):
    X, y = preprocess(sample_df)
    model = LogisticRegression(max_iter=1000, random_state=42)
    metrics = evaluate_model("LR", model, X, y, X, y)
    assert 0.0 <= metrics["Precision (Fraud)"] <= 1.0
    assert 0.0 <= metrics["Recall (Fraud)"] <= 1.0
    assert 0.0 <= metrics["F1 (Fraud)"] <= 1.0
    assert 0.0 <= metrics["ROC-AUC"] <= 1.0
