import numpy as np
import pandas as pd
import pytest

from experiments.compare_qkn_approaches import expectation_features, run_comparison


def test_expectation_features_shape_and_range():
    X = np.array([[-1.0, 0.0, 1.0], [0.5, -0.5, 0.2]])
    out = expectation_features(X, n_qubits=3, layers=1, seed=4)
    assert out.shape == (2, 3)
    assert np.isfinite(out).all()
    assert np.all(out >= -1.000001) and np.all(out <= 1.000001)


def test_approach_comparison_writes_three_models(tmp_path):
    rng = np.random.RandomState(2)
    X = rng.normal(size=(48, 3))
    y = np.array([0, 1] * 24)
    frame = pd.DataFrame(X, columns=["a", "b", "c"])
    frame["label"] = y
    data_path = tmp_path / "tiny.csv"
    output_path = tmp_path / "metrics.json"
    frame.to_csv(data_path, index=False)

    result = run_comparison(data_path, output_path, features=["a", "b", "c"],
                            label="label", n_qubits=3, layers=1, max_samples=0)
    assert set(result["models"]) == {
        "approach_a_expectation_features_rbf_svc",
        "approach_b_fidelity_kernel_svc",
        "classical_rbf_svc",
    }
    assert output_path.exists()
    assert output_path.with_suffix(".csv").exists()
    for metrics in result["models"].values():
        assert {"accuracy", "precision", "recall", "f1"} <= set(metrics)
