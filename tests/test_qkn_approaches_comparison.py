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


def test_group_holdout_keeps_events_disjoint_and_reports_ids(tmp_path):
    rng = np.random.RandomState(17)
    rows = []
    for event_id in range(6):
        for row_id in range(8):
            label = int((event_id + row_id) % 2)
            rows.append({
                "a": float(rng.normal() + label),
                "b": float(rng.normal()),
                "c": float(rng.normal()),
                "label": label,
                "storm_id": f"storm-{event_id}",
            })
    data_path = tmp_path / "events.csv"
    output_path = tmp_path / "event_metrics.json"
    pd.DataFrame(rows).to_csv(data_path, index=False)

    result = run_comparison(
        data_path, output_path, features=["a", "b", "c"], label="label",
        n_qubits=3, layers=1, max_samples=0, group_column="storm_id",
        test_size=0.33, random_state=7,
    )
    train_events = set(result["train_event_ids"])
    test_events = set(result["test_event_ids"])
    assert train_events
    assert test_events
    assert train_events.isdisjoint(test_events)
    assert train_events | test_events == {f"storm-{i}" for i in range(6)}
