import numpy as np
import pandas as pd
import pytest

from experiments.compare_qkn_kernel import evaluate, split_data


def test_metrics_are_computed_for_binary_predictions():
    metrics = evaluate(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]))
    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(0.8)


def test_sample_split_is_reproducible_and_stratified():
    frame = pd.DataFrame({
        "x": np.arange(40, dtype=float),
        "label": [0] * 20 + [1] * 20,
    })
    a = split_data(frame, ["x"], "label", 0.25, 7)
    b = split_data(frame, ["x"], "label", 0.25, 7)
    assert a[4] == "stratified_random_exploratory"
    assert np.array_equal(a[0], b[0])
    assert np.array_equal(a[2], b[2])
    assert set(np.unique(a[2])) == {0, 1}


def test_group_split_keeps_groups_disjoint():
    frame = pd.DataFrame({
        "x": np.arange(60, dtype=float),
        "label": [0, 1] * 30,
        "storm_id": np.repeat(np.arange(10), 6),
    })
    X_train, X_test, y_train, y_test, split_kind = split_data(
        frame, ["x"], "label", 0.3, 17, group_column="storm_id"
    )
    train_groups = set(frame.iloc[np.where(np.isin(frame["x"], X_train[:, 0]))]["storm_id"])
    test_groups = set(frame.iloc[np.where(np.isin(frame["x"], X_test[:, 0]))]["storm_id"])
    assert split_kind == "group_holdout"
    assert train_groups.isdisjoint(test_groups)
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)


def test_group_split_requires_existing_group_column():
    frame = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0], "label": [0, 0, 1, 1]})
    with pytest.raises(ValueError, match="not found"):
        split_data(frame, ["x"], "label", 0.25, 1, group_column="storm_id")
