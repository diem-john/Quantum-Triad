import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pennylane")
pytest.importorskip("sklearn")

from experiments.compare_qkn_approaches import run_comparison


def test_hybrid_qkn_is_evaluated_on_same_group_holdout(tmp_path):
    rows = []
    for event in range(6):
        for label in (0, 1):
            rows.append({
                "storm_id": f"storm-{event}",
                "f1": float(event * 0.2 + label * 0.7),
                "f2": float(event * -0.1 + label * 0.9),
                "Failure_Label": label,
            })
    data_path = tmp_path / "events.csv"
    output_path = tmp_path / "metrics.json"
    pd.DataFrame(rows).to_csv(data_path, index=False)

    result = run_comparison(
        data_path=data_path,
        output_path=output_path,
        features=["f1", "f2"],
        label="Failure_Label",
        test_size=0.33,
        random_state=17,
        n_qubits=2,
        C=1.0,
        group_column="storm_id",
        max_samples=0,
        layers=1,
    )

    models = result["models"]
    assert {
        "approach_a_expectation_features_rbf_svc",
        "approach_b_fidelity_kernel_svc",
        "hybrid_qkn_a_plus_b",
        "classical_rbf_svc",
    }.issubset(models)
    hybrid = models["hybrid_qkn_a_plus_b"]
    assert hybrid["expectation_feature_shape"][1] == 2
    assert hybrid["fidelity_similarity_shape"][1] == result["n_train"]
    assert hybrid["fused_feature_shape"][1] == 2 + result["n_train"]
    assert hybrid["recall"] >= 0.0
    assert json.loads(output_path.read_text(encoding="utf-8"))["models"]["hybrid_qkn_a_plus_b"]
    assert not set(result["train_event_ids"]) & set(result["test_event_ids"])
