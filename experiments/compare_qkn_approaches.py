"""Matched-split comparison of QKN Approach A, Approach B, and a classical baseline.

Approach A here isolates the existing QKN expectation-value feature map and uses
an RBF-SVC downstream classifier. It does NOT run the full spatiotemporal GNN:
the checked-in CSV is row-wise and contains no sequence, node, or edge-index
metadata needed to reconstruct that architecture. See docs/QDF_QKN_APPROACH.md.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pennylane as qml
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

from src.qkn_kernel import QuantumKernelSVC
from experiments.compare_qkn_kernel import DEFAULT_FEATURES, DEFAULT_LABEL, split_data, evaluate


def expectation_features(X: np.ndarray, n_qubits: int, layers: int = 2, seed: int = 42) -> np.ndarray:
    """Generate Approach-A-style Pauli-Z features using QKN's AngleEmbedding + entangler."""
    if X.ndim != 2 or X.shape[1] != n_qubits:
        raise ValueError("Approach A requires a 2-D matrix with exactly one feature per qubit.")
    if n_qubits < 2 or layers < 1:
        raise ValueError("n_qubits must be >= 2 and layers >= 1.")
    dev = qml.device("default.qubit", wires=n_qubits)
    rng = np.random.RandomState(seed)
    weights = rng.standard_normal((layers, n_qubits, 3))

    @qml.qnode(dev)
    def circuit(inputs, trainable_weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits))
        qml.StronglyEntanglingLayers(trainable_weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    return np.asarray([circuit(row, weights) for row in X], dtype=float)


def run_comparison(data_path, output_path, features=None, label=DEFAULT_LABEL,
                   test_size=0.25, random_state=42, n_qubits=3, C=1.0,
                   group_column=None, max_samples=250, layers=2):
    frame = pd.read_csv(data_path)
    features = features or DEFAULT_FEATURES
    missing = [col for col in features + [label] if col not in frame.columns]
    if missing:
        raise ValueError(f"Required columns missing from {data_path}: {missing}")
    required = features + [label] + ([group_column] if group_column else [])
    frame = frame.dropna(subset=required).copy()
    if group_column and frame[group_column].nunique() < 3:
        raise ValueError("Event-level evaluation requires at least 3 distinct groups.")
    if max_samples and len(frame) > max_samples:
        if group_column:
            # Sample whole events, never individual rows, before making the holdout.
            rng = np.random.RandomState(random_state)
            groups = frame[group_column].drop_duplicates().to_numpy()
            rng.shuffle(groups)
            selected_groups = []
            selected_rows = 0
            for group in groups:
                group_rows = int((frame[group_column] == group).sum())
                if selected_rows and selected_rows + group_rows > max_samples:
                    continue
                selected_groups.append(group)
                selected_rows += group_rows
                if selected_rows >= max_samples:
                    break
            frame = frame[frame[group_column].isin(selected_groups)].copy()
            if frame[group_column].nunique() < 3:
                raise ValueError("The row budget retained fewer than 3 complete events. Increase max_samples.")
        else:
            frame, _ = train_test_split(frame, train_size=max_samples, random_state=random_state,
                                        stratify=frame[label])
    X_train, X_test, y_train, y_test, split_kind = split_data(
        frame, features, label, test_size, random_state, group_column
    )
    test_groups = None
    if group_column:
        groups = frame[group_column].to_numpy()
        split = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(split.split(frame[features].to_numpy(), frame[label].to_numpy(), groups))
        test_groups = groups[test_idx]
        train_events = sorted(map(str, pd.unique(groups[train_idx])))
        test_events = sorted(map(str, pd.unique(test_groups)))
        if set(train_events) & set(test_events):
            raise RuntimeError("Event leakage detected: a group occurs in both partitions.")
    if len(features) != n_qubits:
        raise ValueError(f"Expected one qubit per feature: {len(features)} features, n_qubits={n_qubits}.")

    # Fit all transforms on training data only.
    angle_scaler = MinMaxScaler(feature_range=(-np.pi, np.pi))
    X_train_q = angle_scaler.fit_transform(X_train)
    X_test_q = angle_scaler.transform(X_test)
    classical_scaler = StandardScaler()
    X_train_c = classical_scaler.fit_transform(X_train)
    X_test_c = classical_scaler.transform(X_test)

    results = {
        "dataset": str(data_path), "label": label, "features": features,
        "split_type": split_kind, "random_state": random_state,
        "n_samples": int(len(frame)), "n_train": int(len(y_train)), "n_test": int(len(y_test)),
        "n_qubits": n_qubits, "layers": layers, "C": C,
        "train_event_ids": train_events if group_column else [],
        "test_event_ids": test_events if group_column else [],
        "comparison_scope": "Approach A feature-map isolation; full spatiotemporal GNN not run",
        "notes": (
            "Exploratory sample-level split; no event identifier is present. Do not interpret as unseen-storm generalization."
            if split_kind == "stratified_random_exploratory"
            else "Group holdout requested; verify the group column represents complete storm events."
        ),
        "models": {},
    }

    # A: QKN expectation values, followed by a shared RBF-SVC classifier.
    start = time.perf_counter()
    A_train = expectation_features(X_train_q, n_qubits, layers, random_state)
    A_test = expectation_features(X_test_q, n_qubits, layers, random_state)
    A_model = SVC(kernel="rbf", C=C, class_weight="balanced", random_state=random_state)
    A_model.fit(A_train, y_train)
    A_pred = A_model.predict(A_test)
    results["models"]["approach_a_expectation_features_rbf_svc"] = {
        **evaluate(y_test, A_pred),
        "fit_transform_and_predict_seconds": float(time.perf_counter() - start),
        "feature_shape": list(A_train.shape),
    }

    # B: fidelity kernel SVC; training and test partitions are exactly the same.
    start = time.perf_counter()
    B_model = QuantumKernelSVC(n_qubits=n_qubits, C=C, random_state=random_state)
    B_model.fit(X_train_q, y_train)
    B_pred = B_model.predict(X_test_q)
    results["models"]["approach_b_fidelity_kernel_svc"] = {
        **evaluate(y_test, B_pred),
        "fit_and_predict_seconds": float(time.perf_counter() - start),
        "kernel_train_shape": list(B_model.K_train_.shape),
    }

    # Classical control.
    start = time.perf_counter()
    C_model = SVC(kernel="rbf", C=C, class_weight="balanced", random_state=random_state)
    C_model.fit(X_train_c, y_train)
    C_pred = C_model.predict(X_test_c)
    results["models"]["classical_rbf_svc"] = {
        **evaluate(y_test, C_pred),
        "fit_and_predict_seconds": float(time.perf_counter() - start),
    }

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    pd.DataFrame([{"model": name, **metrics} for name, metrics in results["models"].items()]).to_csv(
        output.with_suffix(".csv"), index=False
    )
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/processed/final_quantum_dataset.csv")
    parser.add_argument("--output", default="outputs/qkn_approach_comparison.json")
    parser.add_argument("--features", nargs="+", default=DEFAULT_FEATURES)
    parser.add_argument("--label", default=DEFAULT_LABEL)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-qubits", type=int, default=3)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--group-column", default=None)
    parser.add_argument("--max-samples", type=int, default=250, help="Set 0 to use all rows.")
    args = parser.parse_args()
    result = run_comparison(args.data, args.output, args.features, args.label, args.test_size,
                            args.random_state, args.n_qubits, args.C, args.group_column,
                            args.max_samples, args.layers)
    print(json.dumps(result, indent=2))
    print(f"Saved metrics to {args.output} and {Path(args.output).with_suffix('.csv')}")


if __name__ == "__main__":
    main()
