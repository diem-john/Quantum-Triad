"""Compare explicit quantum fidelity-kernel SVC with a classical RBF-SVC.

Usage:
    python -m experiments.compare_qkn_kernel --data data/processed/final_quantum_dataset.csv

The checked-in CSV has no storm/event identifier, so this runner uses a reproducible
stratified sample split and marks it as exploratory. Do not interpret it as
unseen-storm generalization. Use --group-column once an event ID is available.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

from src.qkn_kernel import QuantumKernelSVC


DEFAULT_FEATURES = [
    "Wind_Speed (rad)",
    "Delta_Wind (rad)",
    "Coastal_Exposure (rad)",
]
DEFAULT_LABEL = "Failure_Label"


def split_data(frame, features, label, test_size, random_state, group_column=None):
    X = frame[features].to_numpy(dtype=float)
    y = frame[label].to_numpy()
    if not np.isfinite(X).all():
        raise ValueError("Feature matrix contains NaN or infinite values.")
    if pd.isna(y).any():
        raise ValueError("Labels contain missing values.")

    if group_column:
        if group_column not in frame:
            raise ValueError(f"Group column {group_column!r} not found in dataset.")
        groups = frame[group_column].to_numpy()
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(splitter.split(X, y, groups))
        if set(np.unique(y[train_idx])) != set(np.unique(y)):
            raise ValueError("Group split omitted a class from training; use more events or adjust split.")
        return X[train_idx], X[test_idx], y[train_idx], y[test_idx], "group_holdout"

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test, "stratified_random_exploratory"


def evaluate(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def run_experiment(data_path, output_path, features=None, label=DEFAULT_LABEL,
                   test_size=0.25, random_state=42, n_qubits=3, C=1.0,
                   group_column=None, max_samples=None):
    frame = pd.read_csv(data_path)
    features = features or DEFAULT_FEATURES
    missing = [c for c in features + [label] if c not in frame.columns]
    if missing:
        raise ValueError(f"Required columns missing from {data_path}: {missing}")
    frame = frame.dropna(subset=features + [label]).copy()
    if max_samples and len(frame) > max_samples:
        # Reproducible stratified downsampling for fast circuit simulation.
        frame, _ = train_test_split(
            frame, train_size=max_samples, random_state=random_state,
            stratify=frame[label]
        )

    X_train, X_test, y_train, y_test, split_kind = split_data(
        frame, features, label, test_size, random_state, group_column
    )
    if n_qubits != len(features):
        raise ValueError(
            f"Use one qubit per selected feature for this experiment: "
            f"{len(features)} features but n_qubits={n_qubits}. "
            "Feature truncation/padding is intentionally not allowed in the benchmark."
        )

    # Fit scaling on training data only. Both models see the same scaled inputs.
    angle_scaler = MinMaxScaler(feature_range=(-np.pi, np.pi))
    X_train_q = angle_scaler.fit_transform(X_train)
    X_test_q = angle_scaler.transform(X_test)

    classical_scaler = StandardScaler()
    X_train_c = classical_scaler.fit_transform(X_train)
    X_test_c = classical_scaler.transform(X_test)

    results = {
        "dataset": str(data_path),
        "label": label,
        "features": features,
        "split_type": split_kind,
        "random_state": random_state,
        "n_samples": int(len(frame)),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "n_qubits": n_qubits,
        "C": C,
        "notes": (
            "Exploratory sample-level split only; dataset has no event identifier. "
            "Do not interpret as unseen-storm generalization."
            if split_kind == "stratified_random_exploratory"
            else "Group holdout requested; verify group column represents complete storm events."
        ),
        "models": {},
    }

    start = time.perf_counter()
    q_model = QuantumKernelSVC(n_qubits=n_qubits, C=C)
    q_model.fit(X_train_q, y_train)
    q_pred = q_model.predict(X_test_q)
    q_elapsed = time.perf_counter() - start
    results["models"]["quantum_fidelity_svc"] = {
        **evaluate(y_test, q_pred),
        "fit_and_predict_seconds": float(q_elapsed),
        "kernel_train_shape": list(q_model.K_train_.shape),
    }

    start = time.perf_counter()
    c_model = SVC(kernel="rbf", C=C, class_weight="balanced")
    c_model.fit(X_train_c, y_train)
    c_pred = c_model.predict(X_test_c)
    c_elapsed = time.perf_counter() - start
    results["models"]["classical_rbf_svc"] = {
        **evaluate(y_test, c_pred),
        "fit_and_predict_seconds": float(c_elapsed),
    }

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    flat = []
    for model_name, metrics in results["models"].items():
        flat.append({"model": model_name, **metrics})
    pd.DataFrame(flat).to_csv(output.with_suffix(".csv"), index=False)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/processed/final_quantum_dataset.csv")
    parser.add_argument("--output", default="outputs/qkn_kernel_comparison.json")
    parser.add_argument("--features", nargs="+", default=DEFAULT_FEATURES)
    parser.add_argument("--label", default=DEFAULT_LABEL)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-qubits", type=int, default=3)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--group-column", default=None)
    parser.add_argument("--max-samples", type=int, default=250,
                        help="Cap dataset size for simulator runtime; set 0 to use all rows.")
    args = parser.parse_args()
    results = run_experiment(
        data_path=args.data,
        output_path=args.output,
        features=args.features,
        label=args.label,
        test_size=args.test_size,
        random_state=args.random_state,
        n_qubits=args.n_qubits,
        C=args.C,
        group_column=args.group_column,
        max_samples=args.max_samples or None,
    )
    print(json.dumps(results, indent=2))
    print(f"\nSaved metrics to {args.output} and {Path(args.output).with_suffix('.csv')}")


if __name__ == "__main__":
    main()
