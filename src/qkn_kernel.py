"""Explicit fidelity-kernel learning, separate from the expectation-feature QKN."""
from __future__ import annotations
from typing import Any
import numpy as np
import pennylane as qml
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class QuantumFidelityKernel:
    """Compute K(x,y)=|<phi(x)|phi(y)>|^2 with a data-dependent feature map."""
    def __init__(self, n_qubits: int = 4):
        if n_qubits < 1:
            raise ValueError("n_qubits must be >= 1")
        self.n_qubits = int(n_qubits)
        self.dev = qml.device("default.qubit", wires=self.n_qubits)
        @qml.qnode(self.dev)
        def state_circuit(features):
            for w in range(self.n_qubits):
                qml.Hadamard(wires=w)
                qml.RZ(2.0 * features[w], wires=w)
            pairs = [(i, i + 1) for i in range(self.n_qubits - 1)]
            if self.n_qubits > 2:
                pairs.append((self.n_qubits - 1, 0))
            for i, j in pairs:
                qml.CNOT(wires=[i, j])
                qml.RZ(2.0 * (np.pi - features[i]) * (np.pi - features[j]), wires=j)
                qml.CNOT(wires=[i, j])
            return qml.state()
        self._state_circuit = state_circuit

    def _prepare(self, x: Any) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(x)):
            raise ValueError("Kernel inputs must be finite.")
        out = np.zeros(self.n_qubits, dtype=np.float64)
        out[:min(len(x), self.n_qubits)] = x[:self.n_qubits]
        return out

    def state(self, x: Any) -> np.ndarray:
        return np.asarray(self._state_circuit(self._prepare(x)), dtype=np.complex128)

    def evaluate(self, x: Any, y: Any) -> float:
        overlap = np.vdot(self.state(x), self.state(y))
        return float(np.clip(abs(overlap) ** 2, 0.0, 1.0))

    def matrix(self, X: Any, Y: Any = None, progress_callback=None) -> np.ndarray:
        """Compute the fidelity matrix, optionally reporting row-wise progress."""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("X must have shape (samples, features)")
        same = Y is None
        Y = X if same else np.asarray(Y, dtype=np.float64)
        if Y.ndim != 2:
            raise ValueError("Y must have shape (samples, features)")
        K = np.empty((len(X), len(Y)), dtype=np.float64)
        total_rows = max(1, len(X))
        if same:
            for i in range(len(X)):
                K[i, i] = 1.0
                for j in range(i + 1, len(Y)):
                    K[i, j] = K[j, i] = self.evaluate(X[i], X[j])
                if progress_callback is not None:
                    progress_callback((i + 1) / total_rows)
        else:
            for i, x in enumerate(X):
                for j, y in enumerate(Y):
                    K[i, j] = self.evaluate(x, y)
                if progress_callback is not None:
                    progress_callback((i + 1) / total_rows)
        return K

class QuantumKernelSVC:
    """SVC trained on a quantum fidelity Gram matrix."""
    def __init__(self, n_qubits: int = 4, C: float = 1.0, class_weight="balanced", random_state=42):
        self.kernel = QuantumFidelityKernel(n_qubits=n_qubits)
        self.classifier = SVC(kernel="precomputed", C=C, class_weight=class_weight, random_state=random_state)
        self.X_train_ = None
        self.K_train_ = None

    def fit(self, X: Any, y: Any):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y).reshape(-1)
        if X.ndim != 2 or len(X) != len(y):
            raise ValueError("X must be 2-D and have the same number of rows as y.")
        if len(np.unique(y)) < 2:
            raise ValueError("Training labels must contain at least two classes.")
        self.X_train_ = X.copy()
        self.K_train_ = self.kernel.matrix(X)
        self.classifier.fit(self.K_train_, y)
        return self

    def predict(self, X: Any) -> np.ndarray:
        if self.X_train_ is None:
            raise RuntimeError("Call fit before predict.")
        K_test = self.kernel.matrix(np.asarray(X, dtype=np.float64), self.X_train_)
        return self.classifier.predict(K_test)

    def score_metrics(self, X: Any, y: Any) -> dict[str, float]:
        y = np.asarray(y).reshape(-1)
        pred = self.predict(X)
        return {"accuracy": float(accuracy_score(y, pred)),
                "precision": float(precision_score(y, pred, zero_division=0)),
                "recall": float(recall_score(y, pred, zero_division=0)),
                "f1": float(f1_score(y, pred, zero_division=0))}