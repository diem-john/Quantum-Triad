"""Hybrid QKN(A+B) feature fusion for the quantum spatiotemporal GNN.

Inputs are raw grid tensors shaped (samples, nodes, time, features). The
expectation-value branch (A) and fidelity-to-training-landmarks branch (B)
are computed per node/time step and concatenated into channels expected by
QuantumSpatiotemporalGNN. Fidelity landmarks are selected from training data
only and must be persisted for validation, testing, and inference.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.qkn_kernel import QuantumFidelityKernel


@dataclass
class HybridQSTGNNFeatureEncoder:
    """Build a leakage-aware A+B quantum representation for graph sequences."""

    n_qubits: int = 3
    n_landmarks: int = 8
    random_state: int = 42
    b_weight: float = 1.0
    save_visualizations: bool = True
    visualization_dir: str = "artifacts/kernel_visualizations"
    save_arrays: bool = True

    def __post_init__(self):
        if self.n_qubits < 2:
            raise ValueError("n_qubits must be >= 2")
        if self.n_landmarks < 1:
            raise ValueError("n_landmarks must be >= 1")
        if not 0.0 <= float(self.b_weight) <= 1.0:
            raise ValueError("b_weight must be between 0 and 1.")
        self.kernel = QuantumFidelityKernel(n_qubits=self.n_qubits)
        self.landmarks_: np.ndarray | None = None

    @staticmethod
    def _validate_tensor(X: Any) -> np.ndarray:
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim != 4:
            raise ValueError("Expected raw X with shape (samples, nodes, time, features).")
        if arr.shape[0] < 1 or arr.shape[1] < 1 or arr.shape[2] < 1 or arr.shape[3] < 1:
            raise ValueError("All dimensions of X must be non-empty.")
        if not np.isfinite(arr).all():
            raise ValueError("Input grid tensor contains non-finite values.")
        return arr

    def fit(self, X_train: Any) -> "HybridQSTGNNFeatureEncoder":
        X = self._validate_tensor(X_train)
        # Each node/time feature vector is a candidate reference point. Select
        # references exclusively from training samples to prevent leakage.
        rows = X.reshape(-1, X.shape[-1])
        n = min(int(self.n_landmarks), len(rows))
        rng = np.random.default_rng(self.random_state)
        chosen = np.sort(rng.choice(len(rows), size=n, replace=False))
        self.landmarks_ = rows[chosen].copy()
        return self

    def transform(self, X: Any, qkn_backend: Any, progress_callback=None) -> torch.Tensor:
        if self.landmarks_ is None:
            raise RuntimeError("Call fit(X_train) before transform.")
        arr = self._validate_tensor(X)
        # A branch: existing QKN expectation-value feature extractor.
        def report(fraction):
            if progress_callback is not None:
                progress_callback(0.70 * float(fraction), "Extracting QKN expectation features")

        A = qkn_backend.extract_temporal_quantum_features(arr, progress_callback=report if progress_callback is not None else None)
        if not isinstance(A, torch.Tensor):
            A = torch.as_tensor(A, dtype=torch.float32)
        A = A.detach().cpu().float()
        expected = (arr.shape[0], arr.shape[1], self.n_qubits, arr.shape[2])
        if tuple(A.shape) != expected:
            raise ValueError(f"QKN branch returned {tuple(A.shape)}; expected {expected}.")

        # B branch: for every node/time step, compute fidelity to the train-only
        # landmark bank. Result shape (samples, nodes, landmarks, time).
        n_samples, n_nodes, n_steps, n_features = arr.shape
        rows = arr.reshape(-1, n_features)
        def report_fidelity_progress(fraction):
            if progress_callback is not None:
                progress_callback(
                    0.70 + 0.25 * float(fraction),
                    f"Computing quantum fidelity features ({float(fraction):.0%})"
                )

        similarities = self.kernel.matrix(
            rows,
            self.landmarks_,
            progress_callback=report_fidelity_progress if progress_callback is not None else None,
        )
        if progress_callback is not None:
            progress_callback(0.95, "Quantum fidelity features computed")
        B = similarities.reshape(n_samples, n_nodes, n_steps, len(self.landmarks_))
        B = np.transpose(B, (0, 1, 3, 2)).copy()

        # Save compact diagnostics for both representation branches. The A plot
        # summarizes one sample/node; the B plot shows the actual fidelity
        # similarities (subsampled only for display when there are many rows).
        if self.save_visualizations or self.save_arrays:
            self._save_kernel_artifacts(A, B, similarities, n_samples, n_nodes, n_steps)

        B = torch.as_tensor(B, dtype=torch.float32)
        # A is the primary representation. Scale B before concatenation so the
        # fidelity branch contributes a smaller signal to the downstream GNN.
        B = B * float(self.b_weight)

        # Feature fusion occurs before temporal and graph message passing.
        fused = torch.cat([A, B], dim=2).contiguous()
        if progress_callback is not None:
            progress_callback(1.0, "Quantum representation complete")
        return fused

    def _save_kernel_artifacts(
        self, A: torch.Tensor, B: np.ndarray, similarities: np.ndarray,
        n_samples: int, n_nodes: int, n_steps: int,
    ) -> None:
        """Persist quantum features, fidelity similarities, plots, and metadata."""
        out_dir = Path(self.visualization_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        a_array = A.detach().cpu().numpy()
        if self.save_arrays:
            np.savez_compressed(
                out_dir / f"quantum_kernel_artifacts_{stamp}.npz",
                expectation_features=a_array,
                fidelity_features=B,
                fidelity_similarity_matrix=similarities,
                training_landmarks=self.landmarks_,
            )
            metadata = {
                "timestamp": stamp,
                "n_samples": int(n_samples),
                "n_nodes": int(n_nodes),
                "n_time_steps": int(n_steps),
                "n_qubits": int(self.n_qubits),
                "n_landmarks": int(len(self.landmarks_)),
                "b_weight": float(self.b_weight),
                "expectation_feature_shape": list(a_array.shape),
                "fidelity_feature_shape": list(B.shape),
                "fidelity_matrix_shape": list(similarities.shape),
                "landmarks_selected_from_training_data_only": True,
            }
            (out_dir / f"kernel_metadata_{stamp}.json").write_text(
                json.dumps(metadata, indent=2), encoding="utf-8"
            )

        if not self.save_visualizations:
            return

        a_view = A[0].mean(dim=0).detach().cpu().numpy()
        fig, ax = plt.subplots(figsize=(max(7, n_steps * 0.35), 4.5))
        image = ax.imshow(a_view, aspect="auto", interpolation="nearest", cmap="viridis")
        ax.set_title("Quantum Feature Extraction (QKN expectation values)")
        ax.set_xlabel("Time step")
        ax.set_ylabel("Qubit / feature channel")
        fig.colorbar(image, ax=ax, label="Expectation value")
        fig.tight_layout()
        fig.savefig(out_dir / f"quantum_features_{stamp}.png", dpi=180)
        plt.close(fig)

        display_rows = min(len(similarities), 256)
        row_ids = np.linspace(0, len(similarities) - 1, display_rows, dtype=int)
        kernel_view = similarities[row_ids]
        fig, ax = plt.subplots(figsize=(max(7, len(self.landmarks_) * 0.6), 5))
        image = ax.imshow(kernel_view, aspect="auto", interpolation="nearest",
                          vmin=0.0, vmax=1.0, cmap="magma")
        ax.set_title("Quantum Fidelity Kernel (input rows vs training landmarks)")
        ax.set_xlabel("Training landmark")
        ax.set_ylabel(f"Input row (up to {display_rows} displayed)")
        fig.colorbar(image, ax=ax, label="Fidelity similarity")
        fig.tight_layout()
        fig.savefig(out_dir / f"quantum_fidelity_kernel_{stamp}.png", dpi=180)
        plt.close(fig)

    @property
    def output_channels(self) -> int:
        if self.landmarks_ is None:
            raise RuntimeError("Encoder has not been fitted.")
        return self.n_qubits + len(self.landmarks_)

    def state_dict(self) -> dict[str, Any]:
        if self.landmarks_ is None:
            raise RuntimeError("Encoder has not been fitted.")
        return {
            "n_qubits": int(self.n_qubits),
            "n_landmarks": int(len(self.landmarks_)),
            "random_state": int(self.random_state),
            "b_weight": float(self.b_weight),
            "landmarks": self.landmarks_.tolist(),
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "HybridQSTGNNFeatureEncoder":
        encoder = cls(
            n_qubits=int(state["n_qubits"]),
            n_landmarks=int(state["n_landmarks"]),
            random_state=int(state.get("random_state", 42)),
            b_weight=float(state.get("b_weight", 0.25)),
        )
        encoder.landmarks_ = np.asarray(state["landmarks"], dtype=np.float64)
        if encoder.landmarks_.ndim != 2 or encoder.landmarks_.shape[1] < 1:
            raise ValueError("Invalid landmark array in encoder state.")
        return encoder
