"""Common interface for Quantum-Triad Phase 3 QKN backends."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch


@runtime_checkable
class QKNBackend(Protocol):
    """Contract shared by all Phase 3 quantum feature-extraction backends."""

    n_qubits: int
    layers: int
    entangling_type: str

    def extract_quantum_features(self, inputs: Any) -> np.ndarray:
        """Return one Pauli-Z expectation value per configured qubit."""

    def extract_temporal_quantum_features(self, X_seq: Any) -> torch.Tensor:
        """Map 3-D/4-D temporal inputs to the common tensor contract."""

    def draw_circuit(self, sample_features: Any = None) -> str:
        """Return a textual circuit representation when supported."""
