"""CUDA-Q backend for Quantum-Triad's Quantum Kernel Network.

This backend mirrors the circuit semantics of src.qkn.QuantumKernelNetwork
while allowing execution through CUDA-Q targets such as CPU and NVIDIA GPU.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch

try:
    import cudaq
    from cudaq import spin
except ImportError:
    cudaq = None
    spin = None

EntanglingType = Literal["StronglyEntangling", "BasicEntangler"]


from src.qkn_interface import QKNBackend


class CudaQQuantumKernelNetwork(QKNBackend):
    """CUDA-Q implementation of the Quantum-Triad quantum feature extractor."""

    def __init__(
        self,
        n_qubits: int = 3,
        layers: int = 2,
        entangling_type: EntanglingType = "StronglyEntangling",
        target: str | None = None,
        seed: int = 42,
    ) -> None:
        if cudaq is None:
            raise ImportError(
                "CUDA-Q is not installed. Install the CUDA-Q dependency "
                "before using CudaQQuantumKernelNetwork."
            )
        if n_qubits < 1 or layers < 1:
            raise ValueError("n_qubits and layers must both be >= 1.")
        if entangling_type not in {"StronglyEntangling", "BasicEntangler"}:
            raise ValueError(
                "entangling_type must be 'StronglyEntangling' or 'BasicEntangler'."
            )

        self.n_qubits = n_qubits
        self.layers = layers
        self.entangling_type = entangling_type

        if target is not None:
            if not cudaq.has_target(target):
                raise ValueError(f"CUDA-Q target '{target}' is unavailable.")
            cudaq.set_target(target)
        elif cudaq.num_available_gpus() > 0 and cudaq.has_target("nvidia"):
            cudaq.set_target("nvidia")
        elif cudaq.has_target("qpp-cpu"):
            cudaq.set_target("qpp-cpu")

        rng = np.random.RandomState(seed)
        if entangling_type == "StronglyEntangling":
            self.weights = rng.standard_normal(
                (layers, n_qubits, 3), dtype=np.float64
            )
        else:
            self.weights = rng.standard_normal(
                (layers, n_qubits), dtype=np.float64
            )

        self._kernel = self._build_kernel()

    def _build_kernel(self):
        """Build the CUDA-Q equivalent of the PennyLane QKN circuit."""

        n_qubits = self.n_qubits
        layers = self.layers
        strongly_entangling = self.entangling_type == "StronglyEntangling"

        @cudaq.kernel
        def kernel(features: list[float], weights: list[float]):
            q = cudaq.qvector(n_qubits)

            # PennyLane AngleEmbedding uses RX rotations by default.
            for i in range(n_qubits):
                rx(features[i], q[i])

            if strongly_entangling:
                # PennyLane Rot(a,b,c) is RZ(a)-RY(b)-RZ(c).
                # StronglyEntanglingLayers then applies ring CNOTs.
                for layer in range(layers):
                    base = layer * n_qubits * 3
                    for i in range(n_qubits):
                        offset = base + i * 3
                        rz(weights[offset], q[i])
                        ry(weights[offset + 1], q[i])
                        rz(weights[offset + 2], q[i])

                    if n_qubits > 1:
                        # Match PennyLane StronglyEntanglingLayers default:
                        # range = layer_index % (n_wires - 1) + 1.
                        entangling_range = layer % (n_qubits - 1) + 1
                        for i in range(n_qubits):
                            x.ctrl(q[i], q[(i + entangling_range) % n_qubits])
            else:
                # PennyLane BasicEntanglerLayers uses RX rotations followed
                # by ring CNOTs.
                for layer in range(layers):
                    base = layer * n_qubits
                    for i in range(n_qubits):
                        rx(weights[base + i], q[i])

                    if n_qubits > 1:
                        for i in range(n_qubits):
                            x.ctrl(q[i], q[(i + 1) % n_qubits])

        return kernel

    def _prepare_features(self, inputs) -> list[float]:
        values = np.asarray(inputs, dtype=np.float64).reshape(-1)
        padded = np.zeros(self.n_qubits, dtype=np.float64)
        padded[: min(values.size, self.n_qubits)] = values[: self.n_qubits]
        return padded.tolist()

    def _flat_weights(self) -> list[float]:
        return np.asarray(self.weights, dtype=np.float64).reshape(-1).tolist()

    def extract_quantum_features(self, inputs) -> np.ndarray:
        """Return Pauli-Z expectation values for one input vector."""

        features = self._prepare_features(inputs)
        weights = self._flat_weights()
        values = []

        # Keep one scalar observable per call. This is compatible with the
        # CUDA-Q observe API across simulator targets.
        for i in range(self.n_qubits):
            observable = spin.z(i)
            result = cudaq.observe(self._kernel, observable, features, weights)
            values.append(result.expectation())

        return np.asarray(values, dtype=np.float64)

    def extract_temporal_quantum_features(self, X_seq) -> torch.Tensor:
        """Match the PennyLane QKN output contract for 3-D/4-D inputs."""

        values = np.asarray(X_seq)

        if values.ndim == 4:
            n_samples, n_nodes, n_steps, _ = values.shape
            output = np.zeros(
                (n_samples, n_nodes, self.n_qubits, n_steps),
                dtype=np.float32,
            )

            for i in range(n_samples):
                for node in range(n_nodes):
                    for t in range(n_steps):
                        output[i, node, :, t] = self.extract_quantum_features(
                            values[i, node, t, :]
                        )

            return torch.from_numpy(output)

        if values.ndim == 3:
            n_samples, n_steps, _ = values.shape
            output = np.zeros(
                (n_samples, self.n_qubits, n_steps),
                dtype=np.float32,
            )

            for i in range(n_samples):
                for t in range(n_steps):
                    output[i, :, t] = self.extract_quantum_features(
                        values[i, t, :]
                    )

            return torch.from_numpy(output)

        raise ValueError(
            "X_seq must be 3-D (batch,time,features) or "
            "4-D (batch,nodes,time,features)."
        )

    def draw_circuit(self, sample_features=None) -> str:
        """Return the CUDA-Q textual circuit representation."""

        features = (
            self._prepare_features(sample_features)
            if sample_features is not None
            else [0.0] * self.n_qubits
        )
        return cudaq.draw(self._kernel, features, self._flat_weights())
