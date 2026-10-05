"""Backend factory for Quantum-Triad quantum feature extraction."""

from __future__ import annotations

from typing import Any

from src.qkn import QuantumKernelNetwork


def create_qkn(backend: str = "pennylane", **kwargs: Any):
    """Create a QKN implementation by backend name.

    Supported backends:
        pennylane: Existing production/reference implementation.
        cudaq: NVIDIA CUDA-Q implementation.

    CUDA-Q remains an optional dependency and is imported only when selected.
    """

    normalized = backend.strip().lower()

    if normalized in {"pennylane", "pl"}:
        return QuantumKernelNetwork(**kwargs)

    if normalized in {"cudaq", "cuda-q", "nvidia"}:
        from src.qkn_cudaq import CudaQQuantumKernelNetwork

        return CudaQQuantumKernelNetwork(**kwargs)

    raise ValueError(
        f"Unsupported QKN backend '{backend}'. "
        "Choose 'pennylane' or 'cudaq'."
    )
