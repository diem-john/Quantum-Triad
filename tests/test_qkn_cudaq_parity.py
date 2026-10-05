"""Parity tests for the PennyLane and CUDA-Q QKN implementations."""

import numpy as np
import pytest

pytest.importorskip("cudaq")
torch = pytest.importorskip("torch")

from src.qkn import QuantumKernelNetwork
from src.qkn_cudaq import CudaQQuantumKernelNetwork


@pytest.mark.parametrize("entangling_type", ["StronglyEntangling", "BasicEntangler"])
def test_single_vector_matches_pennylane(entangling_type):
    pennylane_qkn = QuantumKernelNetwork(
        n_qubits=3,
        layers=2,
        entangling_type=entangling_type,
    )
    cudaq_qkn = CudaQQuantumKernelNetwork(
        n_qubits=3,
        layers=2,
        entangling_type=entangling_type,
        target="qpp-cpu",
    )

    features = np.array([0.37, -1.12, 2.21], dtype=np.float64)

    expected = np.asarray(
        pennylane_qkn.qnode(features, pennylane_qkn.weights),
        dtype=np.float64,
    )
    actual = cudaq_qkn.extract_quantum_features(features)

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_temporal_output_contract():
    cudaq_qkn = CudaQQuantumKernelNetwork(
        n_qubits=3,
        layers=2,
        target="qpp-cpu",
    )

    x = np.array(
        [
            [
                [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
                [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]],
            ]
        ],
        dtype=np.float64,
    )

    output = cudaq_qkn.extract_temporal_quantum_features(x)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (1, 2, 3, 2)
    assert output.dtype == torch.float32
