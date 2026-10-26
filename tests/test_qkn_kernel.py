import numpy as np
import pytest
pytest.importorskip("pennylane")
pytest.importorskip("sklearn")
from src.qkn_kernel import QuantumFidelityKernel, QuantumKernelSVC

def test_fidelity_kernel_properties():
    k = QuantumFidelityKernel(n_qubits=2)
    X = np.array([[0.1, 0.2], [0.7, -0.3], [1.0, 0.4]])
    K = k.matrix(X)
    assert K.shape == (3, 3)
    assert np.all(np.isfinite(K))
    assert np.allclose(K, K.T, atol=1e-8)
    assert np.allclose(np.diag(K), 1.0, atol=1e-8)
    assert np.all(K >= -1e-10) and np.all(K <= 1.0 + 1e-10)

def test_identical_inputs_have_unit_fidelity():
    k = QuantumFidelityKernel(n_qubits=2)
    x = np.array([0.2, -0.1])
    assert k.evaluate(x, x) == pytest.approx(1.0, abs=1e-8)

def test_precomputed_svc_fit_predict():
    X = np.array([[-1.0, -0.8], [-0.8, -1.1], [-0.9, -0.6], [0.8, 0.9], [1.0, 0.7], [0.7, 1.1]])
    y = np.array([0, 0, 0, 1, 1, 1])
    model = QuantumKernelSVC(n_qubits=2).fit(X, y)
    assert model.predict(X).shape == y.shape
    assert model.K_train_.shape == (len(X), len(X))