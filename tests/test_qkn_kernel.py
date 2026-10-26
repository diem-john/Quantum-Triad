import numpy as np
import pytest

pytest.importorskip("pennylane")
pytest.importorskip("sklearn")

from src.qkn_kernel import QuantumFidelityKernel, QuantumKernelSVC


def test_fidelity_kernel_properties():
    kernel = QuantumFidelityKernel(n_qubits=2)
    X = np.array([[0.1, 0.2], [0.7, -0.3], [1.0, 0.4]])
    K = kernel.matrix(X)

    assert K.shape == (3, 3)
    assert np.all(np.isfinite(K))
    assert np.allclose(K, K.T, atol=1e-8)
    assert np.allclose(np.diag(K), 1.0, atol=1e-8)
    assert np.all(K >= -1e-10) and np.all(K <= 1.0 + 1e-10)


def test_fidelity_gram_matrix_is_positive_semidefinite_within_tolerance():
    kernel = QuantumFidelityKernel(n_qubits=2)
    X = np.array([[-0.8, 0.1], [0.2, 0.4], [1.0, -0.7], [0.5, 0.9]])
    K = kernel.matrix(X)
    eigenvalues = np.linalg.eigvalsh(K)
    assert eigenvalues.min() >= -1e-7


def test_cross_kernel_matrix_shape_and_values():
    kernel = QuantumFidelityKernel(n_qubits=2)
    X_train = np.array([[0.1, 0.2], [0.7, -0.3], [1.0, 0.4]])
    X_test = np.array([[0.3, 0.1], [-0.2, 0.8]])
    K_cross = kernel.matrix(X_test, X_train)

    assert K_cross.shape == (len(X_test), len(X_train))
    assert np.all(np.isfinite(K_cross))
    assert np.all(K_cross >= -1e-10) and np.all(K_cross <= 1.0 + 1e-10)


def test_identical_inputs_have_unit_fidelity():
    kernel = QuantumFidelityKernel(n_qubits=2)
    x = np.array([0.2, -0.1])
    assert kernel.evaluate(x, x) == pytest.approx(1.0, abs=1e-8)


def test_non_finite_kernel_inputs_are_rejected():
    kernel = QuantumFidelityKernel(n_qubits=2)
    with pytest.raises(ValueError, match="finite"):
        kernel.evaluate([0.1, np.nan], [0.1, 0.2])


def test_precomputed_svc_fit_predict():
    X = np.array(
        [
            [-1.0, -0.8],
            [-0.8, -1.1],
            [-0.9, -0.6],
            [0.8, 0.9],
            [1.0, 0.7],
            [0.7, 1.1],
        ]
    )
    y = np.array([0, 0, 0, 1, 1, 1])
    model = QuantumKernelSVC(n_qubits=2).fit(X, y)

    assert model.predict(X).shape == y.shape
    assert model.K_train_.shape == (len(X), len(X))
    assert np.allclose(model.K_train_, model.K_train_.T, atol=1e-8)
