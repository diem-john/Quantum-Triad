import numpy as np
import pytest
import torch

pytest.importorskip("pennylane")
pytest.importorskip("torch")

from src.hybrid_qstgnn import HybridQSTGNNFeatureEncoder
from src.qkn import QuantumSpatiotemporalGNN


class DummyQKN:
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits

    def extract_temporal_quantum_features(self, X, progress_callback=None):
        X = np.asarray(X)
        # Deterministic stand-in for A branch, preserving expected tensor shape.
        base = X[..., :self.n_qubits]
        return torch.as_tensor(np.transpose(base, (0, 1, 3, 2)), dtype=torch.float32)


def test_hybrid_encoder_train_only_landmarks_and_shape():
    rng = np.random.default_rng(3)
    X_train = rng.uniform(-1.0, 1.0, size=(3, 4, 3, 2))
    X_test = rng.uniform(-1.0, 1.0, size=(2, 4, 3, 2))
    encoder = HybridQSTGNNFeatureEncoder(n_qubits=2, n_landmarks=3, random_state=9).fit(X_train)
    assert encoder.landmarks_.shape == (3, 2)
    features = encoder.transform(X_test, DummyQKN(n_qubits=2))
    assert features.shape == (2, 4, 5, 3)
    assert torch.isfinite(features).all()


def test_encoder_state_round_trip():
    X = np.arange(24, dtype=float).reshape(1, 4, 3, 2) / 10
    encoder = HybridQSTGNNFeatureEncoder(n_qubits=2, n_landmarks=2).fit(X)
    restored = HybridQSTGNNFeatureEncoder.from_state_dict(encoder.state_dict())
    assert np.allclose(restored.landmarks_, encoder.landmarks_)


def test_fused_features_run_through_qstgnn():
    rng = np.random.default_rng(5)
    X = rng.uniform(-1.0, 1.0, size=(2, 4, 3, 2))
    encoder = HybridQSTGNNFeatureEncoder(n_qubits=2, n_landmarks=2).fit(X)
    fused = encoder.transform(X, DummyQKN(n_qubits=2))
    model = QuantumSpatiotemporalGNN(
        in_channels=encoder.output_channels, seq_len=3,
        conv_out=4, gat_heads=1, lstm_hidden=4, dropout=0.0
    )
    edges = torch.tensor([[0,1,1,2,2,3,3,0],[1,0,2,1,3,2,0,3]], dtype=torch.long)
    output = model(fused, edges)
    assert output.shape == (2, 4)
    assert torch.isfinite(output).all()

@pytest.mark.parametrize("weight_b", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_complementary_fusion_weights_are_applied_after_branch_projection(weight_b):
    """Verify H_fused = (1 - w_B) H_A + w_B H_B at the model fusion point."""
    torch.manual_seed(11)
    x = torch.randn(2, 4, 5, 3)  # A has 2 channels; B has 3 channels.
    model = QuantumSpatiotemporalGNN(
        in_channels=4,
        seq_len=3,
        conv_out=4,
        gat_heads=1,
        lstm_hidden=4,
        dropout=0.0,
        fusion_a_channels=2,
        fusion_b_channels=3,
        fusion_b_weight=weight_b,
    )
    model.eval()

    captured = {}

    def capture_conv_input(_module, inputs):
        captured["fused"] = inputs[0].detach().clone()

    hook = model.conv1.register_forward_pre_hook(capture_conv_input)
    edges = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 0], [1, 0, 2, 1, 3, 2, 0, 3]],
        dtype=torch.long,
    )
    with torch.no_grad():
        model(x, edges)
        a = x[:, :, :2, :].reshape(2 * 4, 2, 3)
        b = x[:, :, 2:, :].reshape(2 * 4, 3, 3)
        h_a = model.fusion_a_projection(a)
        h_b = model.fusion_b_projection(b)
        expected = ((1.0 - weight_b) * h_a + weight_b * h_b)
    hook.remove()

    torch.testing.assert_close(captured["fused"], expected)
    assert model.fusion_weights() == pytest.approx([1.0 - weight_b, weight_b])

