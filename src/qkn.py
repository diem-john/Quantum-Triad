import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import pennylane as qml
from pennylane import numpy as pnp
import numpy as np


class TemporalAttention(nn.Module):
    """
    Calculates attention weights across the temporal dimension of the LSTM output,
    allowing the model to focus on the exact time-step where structural failure becomes inevitable.
    """

    def __init__(self, hidden_size):
        super(TemporalAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, lstm_out):
        # lstm_out shape: (Batch*Nodes, seq_len, hidden_size)
        attn_weights = self.attention(lstm_out)
        attn_weights = F.softmax(attn_weights, dim=1)

        # Context vector: Weighted sum of LSTM outputs across time
        context = torch.sum(attn_weights * lstm_out, dim=1)
        return context, attn_weights


class QuantumSpatiotemporalGNN(nn.Module):
    """
    Advanced Q-STGNN Architecture specifically designed to process Batched 4D Tensors:
    (Batch, Nodes, Channels, Sequence_Length)
    """

    def __init__(
        self, in_channels=4, seq_len=4, conv_out=16, gat_heads=2,
        lstm_hidden=32, dropout=0.2, fusion_a_channels=None,
        fusion_b_channels=None, fusion_b_weight=0.25,
    ):
        super(QuantumSpatiotemporalGNN, self).__init__()

        # Project A and B independently to a shared latent width, then blend
        # with complementary coefficients. For hybrid inputs, in_channels is
        # the fused width, not the raw concatenated input width.
        self.fusion_a_channels = int(fusion_a_channels) if fusion_a_channels is not None else None
        self.fusion_b_channels = int(fusion_b_channels) if fusion_b_channels is not None else None
        self.fusion_b_weight = float(fusion_b_weight)
        self.fusion_a_projection = None
        self.fusion_b_projection = None
        if self.fusion_a_channels is not None or self.fusion_b_channels is not None:
            if self.fusion_a_channels is None or self.fusion_b_channels is None:
                raise ValueError("Hybrid fusion requires both fusion_a_channels and fusion_b_channels.")
            if self.fusion_a_channels < 1 or self.fusion_b_channels < 1:
                raise ValueError("Hybrid branch channel counts must be positive.")
            if not 0.0 <= self.fusion_b_weight <= 1.0:
                raise ValueError("fusion_b_weight must be between 0 and 1.")
            self.fusion_a_projection = nn.Conv1d(self.fusion_a_channels, int(in_channels), kernel_size=1)
            self.fusion_b_projection = nn.Conv1d(self.fusion_b_channels, int(in_channels), kernel_size=1)

        # 1. TEMPORAL BOTTLENECK: Smooths the quantum features over time
        self.conv1 = nn.Conv1d(in_channels, conv_out, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(conv_out)

        # 2. SPATIAL ROUTING: Shares the smoothed features across the IEEE-33 power lines
        # Dropout added here to prevent spatial overfitting
        self.gat = GATConv(
            in_channels=conv_out,
            out_channels=conv_out,
            heads=gat_heads,
            concat=True,
            dropout=dropout
        )
        gat_out_dim = conv_out * gat_heads

        # 3. MOMENTUM EXTRACTION: Analyzes the spatially-aware sequence
        self.lstm = nn.LSTM(gat_out_dim, lstm_hidden, batch_first=True, bidirectional=True)

        # 4. INFLECTION POINT: Finds the critical moment of failure
        self.attention = TemporalAttention(hidden_size=lstm_hidden * 2)

        # Classification Head
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden * 2, lstm_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, 1),
            nn.Sigmoid()  # Required for BCELoss and Focal Loss downstream
        )

    def fusion_weights(self):
        """Return explicit complementary A/B fusion coefficients."""
        if self.fusion_a_projection is None:
            return []
        return [1.0 - self.fusion_b_weight, self.fusion_b_weight]

    def forward(self, x_seq, edge_index):
        # A-only input: (Batch, Nodes, A_channels, seq_len).
        # Hybrid input: (Batch, Nodes, A_channels + B_channels, seq_len).
        b_size, n_nodes, input_channels, s_len = x_seq.shape

        if self.fusion_a_projection is not None:
            expected = self.fusion_a_channels + self.fusion_b_channels
            if input_channels != expected:
                raise ValueError(
                    f"Hybrid fusion expected {expected} input channels "
                    f"(A={self.fusion_a_channels}, B={self.fusion_b_channels}), "
                    f"received {input_channels}."
                )
            a = x_seq[:, :, :self.fusion_a_channels, :].reshape(
                b_size * n_nodes, self.fusion_a_channels, s_len
            )
            b = x_seq[:, :, self.fusion_a_channels:, :].reshape(
                b_size * n_nodes, self.fusion_b_channels, s_len
            )
            h_a = self.fusion_a_projection(a)
            h_b = self.fusion_b_projection(b)
            w_b = self.fusion_b_weight
            x_seq = ((1.0 - w_b) * h_a + w_b * h_b).reshape(
                b_size, n_nodes, int(h_a.shape[1]), s_len
            )
            in_c = int(h_a.shape[1])
        else:
            in_c = input_channels
            if in_c != self.conv1.in_channels:
                raise ValueError(
                    f"Model expects {self.conv1.in_channels} channels, received {in_c}."
                )

        # --- Step 1: Temporal Convolution ---
        # Temporarily merge Batch and Nodes so Conv1D treats them as independent series
        x_flat = x_seq.reshape(b_size * n_nodes, in_c, s_len)
        x_conv = F.relu(self.bn(self.conv1(x_flat)))  # (B*N, conv_out, seq_len)
        x_conv = x_conv.permute(0, 2, 1)  # (B*N, seq_len, conv_out)

        # Unfold back to 4D
        x_conv = x_conv.view(b_size, n_nodes, s_len, -1)  # (Batch, Nodes, seq_len, conv_out)

        # --- Step 2: Graph Message Passing ---
        # Process the GAT independently for each batch and each time-step
        gat_features = []
        for b in range(b_size):
            node_seq = []
            for t in range(s_len):
                # x_conv[b, :, t, :] shape is [Nodes, conv_out]
                gat_t = self.gat(x_conv[b, :, t, :], edge_index)
                gat_t = F.elu(gat_t)
                node_seq.append(gat_t)
            gat_features.append(torch.stack(node_seq, dim=1))  # (Nodes, seq_len, gat_out_dim)

        x_spatial = torch.stack(gat_features, dim=0)  # (Batch, Nodes, seq_len, gat_out_dim)

        # --- Step 3 & 4: LSTM & Attention ---
        # Merge Batch and Nodes to process temporal momentum
        x_spatial_flat = x_spatial.view(b_size * n_nodes, s_len, -1)
        lstm_out, _ = self.lstm(x_spatial_flat)
        context, _ = self.attention(lstm_out)

        # Fully connected layer outputs (Batch * Nodes, 1)
        out = self.fc(context)

        # Reshape to strictly output (Batch, Nodes) to match the target labels
        return out.view(b_size, n_nodes)


from src.qkn_interface import QKNBackend


class QuantumKernelNetwork(QKNBackend):
    """
    Acts as a Quantum Feature Extractor.
    Dynamically swaps topologies and depth based on Optuna's Bayesian suggestions.
    """

    def __init__(self, n_qubits=3, layers=2, entangling_type="StronglyEntangling"):
        self.n_qubits = n_qubits
        self.layers = layers
        self.entangling_type = entangling_type

        # C++ Lightning simulator for massive speedup during Bayesian evaluation
        self.dev = qml.device("lightning.qubit", wires=self.n_qubits)

        np.random.seed(42)

        if self.entangling_type == "StronglyEntangling":
            self.weights = pnp.array(np.random.randn(self.layers, self.n_qubits, 3), requires_grad=False)
        else:
            self.weights = pnp.array(np.random.randn(self.layers, self.n_qubits), requires_grad=False)

        @qml.qnode(self.dev, interface="autograd")
        def quantum_feature_map(inputs, weights):
            padded_inputs = pnp.zeros(self.n_qubits)
            for i in range(min(len(inputs), self.n_qubits)):
                padded_inputs[i] = inputs[i]

            qml.AngleEmbedding(padded_inputs, wires=range(self.n_qubits))

            if self.entangling_type == "StronglyEntangling":
                qml.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))
            else:
                qml.BasicEntanglerLayers(weights, wires=range(self.n_qubits))

            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        self.qnode = quantum_feature_map

    def extract_temporal_quantum_features(self, X_seq, progress_callback=None):
        """
        Extract quantum features, optionally reporting fractional progress via
        progress_callback(completed_fraction). The callback does not affect outputs.
        """
        if len(X_seq.shape) == 4:
            n_samples, n_nodes, n_steps, n_features = X_seq.shape
            extracted = np.zeros((n_samples, n_nodes, self.n_qubits, n_steps))
            total = n_samples * n_nodes * n_steps
            completed = 0
            for i in range(n_samples):
                for n in range(n_nodes):
                    for t in range(n_steps):
                        exp_vals = self.qnode(X_seq[i, n, t, :], self.weights)
                        extracted[i, n, :, t] = exp_vals
                        completed += 1
                        if progress_callback is not None and (completed == total or completed % max(1, total // 100) == 0):
                            progress_callback(completed / total)
            return torch.tensor(extracted, dtype=torch.float32)

        elif len(X_seq.shape) == 3:
            n_samples, n_steps, n_features = X_seq.shape
            extracted = np.zeros((n_samples, self.n_qubits, n_steps))
            total = n_samples * n_steps
            completed = 0
            for i in range(n_samples):
                for t in range(n_steps):
                    exp_vals = self.qnode(X_seq[i, t, :], self.weights)
                    extracted[i, :, t] = exp_vals
                    completed += 1
                    if progress_callback is not None and (completed == total or completed % max(1, total // 100) == 0):
                        progress_callback(completed / total)
            return torch.tensor(extracted, dtype=torch.float32)
        raise ValueError("Expected a 3D or 4D input tensor for quantum feature extraction.")