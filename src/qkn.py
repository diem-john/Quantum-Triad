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

    def __init__(self, in_channels=4, seq_len=4, conv_out=16, gat_heads=2, lstm_hidden=32, dropout=0.2):
        super(QuantumSpatiotemporalGNN, self).__init__()

        # 1. TEMPORAL BOTTLENECK: Smooths the Quantum Kernel outputs over time
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

    def forward(self, x_seq, edge_index):
        # x_seq shape: (Batch, Nodes, in_channels, seq_len)
        b_size, n_nodes, in_c, s_len = x_seq.shape

        # --- Step 1: Temporal Convolution ---
        # Temporarily merge Batch and Nodes so Conv1D treats them as independent series
        x_flat = x_seq.view(b_size * n_nodes, in_c, s_len)
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

    def extract_temporal_quantum_features(self, X_seq):
        """
        Handles the 4D Grid Snapshots: (Batch, Nodes, Time_Steps, Features)
        Output: (Batch, Nodes, Qubits, Time_Steps) -> Formatted for Conv1D ingestion
        """
        # Handle 4D Tensor (New Grid-Centric Architecture)
        if len(X_seq.shape) == 4:
            n_samples, n_nodes, n_steps, n_features = X_seq.shape
            extracted = np.zeros((n_samples, n_nodes, self.n_qubits, n_steps))

            for i in range(n_samples):
                for n in range(n_nodes):
                    for t in range(n_steps):
                        exp_vals = self.qnode(X_seq[i, n, t, :], self.weights)
                        extracted[i, n, :, t] = exp_vals

            return torch.tensor(extracted, dtype=torch.float32)

        # Handle 3D Tensor fallback (For backward compatibility)
        elif len(X_seq.shape) == 3:
            n_samples, n_steps, n_features = X_seq.shape
            extracted = np.zeros((n_samples, self.n_qubits, n_steps))
            for i in range(n_samples):
                for t in range(n_steps):
                    exp_vals = self.qnode(X_seq[i, t, :], self.weights)
                    extracted[i, :, t] = exp_vals
            return torch.tensor(extracted, dtype=torch.float32)