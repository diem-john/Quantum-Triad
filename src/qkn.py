import pennylane as qml
from pennylane import numpy as pnp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Original
# class QuantumKernelNetwork:
#     """
#     Acts as a Quantum Feature Extractor.
#     Maps classical data into a higher-dimensional Hilbert space using
#     Angle Embedding and Strongly Entangling Layers, returning Pauli-Z expectations.
#     """
#
#     def __init__(self, n_qubits=3, layers=2):
#         self.n_qubits = n_qubits
#         self.layers = layers
#         self.dev = qml.device("default.qubit", wires=self.n_qubits)
#
#         # Initialize fixed random weights for the entanglement ansatz
#         # (This acts as a deterministic, non-linear quantum kernel projection)
#         np.random.seed(42)
#         self.weights = pnp.array(np.random.randn(self.layers, self.n_qubits, 3), requires_grad=False)
#
#         @qml.qnode(self.dev, interface="autograd")
#         def quantum_feature_map(inputs, weights):
#             # Pad inputs to match qubit count if necessary
#             padded_inputs = pnp.zeros(self.n_qubits)
#             for i in range(min(len(inputs), self.n_qubits)):
#                 padded_inputs[i] = inputs[i]
#
#             # Embed classical features into quantum phases
#             qml.AngleEmbedding(padded_inputs, wires=range(self.n_qubits))
#
#             # Create highly entangled feature topology
#             qml.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))
#
#             # Measure expectation value of each qubit
#             return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
#
#         self.qnode = quantum_feature_map
#
#     def extract_temporal_quantum_features(self, X_seq):
#         """
#         Runs the quantum circuit iteratively over a 3D sequence tensor.
#         Input: (Samples, Time_Steps, Features)
#         Output: PyTorch Tensor (Samples, Qubits, Time_Steps) -> Ready for Conv1D
#         """
#         n_samples, n_steps, n_features = X_seq.shape
#         # Shape optimized for PyTorch Conv1D input (Batch, Channels, Length)
#         extracted = np.zeros((n_samples, self.n_qubits, n_steps))
#
#         for i in range(n_samples):
#             for t in range(n_steps):
#                 # Run the circuit for this specific time step
#                 exp_vals = self.qnode(X_seq[i, t, :], self.weights)
#                 extracted[i, :, t] = exp_vals
#
#         return torch.tensor(extracted, dtype=torch.float32)


class TemporalAttention(nn.Module):
    def __init__(self, hidden_size):
        super(TemporalAttention, self).__init__()
        self.attention_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, lstm_output):
        attention_scores = self.attention_layer(lstm_output)
        attention_weights = F.softmax(attention_scores, dim=1)
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)
        return context_vector, attention_weights


class QuantumTemporalConvNet(nn.Module):
    """
    Ingests the Quantum Feature Embeddings into a Deep Learning Pipeline.
    Architecture: Conv1D -> BiLSTM -> Temporal Attention -> Linear
    """

    def __init__(self, in_channels, sequence_length=4):
        super(QuantumTemporalConvNet, self).__init__()

        # Spatial/Local Feature Extraction from Quantum Outputs
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(8)
        self.relu1 = nn.ReLU()

        # Sequential Modeling
        self.lstm_hidden = 8
        self.lstm = nn.LSTM(
            input_size=8, hidden_size=self.lstm_hidden,
            num_layers=1, batch_first=True, bidirectional=True
        )

        # Temporal Attention (Finds the critical inflection point in the storm)
        self.attention = TemporalAttention(hidden_size=self.lstm_hidden * 2)

        # Classification Head
        self.fc1 = nn.Linear(self.lstm_hidden * 2, 8)
        self.fc_relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(8, 1)  # Returns logits

    def forward(self, x):
        # x shape: (Batch, Qubits, Time_Steps)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        # Permute for LSTM: (Batch, Time_Steps, Channels)
        x = x.permute(0, 2, 1)

        lstm_out, _ = self.lstm(x)
        context_vector, _ = self.attention(lstm_out)

        x = self.fc1(context_vector)
        x = self.fc_relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        return (self.alpha * (1 - pt) ** self.gamma * BCE_loss).mean()


# class ScaledQuantumTemporalConvNet(nn.Module):
#     def __init__(self, in_channels, conv_dim=32, lstm_hidden=64, lstm_layers=2):
#         super(ScaledQuantumTemporalConvNet, self).__init__()
#
#         # LayerNorm right after Quantum Embedding prevents input saturation
#         self.input_norm = nn.LayerNorm([in_channels, 4])
#
#         self.conv_block = nn.Sequential(
#             nn.Conv1d(in_channels, conv_dim, kernel_size=3, padding=1),
#             nn.BatchNorm1d(conv_dim),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Conv1d(conv_dim, conv_dim * 2, kernel_size=3, padding=1),
#             nn.BatchNorm1d(conv_dim * 2),
#             nn.ReLU()
#         )
#
#         self.lstm = nn.LSTM(
#             input_size=conv_dim * 2,
#             hidden_size=lstm_hidden,
#             num_layers=lstm_layers,
#             batch_first=True,
#             bidirectional=True,
#             dropout=0.3
#         )
#
#         self.attention = TemporalAttention(hidden_size=lstm_hidden * 2)
#
#         self.fc_block = nn.Sequential(
#             nn.Linear(lstm_hidden * 2, 128),
#             nn.ReLU(),
#             nn.Dropout(0.4),
#             nn.Linear(128, 1)
#         )
#
#     def forward(self, x):
#         x = self.input_norm(x)
#         x = self.conv_block(x)
#         x = x.permute(0, 2, 1)
#         lstm_out, _ = self.lstm(x)
#         context, _ = self.attention(lstm_out)
#         return self.fc_block(context)

# ORIGINAL!!!
# class ScaledQuantumTemporalConvNet(nn.Module):
#     def __init__(self, in_channels=4, seq_len=4):  # explicitly set in_channels=4
#         super(ScaledQuantumTemporalConvNet, self).__init__()
#
#         # Reduced dimension: 4 features don't need 32 channels. 16 is enough.
#         self.conv1 = nn.Conv1d(in_channels, 16, kernel_size=3, padding=1)
#         self.bn = nn.BatchNorm1d(16)
#
#         # Single-layer LSTM is plenty for a 4-step sequence
#         self.lstm = nn.LSTM(16, 32, batch_first=True, bidirectional=True)
#
#         self.attention = TemporalAttention(hidden_size=64)
#
#         # Simpler head to prevent overfitting
#         self.fc = nn.Sequential(
#             nn.Linear(64, 32),
#             nn.ReLU(),
#             nn.Linear(32, 1)
#         )
#
#     def forward(self, x):
#         # x: (Batch, 4, 4)
#         x = F.relu(self.bn(self.conv1(x)))
#         x = x.permute(0, 2, 1)  # (Batch, 4, 16)
#         lstm_out, _ = self.lstm(x)
#         context, _ = self.attention(lstm_out)
#         return self.fc(context)

class ScaledQuantumTemporalConvNet(nn.Module):
    def __init__(self, in_channels=4, seq_len=4, conv_out=16, lstm_hidden=32):
        super(ScaledQuantumTemporalConvNet, self).__init__()

        # Dynamic Dimensions tuned by PSO
        self.conv1 = nn.Conv1d(in_channels, conv_out, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(conv_out)

        self.lstm = nn.LSTM(conv_out, lstm_hidden, batch_first=True, bidirectional=True)

        # Temporal Attention hidden size is 2x lstm_hidden due to bidirectional
        self.attention = TemporalAttention(hidden_size=lstm_hidden * 2)

        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden * 2, lstm_hidden),
            nn.ReLU(),
            nn.Linear(lstm_hidden, 1)
        )

    def forward(self, x):
        # x: (Batch, in_channels, seq_len)
        x = F.relu(self.bn(self.conv1(x)))
        x = x.permute(0, 2, 1)  # (Batch, seq_len, conv_out)
        lstm_out, _ = self.lstm(x)
        context, _ = self.attention(lstm_out)
        return self.fc(context)


class QuantumKernelNetwork:
    """
    Acts as a Quantum Feature Extractor.
    Dynamically swaps between Strong and Basic Entanglement topologies based on PSO optimization.
    """

    def __init__(self, n_qubits=3, layers=2, entangling_type="StronglyEntangling"):
        self.n_qubits = n_qubits
        self.layers = layers
        self.entangling_type = entangling_type
        self.dev = qml.device("default.qubit", wires=self.n_qubits)

        np.random.seed(42)
        # Strong Entangling needs 3 weights per qubit, Basic only needs 1
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

            # Apply optimized entanglement topology
            if self.entangling_type == "StronglyEntangling":
                qml.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))
            else:
                qml.BasicEntanglerLayers(weights, wires=range(self.n_qubits))

            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        self.qnode = quantum_feature_map

    def extract_temporal_quantum_features(self, X_seq):
        n_samples, n_steps, n_features = X_seq.shape
        extracted = np.zeros((n_samples, self.n_qubits, n_steps))
        for i in range(n_samples):
            for t in range(n_steps):
                exp_vals = self.qnode(X_seq[i, t, :], self.weights)
                extracted[i, :, t] = exp_vals
        return torch.tensor(extracted, dtype=torch.float32)