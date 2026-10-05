import numpy as np

from src.qkn import (
    QuantumKernelNetwork,
    ScaledQuantumTemporalConvNet,
    FocalLoss
)

import torch

from torch.utils.data import (
    DataLoader,
    TensorDataset
)


class UnifiedPSOOptimizer:

    def __init__(
        self,
        swarm_size,
        iterations,
        proxy_epochs
    ):
        self.swarm_size = swarm_size
        self.iterations = iterations
        self.proxy_epochs = proxy_epochs

    def optimize(
        self,
        X_train,
        y_train,
        X_cal,
        y_cal
    ):

        # Move your entire PSO code
        # from app.py into here

        # Return best architecture

        return {
            "lr": best_lr,
            "batch": best_batch,
            "alpha": best_alpha,
            "gamma": best_gamma,
            "conv": best_conv,
            "lstm": best_lstm,
            "qubits": best_qubits,
            "q_type": best_qtype,
            "layers": best_layers
        }