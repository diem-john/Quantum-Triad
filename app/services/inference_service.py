import torch
import numpy as np

from src.qkn import (
    QuantumKernelNetwork
)


class InferenceService:

    def predict(
        self,
        model,
        X_seq_test,
        q_config
    ):

        qkn = QuantumKernelNetwork(
            n_qubits=q_config["qubits"],
            layers=q_config["layers"],
            entangling_type=q_config["q_type"]
        )

        X_test_tensor = (
            qkn.extract_temporal_quantum_features(
                X_seq_test
            )
        )

        model.eval()

        with torch.no_grad():

            probs = torch.sigmoid(
                model(X_test_tensor)
            )

        return probs.numpy().flatten()

    def conformal_sets(
        self,
        qcp_model,
        probabilities
    ):

        p1 = probabilities

        p0 = 1 - p1

        probs = np.column_stack(
            [p0, p1]
        )

        return qcp_model.predict_sets(
            test_probs=probs
        )