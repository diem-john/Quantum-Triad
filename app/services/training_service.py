import torch

from torch.utils.data import (
    DataLoader,
    TensorDataset
)

from src.qkn import (
    QuantumKernelNetwork,
    ScaledQuantumTemporalConvNet,
    FocalLoss
)


class HybridTrainer:

    def train(
        self,
        X_seq_train,
        y_train,
        X_seq_cal,
        y_cal,
        config
    ):

        qkn = QuantumKernelNetwork(
            n_qubits=config["qubits"],
            layers=config["layers"],
            entangling_type=config["q_type"]
        )

        X_train_tensor = (
            qkn.extract_temporal_quantum_features(
                X_seq_train
            )
        )

        X_val_tensor = (
            qkn.extract_temporal_quantum_features(
                X_seq_cal
            )
        )

        y_train = torch.tensor(
            y_train,
            dtype=torch.float32
        ).unsqueeze(1)

        y_cal = torch.tensor(
            y_cal,
            dtype=torch.float32
        ).unsqueeze(1)

        model = ScaledQuantumTemporalConvNet(
            in_channels=config["qubits"],
            conv_out=config["conv"],
            lstm_hidden=config["lstm"]
        )

        criterion = FocalLoss(
            alpha=config["alpha"],
            gamma=config["gamma"]
        )

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config["lr"]
        )

        train_loader = DataLoader(
            TensorDataset(
                X_train_tensor,
                y_train
            ),
            batch_size=config["batch"],
            shuffle=True
        )

        history = []

        for epoch in range(
            config["epochs"]
        ):

            model.train()

            total_loss = 0

            for bx, by in train_loader:

                optimizer.zero_grad()

                loss = criterion(
                    model(bx),
                    by
                )

                loss.backward()

                optimizer.step()

                total_loss += loss.item()

            history.append(total_loss)

        return model, history