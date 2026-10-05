import numpy as np
import torch

from src.qcp import (
    QuantumConformalPredictor
)


class CalibrationService:

    def calibrate(
        self,
        model,
        X_cal_tensor,
        y_cal,
        target_coverage
    ):

        model.eval()

        with torch.no_grad():

            logits = model(
                X_cal_tensor
            )

            p1 = torch.sigmoid(
                logits
            ).numpy().flatten()

        p0 = 1 - p1

        probs = np.column_stack(
            [p0, p1]
        )

        qcp = QuantumConformalPredictor(
            alpha=(1 - target_coverage)
        )

        q_hat = qcp.calibrate(
            y_cal=y_cal,
            cal_probs=probs
        )

        return (
            qcp,
            q_hat,
            probs
        )