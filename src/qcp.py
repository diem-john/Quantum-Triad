import numpy as np


class QuantumConformalPredictor:
    """Split Conformal Prediction wrapper for the Quantum Kernel Network (Engine Agnostic)."""

    def __init__(self, qkn_model=None, alpha=0.1):
        # qkn_model can be None if probabilities are calculated externally
        self.qkn = qkn_model
        self.alpha = alpha  # 1 - alpha = target coverage (e.g., 90%)
        self.q_hat = None

    def calibrate(self, X_cal=None, y_cal=None, X_train=None, cal_probs=None):
        """
        Calculates the non-conformity scores and the quantile (q_hat).
        Accepts raw probabilities (cal_probs) to support diverse ML backend engines (PyTorch, SVM, etc.).
        """
        # --- ROUTING LOGIC ---
        if cal_probs is None:
            if self.qkn is None or not hasattr(self.qkn, 'svm'):
                raise AttributeError("No cal_probs provided and underlying engine lacks an SVM predict_proba method.")
            if X_cal is None or X_train is None:
                raise ValueError("X_cal and X_train must be provided if cal_probs is None.")

            # Fallback to classical QSVM engine calculation
            K_cal = self.qkn.compute_kernel_matrix(X_cal, X_train)
            cal_probs = self.qkn.svm.predict_proba(K_cal)

        n = len(y_cal)

        # Safety cast to ensure integer indexing doesn't trigger numpy TypeErrors
        y_cal = np.array(y_cal).astype(int)

        # Non-conformity score: 1 minus the predicted probability of the true label
        scores = np.array([1 - cal_probs[i, y_cal[i]] for i in range(n)])

        # Calculate finite-sample correction quantile
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        if q_level > 1.0:
            q_level = 1.0

        self.q_hat = np.quantile(scores, q_level, method='higher')
        return self.q_hat

    def predict_sets(self, X_test=None, X_train=None, classes=[0, 1], test_probs=None):
        """
        Outputs a prediction set for each test instance based on the calibrated q_hat threshold.
        """
        if self.q_hat is None:
            raise ValueError("Model is not calibrated. Please call calibrate() first.")

        # --- ROUTING LOGIC ---
        if test_probs is None:
            if self.qkn is None or not hasattr(self.qkn, 'svm'):
                raise AttributeError("No test_probs provided and underlying engine lacks an SVM predict_proba method.")
            if X_test is None or X_train is None:
                raise ValueError("X_test and X_train must be provided if test_probs is None.")

            # Fallback to classical QSVM engine calculation
            K_test = self.qkn.compute_kernel_matrix(X_test, X_train)
            test_probs = self.qkn.svm.predict_proba(K_test)

        prediction_sets = []
        for probs in test_probs:
            # A class is included in the prediction set if its predicted probability is >= (1 - q_hat)
            valid_classes = [c for c, p in zip(classes, probs) if p >= (1 - self.q_hat)]
            prediction_sets.append(valid_classes)

        return prediction_sets