from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier

import numpy as np


class FeatureService:

    def compute_importance(
        self,
        df,
        feature_cols
    ):

        rf = RandomForestClassifier(
            n_estimators=150,
            random_state=42
        )

        rf.fit(
            df[feature_cols],
            df["Failure_Label"]
        )

        return rf.feature_importances_

    def prepare_quantum_features(
        self,
        df,
        selected_features,
        feature_cols_all
    ):

        scaler = MinMaxScaler(
            feature_range=(-np.pi, np.pi)
        )

        X_raw = df[selected_features].values

        y = df["Failure_Label"].values

        bus_ids = df["Bus_ID"].values

        X = scaler.fit_transform(X_raw)

        seq_raw = np.array(
            df["Sequence"].tolist()
        )

        selected_indices = [
            feature_cols_all.index(f)
            for f in selected_features
        ]

        seq_filtered = seq_raw[:, :, selected_indices]

        seq_flat = seq_filtered.reshape(
            -1,
            len(selected_features)
        )

        seq_scaled = scaler.transform(
            seq_flat
        )

        X_seq = seq_scaled.reshape(
            -1,
            4,
            len(selected_features)
        )

        return (
            X,
            X_seq,
            y,
            bus_ids,
            scaler
        )