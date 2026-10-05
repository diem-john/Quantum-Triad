import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin

from src.utils import (
    haversine,
    rankine_vortex,
    vulnerability_curve
)


class DatasetService:

    def build_quantum_dataset(
        self,
        csv_path,
        ids,
        target_qpu_budget,
        time_steps=4
    ):

        df = pd.read_csv(csv_path)

        df_tw = df[
            (df["lat"] >= 21) &
            (df["lat"] <= 26) &
            (df["lng"] >= 118) &
            (df["lng"] <= 123)
        ].copy()

        taiwan_seq_ids = df_tw["seq_id"].unique()[ids:]

        df_recent_tw = df_tw[
            df_tw["seq_id"].isin(taiwan_seq_ids)
        ]

        top_storm_ids = (
            df_recent_tw.groupby("seq_id")["wind"]
            .max()
            .nlargest(5)
            .index
        )

        df_events = df_recent_tw[
            df_recent_tw["seq_id"].isin(top_storm_ids)
        ]

        records = self._generate_records(
            df_events,
            time_steps
        )

        df_mapped = pd.DataFrame(records)

        df_clean = df_mapped[
            (df_mapped["Raw_Prob"] <= 0.40)
            |
            (df_mapped["Raw_Prob"] >= 0.60)
        ]

        return self._create_coresets(
            df_clean,
            target_qpu_budget
        )

    def _generate_records(
        self,
        df_events,
        time_steps
    ):

        chiayi_lat = 23.48
        chiayi_lng = 120.44
        coastline_lng = 120.15

        np.random.seed(42)

        bus_coords = {
            i: (
                chiayi_lat + np.random.uniform(-0.1, 0.1),
                chiayi_lng + np.random.uniform(-0.1, 0.1)
            )
            for i in range(1, 34)
        }

        records = []

        for seq_id, storm_track in df_events.groupby("seq_id"):

            bus_failed = {
                i: 0 for i in range(1, 34)
            }

            history = {
                i: [] for i in range(1, 34)
            }

            for _, row in storm_track.iterrows():

                ty_lat = row["lat"]
                ty_lng = row["lng"]

                v_max = row["wind"] * 0.51444

                storm_grade = row["grade"]

                for bus_id in range(1, 34):

                    bus_lat, bus_lng = bus_coords[bus_id]

                    dist_km = haversine(
                        ty_lat,
                        ty_lng,
                        bus_lat,
                        bus_lng
                    )

                    wind = rankine_vortex(
                        v_max,
                        dist_km
                    )

                    coastal = max(
                        0,
                        1 - (bus_lng - coastline_lng)
                    )

                    history[bus_id].append([
                        wind,
                        storm_grade,
                        dist_km,
                        coastal
                    ])

                    if len(history[bus_id]) >= time_steps:

                        window = history[bus_id][-time_steps:]

                        sustained = np.mean(
                            [x[0] for x in window]
                        )

                        if bus_failed[bus_id]:

                            label = 1
                            prob = 1.0

                        else:

                            prob = vulnerability_curve(
                                sustained
                            )

                            label = (
                                1
                                if prob >= 0.60
                                else 0
                            )

                            if label:
                                bus_failed[bus_id] = 1

                        records.append({
                            "Bus_ID": bus_id,
                            "Storm_ID": seq_id,
                            "Wind_Speed": sustained,
                            "Storm_Grade": storm_grade,
                            "Distance_to_Eye": dist_km,
                            "Coastal_Exposure": coastal,
                            "Failure_Label": label,
                            "Raw_Prob": prob,
                            "Sequence": window
                        })

        return records

    def _create_coresets(
        self,
        df_clean,
        target_budget
    ):

        feature_cols = [
            "Wind_Speed",
            "Storm_Grade",
            "Distance_to_Eye",
            "Coastal_Exposure"
        ]

        if len(df_clean) <= target_budget:
            return df_clean

        df_fails = df_clean[
            df_clean["Failure_Label"] == 1
        ]

        df_safe = df_clean[
            df_clean["Failure_Label"] == 0
        ]

        k_fail = min(
            len(df_fails),
            target_budget // 2
        )

        k_safe = target_budget - k_fail

        km_safe = KMeans(
            n_clusters=k_safe,
            random_state=42,
            n_init="auto"
        )

        km_safe.fit(df_safe[feature_cols])

        safe_idx = pairwise_distances_argmin(
            km_safe.cluster_centers_,
            df_safe[feature_cols]
        )

        km_fail = KMeans(
            n_clusters=k_fail,
            random_state=42,
            n_init="auto"
        )

        km_fail.fit(df_fails[feature_cols])

        fail_idx = pairwise_distances_argmin(
            km_fail.cluster_centers_,
            df_fails[feature_cols]
        )

        return pd.concat([
            df_safe.iloc[safe_idx],
            df_fails.iloc[fail_idx]
        ])