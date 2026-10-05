import streamlit as st
import pandas as pd

from sklearn.model_selection import (
    train_test_split
)

from app.services.feature_service import FeatureService

from app.visualizations.dashboards import (
    correlation_heatmap,
    feature_importance_chart
)


def render_feature_selection():

    if not st.session_state.get(
        "coresets_generated",
        False
    ):
        return

    st.divider()

    st.markdown(
        "### 🎯 Information Bottleneck"
    )

    df = st.session_state.df_final_distilled

    feature_cols = [
        "Wind_Speed",
        "Storm_Grade",
        "Distance_to_Eye",
        "Coastal_Exposure"
    ]

    service = FeatureService()

    importance = service.compute_importance(
        df,
        feature_cols
    )

    corr_matrix = (
        df[
            feature_cols +
            ["Failure_Label"]
        ]
        .corr()
    )

    col1, col2 = st.columns(2)

    with col1:

        st.plotly_chart(
            correlation_heatmap(
                corr_matrix
            ),
            use_container_width=True
        )

    with col2:

        st.plotly_chart(
            feature_importance_chart(
                feature_cols,
                importance
            ),
            use_container_width=True
        )

    sorted_features = [
        x
        for _, x in sorted(
            zip(
                importance,
                feature_cols
            ),
            reverse=True
        )
    ]

    st.markdown(
        "### Select Embedding Features"
    )

    selected = []

    cols = st.columns(
        len(sorted_features)
    )

    for i, feat in enumerate(
        sorted_features
    ):

        with cols[i]:

            if st.checkbox(
                feat,
                value=(i < 3),
                key=f"feature_{feat}"
            ):
                selected.append(feat)

    st.session_state.selected_features = (
        selected
    )

    st.info(
        f"Selected Features ({len(selected)}): "
        + ", ".join(selected)
    )

    st.divider()

    if st.button(
        "🚀 Finalize Features & Prepare Quantum Dataset",
        type="primary"
    ):

        if len(selected) < 2:

            st.error(
                "Please select at least 2 features."
            )

            return

        with st.spinner(
            "Preparing quantum-ready dataset..."
        ):

            (
                X,
                X_seq,
                y,
                bus_ids,
                scaler
            ) = service.prepare_quantum_features(
                df,
                selected,
                feature_cols
            )

            (
                X_train,
                X_temp,
                X_seq_train,
                X_seq_temp,
                y_train,
                y_temp,
                bus_train,
                bus_temp
            ) = train_test_split(
                X,
                X_seq,
                y,
                bus_ids,
                test_size=0.40,
                random_state=42,
                stratify=y
            )

            (
                X_cal,
                X_test,
                X_seq_cal,
                X_seq_test,
                y_cal,
                y_test,
                bus_cal,
                bus_test
            ) = train_test_split(
                X_temp,
                X_seq_temp,
                y_temp,
                bus_temp,
                test_size=0.50,
                random_state=42,
                stratify=y_temp
            )

            st.session_state.X_train = X_train
            st.session_state.X_cal = X_cal
            st.session_state.X_test = X_test

            st.session_state.X_seq_train = X_seq_train
            st.session_state.X_seq_cal = X_seq_cal
            st.session_state.X_seq_test = X_seq_test

            st.session_state.y_train = y_train
            st.session_state.y_cal = y_cal
            st.session_state.y_test = y_test

            st.session_state.bus_train = bus_train
            st.session_state.bus_cal = bus_cal
            st.session_state.bus_test = bus_test

            st.session_state.scaler = scaler

            st.session_state.feature_cols = (
                selected
            )

            st.session_state.data_loaded = True

        st.success(
            "Quantum dataset successfully prepared."
        )

        summary = pd.DataFrame({
            "Partition": [
                "Training",
                "Calibration",
                "Testing"
            ],
            "Samples": [
                len(X_train),
                len(X_cal),
                len(X_test)
            ]
        })

        st.dataframe(
            summary,
            use_container_width=True
        )

        st.balloons()