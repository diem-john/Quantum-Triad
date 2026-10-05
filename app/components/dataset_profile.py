import streamlit as st
import pandas as pd

from app.visualizations.radar import (
    plot_feature_radar
)

from app.visualizations.dashboards import (
    temporal_sequence_dashboard
)


def render_dataset_profile():

    if not st.session_state.get(
        "data_loaded",
        False
    ):
        return

    st.divider()

    st.markdown(
        "### 📊 Dataset Profiles"
    )

    feature_names = (
        st.session_state.feature_cols
    )

    df = pd.DataFrame(
        st.session_state.X_train,
        columns=feature_names
    )

    df["Failure_Label"] = (
        st.session_state.y_train
    )

    col1, col2 = st.columns(
        [2, 1]
    )

    with col1:

        st.dataframe(
            df,
            use_container_width=True
        )

    with col2:

        sample_idx = st.selectbox(
            "Sample",
            range(len(df))
        )

        sample = (
            st.session_state.X_train[
                sample_idx
            ]
        )

        label = (
            st.session_state.y_train[
                sample_idx
            ]
        )

        st.plotly_chart(
            plot_feature_radar(
                sample,
                feature_names,
                label
            ),
            use_container_width=True
        )

    if "X_seq_train" in st.session_state:

        st.plotly_chart(
            temporal_sequence_dashboard(
                st.session_state.X_seq_train[
                    sample_idx
                ],
                feature_names
            ),
            use_container_width=True
        )