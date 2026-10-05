import streamlit as st

from app.services.dataset_service import DatasetService
from app.services.feature_service import FeatureService

from app.visualizations.maps import plot_microgrid_map

from app.components.feature_selection import (
    render_feature_selection
)

from app.components.dataset_profile import (
    render_dataset_profile
)


def render_phase1():

    st.markdown(
        "### 📍 Phase 1: Geo-Extraction"
    )

    mapper = st.session_state.mapper

    col1, col2 = st.columns([2, 1])

    with col1:

        st.plotly_chart(
            plot_microgrid_map(
                mapper,
                "Enhanced IEEE 33-Bus System in Chiayi"
            ),
            use_container_width=True
        )

    with col2:

        st.info(
            "Generate temporal quantum coresets from historical typhoon data."
        )

        ids = st.slider(
            "How Many Typhoons Are We Considering",
            0,
            100,
            3
        )

        target_budget = st.slider(
            "QPU Budget",
            50,
            1200,
            400
        )

        if st.button(
            "Extract Historical Typhoon Data"
        ):

            service = DatasetService()

            with st.spinner(
                "Generating temporal coresets..."
            ):

                df_final = service.build_quantum_dataset(
                    csv_path="data/raw/typhoon_data.csv",
                    ids=ids,
                    target_qpu_budget=target_budget
                )

            st.session_state.df_final_distilled = (
                df_final
            )

            st.session_state.coresets_generated = True

            st.success(
                f"{len(df_final)} coresets generated."
            )

    render_feature_selection()

    render_dataset_profile()