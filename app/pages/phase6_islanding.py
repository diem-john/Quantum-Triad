import streamlit as st

from src.mapping import (
    QuantumWalkIslandingMapper
)

from app.components.islanding_dashboard import (
    render_islanding_dashboard
)


def render_phase6():

    st.markdown(
        "### 🌊 Phase 6: CTQW Islanding"
    )

    if "failed_edges" not in st.session_state:

        st.info(
            "No failed transmission lines available."
        )

        return

    if st.button(
        "Identify Quantum Islands",
        type="primary"
    ):

        post_grid = (
            st.session_state.mapper
            .simulate_typhoon_failures(
                st.session_state.failed_edges
            )
        )

        mapper = QuantumWalkIslandingMapper(
            post_grid
        )

        zones = mapper.identify_islands()

        st.session_state.islanding_zones = (
            zones
        )

    if "islanding_zones" in st.session_state:

        render_islanding_dashboard(
            st.session_state.islanding_zones
        )