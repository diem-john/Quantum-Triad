import streamlit as st

from app.services.pso_service import (
    UnifiedPSOOptimizer
)

from app.components.pso_dashboard import (
    render_pso_results
)


def render_phase2():

    st.markdown(
        "### 🐝 Global Particle Swarm Optimization"
    )

    if not st.session_state.get(
        "data_loaded",
        False
    ):
        st.error(
            "Please complete Phase 1 first."
        )
        return

    col1, col2, col3 = st.columns(3)

    with col1:

        swarm_size = st.number_input(
            "Swarm Size",
            3,
            20,
            5
        )

    with col2:

        iterations = st.number_input(
            "Iterations",
            2,
            100,
            5
        )

    with col3:

        proxy_epochs = st.number_input(
            "Proxy Epochs",
            3,
            50,
            10
        )

    if st.button(
        "Launch Optimization",
        type="primary"
    ):

        optimizer = UnifiedPSOOptimizer(
            swarm_size,
            iterations,
            proxy_epochs
        )

        best_params = optimizer.optimize(
            st.session_state.X_seq_train,
            st.session_state.y_train,
            st.session_state.X_seq_cal,
            st.session_state.y_cal
        )

        st.session_state.best_params = (
            best_params
        )

        st.session_state.pso_complete = True

        st.success(
            "Optimization Complete"
        )

    render_pso_results()