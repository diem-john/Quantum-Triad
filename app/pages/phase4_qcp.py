import streamlit as st

from app.services.qcp_service import (
    CalibrationService
)

from app.components.qcp_dashboard import (
    render_qcp_dashboard
)


def render_phase4():

    st.markdown(
        "### 🛡️ Phase 4: Uncertainty Calibration"
    )

    if not st.session_state.get(
        "pytorch_qcnn_active",
        False
    ):
        st.error(
            "Train the model first."
        )
        return

    coverage = st.slider(
        "Target Coverage (%)",
        80,
        99,
        90
    ) / 100

    if st.button(
        "Calculate q_hat",
        type="primary"
    ):

        service = CalibrationService()

        qcp_model, q_hat, probs = (
            service.calibrate(
                st.session_state.pytorch_model,
                st.session_state.X_cal_tensor,
                st.session_state.y_cal,
                coverage
            )
        )

        st.session_state.qcp_model = (
            qcp_model
        )

        st.session_state.q_hat = q_hat

        st.session_state.qcp_calibrated = True

        st.session_state.cal_probs = probs

    render_qcp_dashboard()