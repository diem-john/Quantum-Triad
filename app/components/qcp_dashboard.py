import streamlit as st

from app.visualizations.distributions import (
    calibration_distribution,
    add_qhat_threshold
)


def render_qcp_dashboard():

    if not st.session_state.get(
        "qcp_calibrated",
        False
    ):
        return

    fig = calibration_distribution(
        st.session_state.cal_scores,
        st.session_state.y_cal
    )

    fig = add_qhat_threshold(
        fig,
        st.session_state.q_hat
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.success(
        f"q_hat = "
        f"{st.session_state.q_hat:.4f}"
    )