import streamlit as st
import pandas as pd


def render_training_dashboard():

    if not st.session_state.get(
        "pytorch_qcnn_active",
        False
    ):
        return

    st.divider()

    st.markdown(
        "### ⚛️ Training History"
    )

    history = (
        st.session_state.training_history
    )

    df = pd.DataFrame(
        {
            "Loss": history
        }
    )

    st.line_chart(df)