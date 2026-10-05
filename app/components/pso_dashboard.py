import streamlit as st
import pandas as pd


def render_pso_results():

    if not st.session_state.get(
        "pso_complete",
        False
    ):
        return

    st.divider()

    st.markdown(
        "### 🐝 Optimization Results"
    )

    params = (
        st.session_state.best_params
    )

    st.json(params)

    metrics = pd.DataFrame(
        [params]
    )

    st.dataframe(
        metrics,
        use_container_width=True
    )