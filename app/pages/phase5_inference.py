import streamlit as st
import pandas as pd

from app.services.inference_service import (
    InferenceService
)

from app.visualizations.distributions import (
    prediction_distribution
)


def render_phase5():

    st.markdown(
        "### 🧪 Phase 5: Inference Testing"
    )

    if not st.session_state.get(
        "qcp_calibrated",
        False
    ):
        st.error(
            "Run Phase 4 first."
        )
        return

    if st.button(
        "Run Inference",
        type="primary"
    ):

        service = InferenceService()

        q_config = {
            "qubits": st.session_state.active_q_qubits,
            "q_type": st.session_state.active_q_type,
            "layers": st.session_state.active_q_depth
        }

        probabilities = service.predict(
            st.session_state.pytorch_model,
            st.session_state.X_seq_test,
            q_config
        )

        prediction_sets = (
            service.conformal_sets(
                st.session_state.qcp_model,
                probabilities
            )
        )

        st.session_state.test_probs = (
            probabilities
        )

        st.session_state.prediction_sets = (
            prediction_sets
        )

        st.success(
            "Inference Complete"
        )

    if "test_probs" in st.session_state:

        labels = [
            "Failure"
            if x == 1
            else "Safe"
            for x in st.session_state.y_test
        ]

        st.plotly_chart(
            prediction_distribution(
                st.session_state.test_probs,
                labels
            ),
            use_container_width=True
        )

        results = pd.DataFrame({
            "Bus": st.session_state.bus_test,
            "Probability": st.session_state.test_probs,
            "Prediction_Set": [
                str(x)
                for x in st.session_state.prediction_sets
            ]
        })

        st.dataframe(
            results,
            use_container_width=True
        )