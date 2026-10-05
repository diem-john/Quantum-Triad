import streamlit as st

from app.services.training_service import (
    HybridTrainer
)

from app.components.training_dashboard import (
    render_training_dashboard
)


def render_phase3():

    st.markdown(
        "### ⚛️ Phase 3: Quantum Training"
    )

    if not st.session_state.get(
        "data_loaded",
        False
    ):
        st.error(
            "Complete Phase 1 first."
        )
        return

    epochs = st.number_input(
        "Training Epochs",
        50,
        1000,
        200
    )

    if st.button(
        "Train Final Pipeline",
        type="primary"
    ):

        config = (
            st.session_state.best_params
            if st.session_state.get(
                "pso_complete",
                False
            )
            else {
                "lr": 0.005,
                "batch": 32,
                "alpha": 0.60,
                "gamma": 2.0,
                "conv": 16,
                "lstm": 32,
                "qubits": 3,
                "q_type": "StronglyEntangling",
                "layers": 2
            }
        )

        config["epochs"] = epochs

        trainer = HybridTrainer()

        model, history = trainer.train(
            st.session_state.X_seq_train,
            st.session_state.y_train,
            st.session_state.X_seq_cal,
            st.session_state.y_cal,
            config
        )

        st.session_state.pytorch_model = model

        st.session_state.training_history = (
            history
        )

        st.session_state.pytorch_qcnn_active = (
            True
        )

        st.session_state.active_q_qubits = (
            config["qubits"]
        )

        st.session_state.active_q_type = (
            config["q_type"]
        )

        st.session_state.active_q_depth = (
            config["layers"]
        )

        st.success(
            "Training Complete"
        )

    render_training_dashboard()