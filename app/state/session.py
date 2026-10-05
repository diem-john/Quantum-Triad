import streamlit as st

from src.mapping import ChiayiMicrogridMapper


def initialize_session():
    """
    Centralized Session State Initialization
    """

    defaults = {

        # Dataset
        "data_loaded": False,
        "coresets_generated": False,

        # Quantum
        "qkn_trained": False,
        "pytorch_qcnn_active": False,

        # Conformal
        "qcp_calibrated": False,

        # Optimization
        "pso_complete": False,

        # Dataset Containers
        "bus_train": [],
        "bus_cal": [],
        "bus_test": [],

        # Active Architecture
        "active_q_qubits": None,
        "active_q_type": None,
        "active_q_depth": None,

        # Models
        "pytorch_model": None,
        "qcp_model": None
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if "mapper" not in st.session_state:

        mapper = ChiayiMicrogridMapper()

        mapper.generate_topology()

        st.session_state.mapper = mapper


def reset_training():
    """
    Reset training artifacts only.
    """

    keys = [
        "pytorch_model",
        "pytorch_qcnn_active",
        "qcp_model",
        "qcp_calibrated"
    ]

    for key in keys:
        if key in st.session_state:
            del st.session_state[key]


def reset_everything():
    """
    Full application reset.
    """

    for key in list(st.session_state.keys()):
        del st.session_state[key]

    initialize_session()