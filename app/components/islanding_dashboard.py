import streamlit as st
import pandas as pd


def render_islanding_dashboard(
    zones
):

    st.divider()

    st.markdown(
        "### 🌊 Identified Islands"
    )

    rows = []

    for zone, info in zones.items():

        rows.append(
            {
                "Zone": zone,
                "Size": info["size"],
                "Nodes": (
                    ",".join(
                        map(
                            str,
                            info["nodes"]
                        )
                    )
                )
            }
        )

    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True
    )

    st.json(zones)