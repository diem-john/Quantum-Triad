import numpy as np
import plotly.graph_objects as go


def plot_feature_radar(
        sample,
        feature_names,
        label
):
    """
    Quantum Feature Radar Chart
    """

    color = (
        "#d62728"
        if label == 1
        else "#1f77b4"
    )

    r_vals = sample.tolist()
    r_vals.append(sample[0])

    theta = feature_names.copy()
    theta.append(feature_names[0])

    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=r_vals,
            theta=theta,
            fill="toself",
            fillcolor=color,
            opacity=0.5,
            line=dict(
                color=color,
                width=2
            )
        )
    )

    fig.update_layout(
        showlegend=False,
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[-np.pi, np.pi]
            )
        )
    )

    return fig