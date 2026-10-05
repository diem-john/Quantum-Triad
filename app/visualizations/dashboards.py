import pandas as pd
import plotly.express as px


def correlation_heatmap(
        correlation_matrix
):
    """
    Pearson Correlation Dashboard
    """

    fig = px.imshow(
        correlation_matrix,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        aspect="auto",
        zmin=-1,
        zmax=1
    )

    fig.update_layout(
        coloraxis_showscale=False
    )

    return fig


def feature_importance_chart(
        features,
        importances
):
    """
    Random Forest Importance Chart
    """

    df = pd.DataFrame({
        "Feature": features,
        "Importance": importances
    })

    df = df.sort_values(
        "Importance",
        ascending=True
    )

    fig = px.bar(
        df,
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale="Blues"
    )

    fig.update_layout(
        coloraxis_showscale=False
    )

    return fig


def temporal_sequence_dashboard(
        sequence_data,
        feature_names
):
    """
    LSTM Temporal Sequence Visualization
    """

    import plotly.graph_objects as go

    fig = go.Figure()

    timeline = [
        "t-3",
        "t-2",
        "t-1",
        "t"
    ]

    for idx, feature in enumerate(feature_names):

        fig.add_trace(
            go.Scatter(
                x=timeline,
                y=sequence_data[:, idx],
                mode="lines+markers",
                name=feature
            )
        )

    fig.update_layout(
        xaxis_title="Timeline",
        yaxis_title="Scaled Stress Level (rad)",
        hovermode="x unified"
    )

    return fig