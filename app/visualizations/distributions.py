import pandas as pd
import plotly.express as px


def prediction_distribution(
        probabilities,
        labels
):
    """
    Phase 3 Prediction Histogram
    """

    df = pd.DataFrame({
        "Probability": probabilities,
        "True Label": labels
    })

    fig = px.histogram(
        df,
        x="Probability",
        color="True Label",
        barmode="overlay",
        nbins=40,
        opacity=0.7
    )

    fig.update_layout(
        title="Prediction Confidence Separation"
    )

    return fig


def calibration_distribution(
        scores,
        labels
):
    """
    QCP Non-Conformity Distribution
    """

    df = pd.DataFrame({
        "Score": scores,
        "Label": labels
    })

    fig = px.histogram(
        df,
        x="Score",
        color="Label",
        marginal="box",
        nbins=30
    )

    return fig


def add_qhat_threshold(
        fig,
        q_hat
):
    """
    Add conformal threshold
    """

    fig.add_vline(
        x=q_hat,
        line_dash="dash",
        line_color="green",
        annotation_text=f"q_hat={q_hat:.3f}"
    )

    return fig