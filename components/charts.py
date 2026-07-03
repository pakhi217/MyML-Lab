"""
Reusable, theme-aware Plotly chart builders.
Every function takes the active theme dict (from components.theme.get_theme)
so charts automatically re-color when the user switches dark/light mode.
"""

import plotly.graph_objects as go
import numpy as np


def _base_layout(theme, title=None, height=380):
    return dict(
        title=dict(text=title, font=dict(size=14, color=theme["text"])) if title else None,
        paper_bgcolor=theme["chart_bg"],
        plot_bgcolor=theme["chart_bg"],
        font=dict(color=theme["text_secondary"], family="Plus Jakarta Sans"),
        margin=dict(l=40, r=20, t=40 if title else 20, b=40),
        height=height,
        xaxis=dict(gridcolor=theme["chart_grid"], zerolinecolor=theme["chart_grid"]),
        yaxis=dict(gridcolor=theme["chart_grid"], zerolinecolor=theme["chart_grid"]),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )


def correlation_heatmap(theme, corr_df):
    fig = go.Figure(
        data=go.Heatmap(
            z=corr_df.values,
            x=corr_df.columns,
            y=corr_df.columns,
            colorscale=[[0, theme["secondary_accent"]], [0.5, theme["chart_bg"]], [1, theme["accent"]]],
            zmid=0,
            text=np.round(corr_df.values, 2),
            texttemplate="%{text}",
            textfont=dict(size=9),
        )
    )
    fig.update_layout(**_base_layout(theme, "Correlation Heatmap", 420))
    return fig


def scatter_plot(theme, x, y, color=None, x_label="X", y_label="Y", title="Scatter Plot"):
    fig = go.Figure(
        data=go.Scatter(
            x=x, y=y, mode="markers",
            marker=dict(
                color=color if color is not None else theme["accent"],
                colorscale=[[0, theme["secondary_accent"]], [1, theme["accent"]]],
                size=8, opacity=0.8,
                line=dict(width=0.5, color=theme["bg"]),
            ),
        )
    )
    layout = _base_layout(theme, title)
    layout["xaxis"]["title"] = x_label
    layout["yaxis"]["title"] = y_label
    fig.update_layout(**layout)
    return fig


def histogram(theme, data, x_label="Value", title="Distribution"):
    fig = go.Figure(data=go.Histogram(x=data, marker=dict(color=theme["accent"], opacity=0.85)))
    layout = _base_layout(theme, title)
    layout["xaxis"]["title"] = x_label
    layout["yaxis"]["title"] = "Count"
    fig.update_layout(**layout)
    return fig


def box_plot(theme, df, columns, title="Box Plot"):
    fig = go.Figure()
    colors = [theme["accent"], theme["secondary_accent"]]
    for i, col in enumerate(columns):
        fig.add_trace(go.Box(y=df[col], name=col, marker_color=colors[i % 2]))
    fig.update_layout(**_base_layout(theme, title))
    return fig


def feature_importance_chart(theme, feature_names, importances, title="Feature Importance"):
    order = np.argsort(importances)
    fig = go.Figure(
        data=go.Bar(
            x=np.array(importances)[order],
            y=np.array(feature_names)[order],
            orientation="h",
            marker=dict(
                color=np.array(importances)[order],
                colorscale=[[0, theme["secondary_accent"]], [1, theme["accent"]]],
            ),
        )
    )
    fig.update_layout(**_base_layout(theme, title, height=max(320, 32 * len(feature_names))))
    return fig


def confusion_matrix_chart(theme, matrix, labels, title="Confusion Matrix"):
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=[f"Pred {l}" for l in labels],
            y=[f"Actual {l}" for l in labels],
            colorscale=[[0, theme["chart_bg"]], [1, theme["accent"]]],
            text=matrix,
            texttemplate="%{text}",
            textfont=dict(size=13, color=theme["text"]),
        )
    )
    fig.update_layout(**_base_layout(theme, title))
    return fig


def roc_curve_chart(theme, fpr, tpr, auc, title="ROC Curve"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC={auc:.3f})",
                              line=dict(color=theme["accent"], width=3)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random",
                              line=dict(color=theme["text_secondary"], width=1, dash="dash")))
    layout = _base_layout(theme, title)
    layout["xaxis"]["title"] = "False Positive Rate"
    layout["yaxis"]["title"] = "True Positive Rate"
    fig.update_layout(**layout)
    return fig


def actual_vs_predicted_chart(theme, y_true, y_pred, title="Actual vs Predicted"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_true, y=y_pred, mode="markers",
                              marker=dict(color=theme["accent"], size=7, opacity=0.75),
                              name="Predictions"))
    lo, hi = min(np.min(y_true), np.min(y_pred)), max(np.max(y_true), np.max(y_pred))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                              line=dict(color=theme["secondary_accent"], dash="dash"),
                              name="Ideal"))
    layout = _base_layout(theme, title)
    layout["xaxis"]["title"] = "Actual"
    layout["yaxis"]["title"] = "Predicted"
    fig.update_layout(**layout)
    return fig


def loss_curve_chart(theme, loss_history, title="Training Loss Curve"):
    fig = go.Figure(
        data=go.Scatter(
            y=loss_history, mode="lines",
            line=dict(color=theme["accent"], width=2.5),
            fill="tozeroy", fillcolor=theme["accent_soft"],
        )
    )
    layout = _base_layout(theme, title)
    layout["xaxis"]["title"] = "Iteration"
    layout["yaxis"]["title"] = "Loss"
    fig.update_layout(**layout)
    return fig


def pca_projection_chart(theme, components, color=None, title="PCA Projection"):
    fig = go.Figure(
        data=go.Scatter(
            x=components[:, 0], y=components[:, 1], mode="markers",
            marker=dict(
                color=color if color is not None else theme["accent"],
                colorscale=[[0, theme["secondary_accent"]], [1, theme["accent"]]],
                size=8, opacity=0.8,
            ),
        )
    )
    layout = _base_layout(theme, title)
    layout["xaxis"]["title"] = "Principal Component 1"
    layout["yaxis"]["title"] = "Principal Component 2"
    fig.update_layout(**layout)
    return fig


def cluster_visualization_chart(theme, X, labels, centroids=None, title="Cluster Visualization"):
    fig = go.Figure(
        data=go.Scatter(
            x=X[:, 0], y=X[:, 1], mode="markers",
            marker=dict(
                color=labels, colorscale="Sunsetdark",
                size=8, opacity=0.8, line=dict(width=0.4, color=theme["bg"]),
            ),
            name="Points",
        )
    )
    if centroids is not None:
        fig.add_trace(go.Scatter(
            x=centroids[:, 0], y=centroids[:, 1], mode="markers",
            marker=dict(color=theme["accent"], size=16, symbol="x", line=dict(width=2, color=theme["text"])),
            name="Centroids",
        ))
    layout = _base_layout(theme, title)
    layout["xaxis"]["title"] = "Feature 1"
    layout["yaxis"]["title"] = "Feature 2"
    fig.update_layout(**layout)
    return fig


def elbow_chart(theme, k_range, inertias, title="Elbow Method"):
    fig = go.Figure(
        data=go.Scatter(x=list(k_range), y=inertias, mode="lines+markers",
                         line=dict(color=theme["accent"], width=2.5),
                         marker=dict(size=8, color=theme["secondary_accent"]))
    )
    layout = _base_layout(theme, title)
    layout["xaxis"]["title"] = "Number of Clusters (k)"
    layout["yaxis"]["title"] = "Inertia"
    fig.update_layout(**layout)
    return fig
