"""Saved clustering figures shared by Streamlit, notebooks and Quarto.

Rendering the saved linkage matrix never fits a clustering model or reads data.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.cluster.hierarchy import dendrogram


def hierarchy_figure(hierarchy, snapshot, n_clusters=5):
    """Truncate the actual Ward tree; show the cut used for exact-k memberships."""
    rows = hierarchy.loc[pd.to_datetime(hierarchy.date).eq(pd.Timestamp(snapshot))]
    figure = go.Figure()
    if rows.empty:
        return figure.update_layout(title="No saved merge tree at this snapshot")
    tree = rows[["left", "right", "distance", "count"]].to_numpy(dtype=float)
    n = len(tree) + 1
    k = min(n_clusters, n)
    included = n - k
    lower = tree[included - 1, 2] if included else 0.
    upper = tree[included, 2] if included < len(tree) else tree[-1, 2]
    cut = (lower + upper) / 2
    data = dendrogram(tree, truncate_mode="lastp", p=max(25, k), no_plot=True,
                      show_leaf_counts=True, color_threshold=cut)
    colors = {"C0": "#68716e", "C1": "#287d68", "C2": "#536da6", "C3": "#b36a2e",
              "C4": "#7a4e8a", "C5": "#247a9b"}
    for xs, ys, color in zip(data['icoord'], data['dcoord'], data['color_list']):
        figure.add_trace(go.Scatter(x=xs, y=ys, mode='lines', line=dict(color=colors.get(color, '#68716e')),
            showlegend=False, hovertemplate='Ward merge height %{y:.2f}<extra></extra>'))
    figure.add_hline(y=cut, line_dash='dash', line_color='#b13636',
                     annotation_text=f'Cut for {k} groups', annotation_position='top left')
    if lower == upper:
        figure.add_annotation(text='Tied merge heights: exact group count follows merge order.',
                              xref='paper', yref='paper', x=.5, y=1.10, showarrow=False)
    # A singleton leaf is its within-fit row index; label that explicitly, never as a stock ID.
    labels = [label if label.startswith('(') else f'row {label}' for label in data['ivl']]
    figure.update_layout(template='plotly_white', height=450,
        title=f'Actual Ward merge tree · {pd.Timestamp(snapshot):%b %Y} · {n:,} stocks',
        margin=dict(l=10, r=10, t=65, b=25), yaxis_title='Ward merge height',
        xaxis=dict(tickmode='array', tickvals=np.arange(len(labels))*10+5, ticktext=labels,
                   title='Collapsed branches (stock counts) or individual input rows'))
    return figure
