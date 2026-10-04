from collections import Counter

import pandas as pd
import plotly.express as px
from dash import dcc, html

UNASSIGNED = "Unassigned"

SPECIES = "Species (nodes)"
INTERACTIONS = "Interactions (by consumer)"

METRIC_LABELS = {
    SPECIES: "Number of species",
    INTERACTIONS: "Number of feeding links",
}


def _phylum(data):
    value = data.get("phylum")
    return UNASSIGNED if not value or value == "NA" else value


def _counts(graph, metric):
    if metric == INTERACTIONS:
        counter = Counter(
            _phylum(graph.nodes[consumer]) for consumer, _ in graph.edges()
        )
    else:
        counter = Counter(_phylum(data) for _, data in graph.nodes(data=True))
    return counter.most_common()


def phylum_figure(graph, metric=SPECIES):
    ordered = _counts(graph, metric)
    frame = pd.DataFrame(ordered, columns=["Phylum", "Count"])
    total = frame["Count"].sum()
    frame["Share"] = frame["Count"] / total
    count_label = METRIC_LABELS[metric]

    figure = px.bar(
        frame,
        x="Count",
        y="Phylum",
        orientation="h",
        color="Count",
        hover_data={"Share": ":.1%"},
        labels={
            "x": count_label,
            "y": "Phylum",
            "color": count_label,
        },
        color_continuous_scale=px.colors.sequential.Turbo,
        title=f"{count_label} per phylum",
        text=frame["Count"],
    )
    figure.update_traces(
        hovertemplate=(
            "Phylum: %{y}<br>"
            + count_label
            + ": %{x}<br>Share: %{customdata[0]:.1%}"
            + "<extra></extra>"
        ),
        textposition="outside",
        texttemplate="%{x}",
        cliponaxis=False,
        marker_line_width=0,
    )
    figure.update_layout(
        yaxis=dict(autorange="reversed", title=None),
        xaxis=dict(
            range=[0, frame["Count"].max() * 1.15],
            gridcolor="rgba(0,0,0,0.08)",
        ),
        coloraxis_colorbar=dict(title=""),
        margin=dict(l=10, r=10, t=60, b=50),
        height=max(500, 24 * len(frame)),
        hoverlabel=dict(font_size=14),
    )
    return figure


def render(graph, metric=SPECIES):
    return html.Div([
        html.H2("Phylum distribution"),
        html.P(
            f"{graph.number_of_nodes()} taxa"
        ),
        html.P(
            f"{graph.number_of_edges()} feeding links (edges)"
        ),
        dcc.RadioItems(
            id="phylum-metric",
            options=[
                {"label": SPECIES, "value": SPECIES},
                {"label": INTERACTIONS, "value": INTERACTIONS},
            ],
            value=metric,
            inline=True,
            style={"marginBottom": "12px"},
        ),
        dcc.Graph(
            id="phylum-bar",
            figure=phylum_figure(graph, metric),
        ),
    ])