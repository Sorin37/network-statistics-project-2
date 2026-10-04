import hashlib
import math

import networkx as nx
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import dcc, html

from components.phylum_info import UNASSIGNED, _phylum

PALETTE = px.colors.sequential.Turbo
NODE_SIZE = 9
ARROW_SIZE = 6
HUB_RADIUS = 0.1
EDGE_RADIUS = 1.05
HIGHLIGHT_RED = "rgb(214,39,40)"
HIGHLIGHT_GREEN = "rgb(44,160,44)"
FADE_OPACITY = 0.06
NODE_OPACITY = 0.95
EDGE_OPACITY = 0.35
ARROW_OPACITY = 0.75
SELECTED_OUTLINE_WIDTH = 2.5


def _colors(phyla):
    phyla = sorted(set(phyla))
    samples = [index / max(len(phyla) - 1, 1) for index in range(len(phyla))]
    return dict(zip(phyla, px.colors.sample_colorscale(PALETTE, samples)))


def _jitter(node):
    digest = hashlib.md5(str(node).encode()).hexdigest()
    return (int(digest[:8], 16) / 2**32 - 0.5) * 0.05


def _outward_positions(graph):
    positions = nx.spring_layout(graph, seed=42, iterations=100)
    degree = dict(graph.degree())
    rank_order = sorted(graph.nodes, key=degree.get, reverse=True)
    count = max(len(rank_order) - 1, 1)
    out = {}
    for index, node in enumerate(rank_order):
        radius = HUB_RADIUS + (EDGE_RADIUS - HUB_RADIUS) * (index / count)
        angle = math.atan2(positions[node][1], positions[node][0]) + _jitter(node)
        out[node] = (radius * math.cos(angle), radius * math.sin(angle))
    return out


def _edge_segments(positions, graph, phylum, incident=None):
    x, y = [], []
    for source, target in graph.edges():
        if _phylum(graph.nodes[source]) != phylum:
            continue
        if incident is not None and (source == incident or target == incident):
            continue
        ax, ay = positions[source]
        bx, by = positions[target]
        x.extend((ax, bx, None))
        y.extend((ay, by, None))
    return x, y


def _arrow_layer(positions, graph, phylum, incident=None):
    x, y, angles = [], [], []
    for source, target in graph.edges():
        if _phylum(graph.nodes[source]) != phylum:
            continue
        if incident is not None and (source == incident or target == incident):
            continue
        (ax, ay), (bx, by) = positions[source], positions[target]
        x.append((ax + bx) / 2)
        y.append((ay + by) / 2)
        angles.append(np.degrees(np.arctan2(bx - ax, by - ay)))
    return x, y, angles


def _link_layers(positions, edges):
    seg_x, seg_y, arrow_x, arrow_y, angles = [], [], [], [], []
    for source, target in edges:
        (ax, ay), (bx, by) = positions[source], positions[target]
        seg_x.extend((ax, bx, None))
        seg_y.extend((ay, by, None))
        arrow_x.append((ax + bx) / 2)
        arrow_y.append((ay + by) / 2)
        angles.append(np.degrees(np.arctan2(bx - ax, by - ay)))
    return seg_x, seg_y, arrow_x, arrow_y, angles


def _nodes(graph, phylum):
    return [
        (node, data)
        for node, data in graph.nodes(data=True)
        if _phylum(data) == phylum
    ]


def _neighbors(graph, selected):
    prey = {target for source, target in graph.edges() if source == selected}
    predators = {source for source, target in graph.edges() if target == selected}
    return prey | predators


def graph_figure(graph, selected=None):
    positions = _outward_positions(graph)
    colors = _colors(_phylum(data) for _, data in graph.nodes(data=True))
    highlighting = selected is not None
    keep_full = _neighbors(graph, selected) | {selected} if highlighting else None

    figure = go.Figure()
    for phylum in sorted(colors):
        figure.add_trace(go.Scatter(
            x=_edge_segments(positions, graph, phylum, incident=selected if highlighting else None)[0],
            y=_edge_segments(positions, graph, phylum, incident=selected if highlighting else None)[1],
            mode="lines",
            line=dict(color=colors[phylum], width=0.7),
            opacity=FADE_OPACITY if highlighting else EDGE_OPACITY,
            hoverinfo="skip",
            showlegend=False,
            legendgroup=phylum,
        ))
        arrow_x, arrow_y, angles = _arrow_layer(
            positions, graph, phylum, incident=selected if highlighting else None
        )
        figure.add_trace(go.Scatter(
            x=arrow_x,
            y=arrow_y,
            mode="markers",
            marker=dict(
                symbol="arrow-up",
                size=ARROW_SIZE,
                color=colors[phylum],
                angle=angles,
            ),
            opacity=FADE_OPACITY if highlighting else ARROW_OPACITY,
            hoverinfo="skip",
            showlegend=False,
            legendgroup=phylum,
        ))

    if highlighting:
        prey_links = [(source, target) for source, target in graph.edges() if source == selected]
        predator_links = [(source, target) for source, target in graph.edges() if target == selected]
        for color, links in (
            (HIGHLIGHT_GREEN, prey_links),
            (HIGHLIGHT_RED, predator_links),
        ):
            seg_x, seg_y, arrow_x, arrow_y, angles = _link_layers(positions, links)
            for kind, x, y in (
                ("lines", seg_x, seg_y),
                ("arrows", arrow_x, arrow_y),
            ):
                figure.add_trace(go.Scatter(
                    x=x,
                    y=y,
                    mode="lines" if kind == "lines" else "markers",
                    line=dict(color=color, width=1.5) if kind == "lines" else None,
                    marker=dict(
                        symbol="arrow-up",
                        size=ARROW_SIZE,
                        color=color,
                        angle=angles,
                    ) if kind == "arrows" else None,
                    hoverinfo="skip",
                    showlegend=False,
                    legendgroup="selected-links",
                ))

    for phylum in sorted(colors):
        nodes = _nodes(graph, phylum)
        if highlighting:
            opacity = [
                1.0 if node in keep_full else FADE_OPACITY
                for node, _ in nodes
            ]
            outline = [
                SELECTED_OUTLINE_WIDTH if node == selected else 1
                for node, _ in nodes
            ]
        else:
            opacity = NODE_OPACITY
            outline = 1
        figure.add_trace(go.Scatter(
            x=[positions[node][0] for node, _ in nodes],
            y=[positions[node][1] for node, _ in nodes],
            mode="markers",
            name=phylum,
            legendgroup=phylum,
            marker=dict(
                size=NODE_SIZE,
                color=colors[phylum],
                line=dict(width=outline, color="black"),
                opacity=opacity,
            ),
            customdata=[
                [data.get("name") or node, graph.degree(node)]
                for node, data in nodes
            ],
            hovertemplate=(
                "Name: %{customdata[0]}<br>"
                "Phylum: " + phylum + "<br>"
                "Feeding links: %{customdata[1]}<extra></extra>"
            ),
        ))

    figure.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(
            visible=False,
            scaleanchor="x",
            scaleratio=1,
        ),
        legend=dict(title="Phylum"),
        margin=dict(l=10, r=10, t=60, b=10),
        height=750,
    )
    return figure


def _options(graph):
    labeled = sorted(
        (
            (data.get("name") or str(node), _phylum(data), node)
            for node, data in graph.nodes(data=True)
        ),
        key=lambda entry: entry[0].casefold(),
    )
    return [
        {"label": f"{name} ({phylum})", "value": node}
        for name, phylum, node in labeled
    ]


def render(graph):
    return html.Div([
        html.H2("Food web graph"),
        dcc.Dropdown(
            id="node-select",
            options=_options(graph),
            value=None,
            placeholder="Select a taxon to highlight…",
            clearable=True,
            style={"marginBottom": "15px"},
        ),
        dcc.Graph(
            id="foodweb-graph",
            figure=graph_figure(graph),
        ),
    ])