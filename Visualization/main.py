from dash import Dash, html, dcc, Input, Output
from components import phylum_info, graph_info
import networkx as nx


app = Dash(__name__, suppress_callback_exceptions=True)

G = nx.read_gml("data/Foodweb_Sanak_Intertidal.gml")

app.layout = html.Div([
    dcc.Tabs(
        id="tabs",
        value="tab-1",
        children=[
            dcc.Tab(label="Phylum", value="tab-1"),
            dcc.Tab(label="Graph", value="tab-2"),
        ]
    ),
    html.Div(id="tab-content")
])

@app.callback(
    Output("tab-content", "children"),
    Input("tabs", "value")
)
def render_tab(tab):
    if tab == "tab-1":
        return phylum_info.render(G)
    if tab == "tab-2":
        return graph_info.render(G)
    return html.Div()


@app.callback(
    Output("phylum-bar", "figure"),
    Input("phylum-metric", "value")
)
def update_phylum_figure(metric):
    return phylum_info.phylum_figure(G, metric or phylum_info.SPECIES)


@app.callback(
    Output("foodweb-graph", "figure"),
    Input("node-select", "value")
)
def update_foodweb_graph(selected):
    return graph_info.graph_figure(G, selected)

if __name__ == "__main__":
    app.run(debug=True)