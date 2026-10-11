from metropolis import initial_convergence, log_weight
import networkx as nx
import numpy as np
import os
import plotly.graph_objects as go
from tqdm import tqdm


def run_number_of_groups_experiment(graph_path="foodweb_undirected.gml", steps=3000, max_number_of_groups=50):
    """
    Runs the initial convergence for every number of groups from 2 up to
    max_number_of_groups on the same graph and with the same parameters as
    main, except for the number of groups.

    Parameters
    ----------
    graph_path : str, optional;
        Path to the gml file of the graph. The default is
        "foodweb_undirected.gml".
    steps : int, optional;
        The number of iterative steps taken to converge the weight, passed
        to initial_convergence. The default is 3000.
    max_number_of_groups : int, optional;
        The largest number of groups to try. The default is 50.

    Returns
    -------
    number_of_groups_values : list of int;
        The number of groups used in each experiment.
    weights : list of float;
        The converged weight for each number of groups.

    """
    graph = nx.read_gml(graph_path)
    adjacency_matrix = nx.to_numpy_array(graph)

    number_of_groups_values = list(range(2, max_number_of_groups + 1))
    weights = []

    for number_of_groups in tqdm(number_of_groups_values, desc="Number of groups"):
        weight, grouping = initial_convergence(adjacency_matrix, number_of_groups, steps, show_progress=False)
        weights.append(weight)

    plot_number_of_groups_experiment(number_of_groups_values, weights)
    return number_of_groups_values, weights


def plot_number_of_groups_experiment(number_of_groups_values, weights):
    """
    Plots the converged weight for each number of groups, with a marker on
    the best value, saving it as an interactive HTML file in the plots
    folder.

    Parameters
    ----------
    number_of_groups_values : list of int;
        The number of groups used in each experiment.
    weights : list of float;
        The converged weight for each number of groups.

    Returns
    -------
    None.

    """
    best_index = int(np.argmax(weights))
    best_number_of_groups = number_of_groups_values[best_index]
    best_weight = weights[best_index]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=number_of_groups_values,
        y=weights,
        mode="lines+markers",
        name="weight",
        hovertemplate="number of groups=%{x}<br>weight=%{y:.4f}<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=[best_number_of_groups],
        y=[best_weight],
        mode="markers",
        name=f"best ({best_number_of_groups})",
        marker=dict(color="green", size=16, symbol="star"),
        hovertemplate="best<br>number of groups=%{x}<br>weight=%{y:.4f}<extra></extra>"
    ))
    fig.update_layout(
        title="Weight by number of groups",
        xaxis_title="Number of groups",
        yaxis_title="weight",
        template="plotly_white"
    )

    os.makedirs("plots", exist_ok=True)
    fig.write_html(os.path.join("plots", "number_of_groups_experiment.html"),
    config={
        'toImageButtonOptions': {
            'format': 'svg',
            'filename': 'custom_image',
            'height': 500,
            'width': 700,
            'scale': 1
        }
    })

    print(f"Best number of groups: {best_number_of_groups} (weight {best_weight:.4f})")


if __name__ == "__main__":
    run_number_of_groups_experiment()
