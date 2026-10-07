import sys
import os

# Adds the project root directory to Python's system path dynamically
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from sbm.sbm_weights import log_weight
import numpy as np
import math
import os
import plotly.graph_objects as go
from tqdm import tqdm
from sklearn.metrics import normalized_mutual_info_score

def initial_convergence(adjacency_matrix, number_of_groups=10, steps=1000):
    """
    Performs initial convergence of the weight in the Metropolis algorithm.
    Also plots the history to plots/initial_convergence_history.html.

    Parameters
    ----------
    adjacency_matrix : Symmetric np.array() with {0, 1} entries;
        Adjacency matrix of the input graph.
    number_of_groups : int, optional;
        The number of communities we assume. The default is 100.
    steps : int, optional;
        The number of iterative steps taken to converge the weight. The default is 1000.

    Returns
    -------
    weight : float;
        The final (converged) value of log(f(P)). The logarithm is given to
        keep the algorithm numerically stable.
    old_grouping : np.array() with int entries;
        The grouping that produced the weight.

    """
    # Initial training for f
    weight = - 10 ** 10
    old_grouping = np.random.choice(number_of_groups, adjacency_matrix.shape[0])
    weight_history = [weight]
    for step in tqdm(range(steps), desc="Initial convergence"):
        fail = True
        while fail:
            # Create new P
            index = np.random.randint(adjacency_matrix.shape[0])
            new_grouping = old_grouping.copy()
            new_grouping[index] = np.random.choice(np.delete(np.arange(number_of_groups), old_grouping[index]))

            new_weight = log_weight(adjacency_matrix, new_grouping, number_of_groups)

            # Test whether we accept the change, and if accepted: update
            log_alpha = new_weight - weight
            if log_alpha >= 0 or np.log(np.random.rand()) < log_alpha:
                weight = new_weight
                old_grouping = new_grouping.copy()
                fail = False
        weight_history.append(weight)

    plot_weight_history(weight_history)
    return weight, old_grouping


def plot_weight_history(f_history):
    """
    Plots the history of the weight values recorded during the initial
    convergence of the Metropolis algorithm
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=f_history,
        mode="lines",
        name="weight"
    ))
    fig.update_layout(
        title="History of the weight during initial convergence",
        xaxis_title="Iteration",
        yaxis_title="log weight",
        template="plotly_white"
    )

    os.makedirs("plots", exist_ok=True)
    fig.write_html(os.path.join("plots", "convergence_weight_history.html"))


def sample_different_groupings(A, initial_weight, initial_grouping, number_of_groups, steps=1000, nmi_threshold=0.5, number_of_changes=1):
    """
    Generates grouping of the given graph depicted by the adjacency matrix A, taking the initialF and initialP from the convergence step
    Makes sure that the generated groupings are different by rejecting changes that are similar to the already accepted groupings
    Will stop after the steps are achieved.
    Also plots the history to plots/sample_history.html.

    Parameters
    ----------
    A : Symmetric np.array() with {0, 1} entries;
        Adjacency matrix of the input graph.
    initial_weight : float;
        Converged value of log(F(P)).
    initial_grouping : np.array() with int entries;
        Permutation that corresponds to F.
    number_of_groups : int, optional;
        The number of communities we assume. The default is 100.
    steps : int, optional;
        The number of random changes the algorithm does before stopping. The default is 1000.
    nmi_threshold : float, optional;
        Threshold that accepts or rejects a new grouping, when compared with the NMI (normal mutual information)
        of the already accepted groupings (measures correlation of groupings).
        The default is 0.5.
    number_of_changes: int, optional;
        The number of nodes that are given a random grouping
        The default is 1.

    Returns
    -------
    accepted_groupings : a list of groupings (which are lists with int entries from 0 to K-1);
        Gives the accepted groupings configurations.
    """
    weight = initial_weight
    old_grouping = initial_grouping.copy()
    accepted_groupings = [initial_grouping.copy()]
    weight_history = []
    accepted_steps = []
    accepted_weights = []
    accepted_nmis = []
    rejected_steps = []
    rejected_weights = []
    rejected_nmis = []

    # Doing the iterations
    for step in tqdm(range(steps), desc='Metropolis'):

        # Create new grouping
        index = np.random.choice(A.shape[0], size=number_of_changes, replace=False)
        new_grouping = old_grouping.copy()
        for idx in index:
            new_grouping[idx] = np.random.choice(np.delete(np.arange(number_of_groups), old_grouping[idx]))

        # Compute the weight
        new_weight = log_weight(A, new_grouping, number_of_groups)

        # Decide whether to keep the new grouping
        log_alpha = new_weight - weight
        if log_alpha >= 0 or np.log(np.random.rand()) < log_alpha:
            # Max NMI across already accepted groupings
            max_nmi = 0.0
            for accepted_grouping in accepted_groupings:
                nmi = normalized_mutual_info_score(
                    accepted_grouping,
                    new_grouping
                )
                max_nmi = max(max_nmi, nmi)

            # change is accepted so the next iteration will begin with the new grouping
            # lower NMI is better so the groupings are more different
            if max_nmi < nmi_threshold:
                accepted_groupings.append(new_grouping)
                weight = new_weight
                old_grouping = new_grouping.copy()
                accepted_steps.append(step)
                accepted_weights.append(weight)
                accepted_nmis.append(max_nmi)
            else:
                rejected_steps.append(step)
                rejected_weights.append(weight)
                rejected_nmis.append(max_nmi)

        weight_history.append(weight)

    plot_sample_history(
        weight_history,
        accepted_steps, accepted_weights, accepted_nmis,
        rejected_steps, rejected_weights, rejected_nmis
    )
    return accepted_groupings


def plot_sample_history(weight_history, accepted_steps, accepted_weights, accepted_nmis,
                        rejected_steps, rejected_weights, rejected_nmis):
    """
    Plots the weight of the Metropolis chain for every step and marks each
    accepted grouping and each grouping rejected by the NMI threshold,
    saving it as an interactive HTML file in the plots folder. Hovering over
    a marker shows the max NMI of that step.

    Parameters
    ----------
    weight_history : list of float;
        The weight log(f(P)) of the chain after each step.
    accepted_steps : list of int;
        The step indices at which a new grouping was accepted.
    accepted_weights : list of float;
        The weight of the chain at each accepted step.
    accepted_nmis : list of float;
        The max NMI with the already accepted groupings at each accepted step.
    rejected_steps : list of int;
        The step indices at which a grouping was rejected by the NMI threshold.
    rejected_weights : list of float;
        The weight of the chain at each NMI-rejected step.
    rejected_nmis : list of float;
        The max NMI with the already accepted groupings at each NMI-rejected
        step.

    Returns
    -------
    None.

    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=weight_history,
        mode="lines",
        name="weight",
        line=dict(color="orange", width=1)
    ))
    fig.add_trace(go.Scatter(
        x=accepted_steps,
        y=accepted_weights,
        customdata=accepted_nmis,
        mode="markers",
        name="accepted grouping",
        marker=dict(color="green", size=10, symbol="x"),
        hovertemplate="step=%{x}<br>weight=%{y:.4f}<br>max NMI=%{customdata:.4f}<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=rejected_steps,
        y=rejected_weights,
        customdata=rejected_nmis,
        mode="markers",
        name="rejected by NMI",
        marker=dict(color="red", size=8, symbol="triangle-down"),
        hovertemplate="step=%{x}<br>weight=%{y:.4f}<br>max NMI=%{customdata:.4f}<extra></extra>"
    ))
    fig.update_layout(
        title="Metropolis sampling history",
        xaxis_title="Step",
        yaxis_title="weight",
        template="plotly_white"
    )

    os.makedirs("plots", exist_ok=True)
    fig.write_html(os.path.join("plots", "sample_history.html"))