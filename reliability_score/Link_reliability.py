
"""
Link reliability calculation for the SBM link prediction method.

This module:
1. Finds all currently unobserved links.
2. Calculates their reliability scores using partitions sampled
   by the Metropolis algorithm.
3. Stores the results in a DataFrame.
4. Calculates descriptive statistics of the reliability scores.

The link probability formula follows the lecture slides:

    P(link | T) = (l_ts + 1) / (r_ts + 2)

The final reliability is estimated by averaging this quantity
over the retained partitions.

Assumptions:
- The network is undirected and unweighted.
- Self-loops are excluded.
- The sampled partitions represent the target SBM distribution.
"""

import numpy as np
import pandas as pd

from sbm.sbm_weights import block_counts


def candidate_links(A):
    """
    Find all currently unobserved node pairs.

    Parameters
    ----------
    A : numpy.ndarray
        Symmetric adjacency matrix.

    Returns
    -------
    candidates : list of tuple
        Unobserved node pairs (v, w), with v < w.
    """

    number_of_nodes = A.shape[0]
    candidates = []

    for v in range(number_of_nodes):
        for w in range(v + 1, number_of_nodes):

            if A[v, w] == 0:
                candidates.append((v, w))

    return candidates


def calculate_reliabilities(
    A,
    accepted_groupings,
    accepted_groupings_weights,
    initial_weight,
    K
):
    """
    Calculate reliability scores for all currently unobserved links.

    For each retained partition T, the conditional link
    probability is

        P(link | T) = (l_ts + 1) / (r_ts + 2)

    where:
        l_ts = observed links between blocks t and s
        r_ts = possible links between blocks t and s

    The reliability is estimated by averaging these probabilities
    across retained partitions.

    Parameters
    ----------
    A : numpy.ndarray
        Symmetric adjacency matrix.

    accepted_groupings : list of numpy.ndarray
        Partitions retained by the Metropolis algorithm.

    accepted_groupings_weights : list of float
        Log-weights corresponding to retained partitions.
        Included for compatibility with Sorin's output.

    initial_weight : float
        Initial log-weight returned by the Metropolis algorithm.
        Included for compatibility with the existing interface.

    K : int
        Number of possible SBM groups.

    Returns
    -------
    reliabilities : dict
        Maps each candidate pair (v, w) to its reliability score.
    """

    if len(accepted_groupings) == 0:
        raise ValueError("No sampled partitions were provided.")

    if len(accepted_groupings) != len(accepted_groupings_weights):
        raise ValueError(
            "The number of groupings and weights must match."
        )

    candidates = candidate_links(A)

    reliabilities = {
        pair: 0.0
        for pair in candidates
    }

    for partition in accepted_groupings:

        L, R = block_counts(A, partition, K)

        for v, w in candidates:

            t = partition[v]
            s = partition[w]

            l_ts = L[t, s]
            r_ts = R[t, s]

            link_probability = (
                (l_ts + 1) / (r_ts + 2)
            )

            reliabilities[(v, w)] += link_probability

    number_of_partitions = len(accepted_groupings)

    for pair in reliabilities:
        reliabilities[pair] /= number_of_partitions

    return reliabilities


def reliabilities_to_dataframe(reliabilities):
    """
    Convert reliability scores into a sorted DataFrame.
    """

    rows = []

    for (v, w), score in reliabilities.items():
        rows.append({
            "node_1": v,
            "node_2": w,
            "reliability": score
        })

    df = pd.DataFrame(
        rows,
        columns=["node_1", "node_2", "reliability"]
    )

    df = df.sort_values(
        by="reliability",
        ascending=False
    ).reset_index(drop=True)

    return df


def reliability_statistics(reliability_df):
    """
    Calculate descriptive statistics of reliability scores.

    Useful for inspecting the distribution before selecting
    a prediction threshold.
    """

    scores = reliability_df["reliability"]

    stats = pd.Series({
        "number_of_candidates": len(scores),
        "minimum": scores.min(),
        "mean": scores.mean(),
        "median": scores.median(),
        "standard_deviation": scores.std(),
        "75th_percentile": scores.quantile(0.75),
        "90th_percentile": scores.quantile(0.90),
        "95th_percentile": scores.quantile(0.95),
        "99th_percentile": scores.quantile(0.99),
        "maximum": scores.max()
    })

    return stats


def top_predictions(reliability_df, n=20):
    """
    Return the n candidate links with the highest reliability.
    """

    return reliability_df.head(n)
