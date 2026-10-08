
"""
Run SBM link reliability analysis.

This script:
1. Loads the observed food web.
2. Runs the Metropolis algorithm.
3. Calculates link reliability scores.
4. Exports scores and descriptive statistics to CSV.
5. Displays the highest-scoring candidate links.

It can also update taxon names in an existing CSV without
rerunning the Metropolis algorithm.
"""

from pathlib import Path
import sys

import networkx as nx
import numpy as np
import pandas as pd

# Make the project root importable.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from metropolis_sorin.metropolis import (
    initial_convergence,
    sample_different_groupings
)

from reliability_score.Link_reliability import (
    calculate_reliabilities,
    reliabilities_to_dataframe,
    reliability_statistics,
    top_predictions
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

# True: update names in existing CSV without Metropolis.
# False: run the full reliability calculation.
UPDATE_NAMES_ONLY = False
NUMBER_OF_GROUPS = 3
INITIAL_STEPS = 100
METROPOLIS_STEPS = 100000
NMI_THRESHOLD = 0.7
NUMBER_OF_CHANGES = 20


def main():

    # --------------------------------------------------
    # 1. Load the observed network
    # --------------------------------------------------

    graph_path = (
        PROJECT_ROOT
        / "metropolis_sorin"
        / "foodweb_undirected.gml"
    )

    G = nx.read_gml(graph_path)

    # Keep a fixed node order.
    nodes = list(G.nodes())

    print("Number of nodes:", G.number_of_nodes())
    print("Number of observed edges:", G.number_of_edges())

    # Map adjacency matrix indices to actual taxon names.
    taxon_names = {
        i: G.nodes[node].get("name", str(node))
        for i, node in enumerate(nodes)
    }

    # Folder for exported results.
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    scores_path = results_dir / "reliability_scores.csv"
    stats_path = results_dir / "reliability_statistics.csv"

    # --------------------------------------------------
    # 2. Update existing CSV without rerunning Metropolis
    # --------------------------------------------------

    if UPDATE_NAMES_ONLY:

        if not scores_path.exists():
            print("No existing reliability_scores.csv found.")
            print("Set UPDATE_NAMES_ONLY = False to calculate scores.")
            return

        reliability_df = pd.read_csv(scores_path)

        reliability_df["taxon_1"] = (
            reliability_df["node_1"].map(taxon_names)
        )

        reliability_df["taxon_2"] = (
            reliability_df["node_2"].map(taxon_names)
        )

        reliability_df.to_csv(scores_path, index=False)

        print("\nTaxon names updated successfully!")
        print("\nTop 20 highest-scoring candidate links:")
        print(top_predictions(reliability_df, 20).to_string(index=False))

        print("\nUpdated CSV:", scores_path)
        return

    # --------------------------------------------------
    # 3. Construct adjacency matrix
    # --------------------------------------------------

    A = nx.to_numpy_array(
        G,
        nodelist=nodes,
        dtype=int
    )

    # --------------------------------------------------
    # 4. Run Metropolis algorithm
    # --------------------------------------------------

    initial_weight, initial_grouping = initial_convergence(
        A,
        NUMBER_OF_GROUPS,
        INITIAL_STEPS
    )

    accepted_groupings, accepted_groupings_weights = (
        sample_different_groupings(
            A,
            initial_weight,
            initial_grouping,
            NUMBER_OF_GROUPS,
            METROPOLIS_STEPS,
            NMI_THRESHOLD,
            number_of_changes=NUMBER_OF_CHANGES
        )
    )

    print(
        "\nNumber of retained partitions:",
        len(accepted_groupings)
    )

    # --------------------------------------------------
    # 5. Calculate link reliability scores
    # --------------------------------------------------

    reliabilities = calculate_reliabilities(
        A,
        accepted_groupings,
        accepted_groupings_weights,
        initial_weight,
        NUMBER_OF_GROUPS
    )

    reliability_df = reliabilities_to_dataframe(reliabilities)

    # Add actual taxon names.
    reliability_df["taxon_1"] = (
        reliability_df["node_1"].map(taxon_names)
    )

    reliability_df["taxon_2"] = (
        reliability_df["node_2"].map(taxon_names)
    )

    # --------------------------------------------------
    # 6. Calculate descriptive statistics
    # --------------------------------------------------

    stats = reliability_statistics(reliability_df)

    print("\nReliability statistics:")
    print(stats)

    print("\nTop 20 highest-scoring candidate links:")
    print(top_predictions(reliability_df, 20).to_string(index=False))

    # --------------------------------------------------
    # 7. Export results
    # --------------------------------------------------

    reliability_df.to_csv(scores_path, index=False)

    stats.rename_axis("statistic").reset_index(
        name="value"
    ).to_csv(stats_path, index=False)

    print("\nResults exported successfully!")
    print("Reliability scores:", scores_path)
    print("Reliability statistics:", stats_path)


if __name__ == "__main__":
    main()