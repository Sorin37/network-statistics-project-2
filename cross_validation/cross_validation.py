"""Edge-level 5-fold cross-validation for the Sanak undirected food-web SBM.

Run from the project root:
    python cross_validation/cross_validation.py --mode split
    python cross_validation/cross_validation.py --mode quick
    python cross_validation/cross_validation.py --mode full

'quick' is a plumbing check only. Do not report quick-mode metrics as final results.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_graph():
    # This is the file used by reliability_score/results_reliability.py.
    path = PROJECT_ROOT / "metropolis_sorin" / "foodweb_undirected.gml"
    G = nx.read_gml(path)
    if G.is_directed() or nx.number_of_selfloops(G):
        raise ValueError("Expected an undirected graph without self-loops.")
    return G


def is_human(attrs):
    flag = str(attrs.get("is_human", "")).strip().lower()
    return attrs.get("name", "").strip().lower() == "homo sapiens" or flag in {"true", "1"}


def valid_phylum(value):
    value = str(value).strip()
    return value if value.lower() not in {"", "none", "nan", "na", "unknown"} else None


def safe_metrics(y_true, scores):
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    if len(np.unique(y)) < 2:
        return {"roc_auc": np.nan, "average_precision": np.nan}
    return {
        "roc_auc": float(roc_auc_score(y, s)),
        "average_precision": float(average_precision_score(y, s)),
    }


def run(mode, n_splits=5, seed=42, groups=16):
    G = load_graph()
    nodes = list(G.nodes())  # MUST stay fixed across every fold.
    idx = {node: i for i, node in enumerate(nodes)}
    A_full = nx.to_numpy_array(G, nodelist=nodes, dtype=int)
    edges = [tuple(sorted((idx[u], idx[v]))) for u, v in G.edges()]
    edges = np.asarray(edges, dtype=int)
    if len(edges) < n_splits:
        raise ValueError("Number of edges must be at least the number of folds.")
    upper_i, upper_j = np.triu_indices(len(nodes), k=1)
    original_nonedges = int(np.sum(A_full[upper_i, upper_j] == 0))
    print(f"Original network: {len(nodes)} nodes, {len(edges)} observed edges, "
          f"{original_nonedges} original non-edges")

    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    summary_rows = []
    prediction_frames = []
    curve_data = []

    # Import the model only when necessary, so split-only is fast.
    if mode != "split":
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from metropolis_sorin.metropolis import (
            initial_convergence, sample_different_groupings,
        )
        from reliability_score.Link_reliability import calculate_reliabilities
        initial_steps = 50 if mode == "quick" else 3000
        metropolis_steps = 1000 if mode == "quick" else 300000
        output_dir = PROJECT_ROOT / "cross_validation" / "results"
        output_dir.mkdir(parents=True, exist_ok=True)

    phyla = [valid_phylum(G.nodes[n].get("phylum", "")) for n in nodes]
    humans = np.asarray([is_human(G.nodes[n]) for n in nodes], dtype=bool)
    for fold, (train_idx, test_idx) in enumerate(kfold.split(edges), start=1):
        test_pairs = edges[test_idx]
        A_train = A_full.copy()
        A_train[test_pairs[:, 0], test_pairs[:, 1]] = 0
        A_train[test_pairs[:, 1], test_pairs[:, 0]] = 0
        assert np.array_equal(A_train, A_train.T)
        assert not A_train.diagonal().any()
        print(f"Fold {fold}: train edges={len(train_idx)}, "
              f"hidden positive edges={len(test_idx)}")

        if mode == "split":
            continue

        # Fit afresh on A_train; never use partitions trained on A_full.
        np.random.seed(seed + fold)
        initial_weight, initial_grouping = initial_convergence(
            A_train, number_of_groups=groups, steps=initial_steps
        )
        partitions, partition_weights = sample_different_groupings(
            A_train,
            initial_weight,
            initial_grouping,
            number_of_groups=groups,
            steps=metropolis_steps,
            nmi_threshold=0.95,
            number_of_changes=1,
        )
        reliability = calculate_reliabilities(
            A_train, partitions, partition_weights, initial_weight, groups
        )

        # Candidates in A_train include both hidden positives and original non-edges.
        mask = A_train[upper_i, upper_j] == 0
        vi, wi = upper_i[mask], upper_j[mask]
        labels = A_full[vi, wi].astype(int)
        scores = np.array([reliability[(int(v), int(w))]
                           for v, w in zip(vi, wi)], dtype=float)
        assert int(labels.sum()) == len(test_idx)
        assert len(labels) == len(test_idx) + original_nonedges

        overall = safe_metrics(labels, scores)
        print(f"  retained partitions={len(partitions)}; "
              f"ROC-AUC={overall['roc_auc']:.4f}; AP={overall['average_precision']:.4f}")
        summary_rows.append({
            "fold": fold, "subgroup": "overall", "n_test": len(labels),
            "n_positive": int(labels.sum()), "n_negative": int((1 - labels).sum()),
            "n_partitions": len(partitions), **overall,
        })

        # Save metadata for the project's research question.
        frame = pd.DataFrame({
            "fold": fold, "node_1": vi, "node_2": wi,
            "taxon_1": [G.nodes[nodes[int(i)]].get("name", str(nodes[int(i)])) for i in vi],
            "taxon_2": [G.nodes[nodes[int(i)]].get("name", str(nodes[int(i)])) for i in wi],
            "phylum_1": [phyla[int(i)] for i in vi],
            "phylum_2": [phyla[int(i)] for i in wi],
            "human_related": humans[vi] | humans[wi],
            "y_true": labels, "reliability": scores,
        })
        prediction_frames.append(frame)
        valid = frame.phylum_1.notna() & frame.phylum_2.notna()
        subgroups = {
            "within_phylum": valid & (frame.phylum_1 == frame.phylum_2),
            "across_phylum": valid & (frame.phylum_1 != frame.phylum_2),
            "human_related": frame.human_related,
            "non_human": ~frame.human_related,
        }
        for name, sel in subgroups.items():
            sub = frame.loc[sel]
            m = safe_metrics(sub.y_true, sub.reliability)
            summary_rows.append({
                "fold": fold, "subgroup": name, "n_test": len(sub),
                "n_positive": int(sub.y_true.sum()),
                "n_negative": int(len(sub) - sub.y_true.sum()),
                "n_partitions": len(partitions), **m,
            })
        fpr, tpr, _ = roc_curve(labels, scores)
        precision, recall, _ = precision_recall_curve(labels, scores)
        curve_data.append((fold, fpr, tpr, recall, precision, overall))

    if mode == "split":
        print("Split-only check complete. No SBM was fitted and no metrics were calculated.")
        return

    metrics = pd.DataFrame(summary_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    predictions.to_csv(output_dir / "test_predictions.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 5))
    for fold, fpr, tpr, _, _, m in curve_data:
        ax.plot(fpr, tpr, label=f"Fold {fold} (AUC={m['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random ranking")
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="5-Fold ROC Curves")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "roc_curves.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for fold, _, _, recall, precision, m in curve_data:
        ax.plot(recall, precision, label=f"Fold {fold} (AP={m['average_precision']:.3f})")
    ax.set(xlabel="Recall", ylabel="Precision", title="5-Fold Precision–Recall Curves")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "pr_curves.png", dpi=180)
    plt.close(fig)

    overall = metrics.query("subgroup == 'overall'")
    print("\nOverall mean (across 5 folds):")
    print(overall[["roc_auc", "average_precision"]].agg(["mean", "std"]).to_string())
    print(f"\nOutputs saved under {output_dir}")
    if mode == "quick":
        print("WARNING: quick-mode metrics are for pipeline testing only, NOT a final evaluation.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["split", "quick", "full"], default="split")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--groups", type=int, default=16)
    args = parser.parse_args()
    run(args.mode, n_splits=args.folds, seed=args.seed, groups=args.groups)
