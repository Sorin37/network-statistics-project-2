from collections import Counter
from pathlib import Path

import networkx as nx
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"


def find_raw_gml():
    files = [p for p in ROOT.rglob("Foodweb_Sanak_Intertidal.gml") if p.parent.name.lower() == "raw"]
    if len(files) != 1:
        raise ValueError(f"Expected one raw GML file, found {len(files)}")
    return files[0]


def duplicate_count(values):
    return sum(count - 1 for count in Counter(values).values() if count > 1)


def check(actual, expected, label):
    if actual != expected:
        raise ValueError(f"Expected {expected} {label}, found {actual}")


def main():
    G_raw = nx.read_gml(find_raw_gml(), label=None)
    if not G_raw.is_directed():
        raise ValueError("The raw graph should be directed")

    node_rows = []
    for node_id, data in G_raw.nodes(data=True):
        node_rows.append({
            "id": int(node_id),
            "original_label": data.get("label"),
            "TSN": data.get("TSN"),
            "name": data.get("name"),
            "phylum": data.get("phylum"),
            "is_human": data.get("name") == "Homo sapiens",
        })
    nodes = pd.DataFrame(node_rows).sort_values("id").reset_index(drop=True)

    missing_name = sum(v is None or str(v).strip() == "" for v in nodes["name"])
    missing_tsn = sum(v is None or str(v).strip() == "" for v in nodes["TSN"])
    missing_phylum = sum(v is None or str(v).strip() == "" for v in nodes["phylum"])
    duplicate_ids = duplicate_count(nodes["id"])
    duplicate_names = duplicate_count(nodes["name"])
    duplicate_tsns = duplicate_count(nodes["TSN"])
    unique_phyla = nodes["phylum"].nunique()

    directed_edges = set(G_raw.edges())
    reciprocal_pairs = sum(1 for u, v in directed_edges if u < v and (v, u) in directed_edges)
    raw_self_loops = nx.number_of_selfloops(G_raw)

    G = nx.Graph()
    G.add_nodes_from(G_raw.nodes(data=True))
    G.add_edges_from(G_raw.edges())
    projected_edges = G.number_of_edges()
    projected_self_loops = nx.number_of_selfloops(G)

    # Self-links are not candidates when predicting relationships between different taxa.
    G.remove_edges_from(nx.selfloop_edges(G))

    human_rows = nodes[nodes["is_human"]]
    if len(human_rows) != 1:
        raise ValueError(f"Expected one Homo sapiens node, found {len(human_rows)}")
    human_id = int(human_rows.iloc[0]["id"])

    edge_rows = []
    for first, second in G.edges():
        u, v = sorted((int(first), int(second)))
        edge_rows.append({
            "u": u,
            "v": v,
            "same_phylum": G.nodes[u]["phylum"] == G.nodes[v]["phylum"],
            "involves_human": G.nodes[u]["name"] == "Homo sapiens" or G.nodes[v]["name"] == "Homo sapiens",
        })
    edges = pd.DataFrame(edge_rows).sort_values(["u", "v"]).reset_index(drop=True)

    same_phylum_edges = int(edges["same_phylum"].sum())
    cross_phylum_edges = len(edges) - same_phylum_edges
    human_edges = int(edges["involves_human"].sum())
    degrees = pd.Series(dict(G.degree()).values(), dtype="float64")
    components = nx.number_connected_components(G)
    isolates = nx.number_of_isolates(G)
    human_degree = G.degree(human_id)

    expected_values = [
        (G_raw.number_of_nodes(), 235, "raw nodes"),
        (G_raw.number_of_edges(), 1804, "raw directed edges"),
        (raw_self_loops, 47, "raw self-loops"),
        (reciprocal_pairs, 14, "reciprocal pairs"),
        (duplicate_ids, 0, "duplicate node IDs"),
        (duplicate_names, 0, "duplicate names"),
        (duplicate_tsns, 0, "duplicate TSNs"),
        (missing_name, 0, "missing names"),
        (missing_tsn, 0, "missing TSNs"),
        (missing_phylum, 0, "missing phylum values"),
        (unique_phyla, 28, "unique phyla"),
        (projected_edges, 1790, "projected edges"),
        (projected_self_loops, 47, "projected self-loops"),
        (G.number_of_nodes(), 235, "final nodes"),
        (G.number_of_edges(), 1743, "final edges"),
        (components, 1, "connected components"),
        (isolates, 0, "isolates"),
        (human_degree, 70, "Homo sapiens degree"),
        (same_phylum_edges, 457, "same-phylum edges"),
        (cross_phylum_edges, 1286, "cross-phylum edges"),
        (human_edges, 70, "human-related edges"),
    ]
    for actual, expected, label in expected_values:
        check(actual, expected, label)

    human = human_rows.iloc[0]
    for field, expected in {
        "id": 56, "original_label": "57", "TSN": "180092",
        "name": "Homo sapiens", "phylum": "Chordata", "is_human": True,
    }.items():
        check(human[field], expected, f"Homo sapiens {field}")

    na_names = list(nodes.loc[nodes["phylum"] == "NA", "name"])
    check(na_names, ["Detritus complex", "Biofilm Complex"], 'names with phylum "NA"')

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    nodes.to_csv(PROCESSED_DIR / "nodes_clean.csv", index=False)
    edges.to_csv(PROCESSED_DIR / "edges_undirected.csv", index=False)
    nx.write_gml(G, PROCESSED_DIR / "foodweb_undirected.gml")

    summary = f"""Raw graph
---------
Graph type: {type(G_raw).__name__}
Nodes: {G_raw.number_of_nodes()}
Directed edges: {G_raw.number_of_edges()}
Self-loops: {raw_self_loops}
Reciprocal pairs: {reciprocal_pairs}

Undirected analysis graph
-------------------------
Nodes: {G.number_of_nodes()}
Edges: {G.number_of_edges()}
Connected components: {components}
Isolated nodes: {isolates}
Density: {nx.density(G):.8f}
Minimum degree: {int(degrees.min())}
Mean degree: {degrees.mean():.4f}
Median degree: {degrees.median():.4f}
Maximum degree: {int(degrees.max())}
Human degree: {human_degree}

Metadata
--------
Unique phyla: {unique_phyla}
Missing name: {missing_name}
Missing TSN: {missing_tsn}
Missing phylum: {missing_phylum}
Human node ID: {human_id}

RQ labels on observed undirected edges
---------------------------------------
Same-phylum edges: {same_phylum_edges}
Cross-phylum edges: {cross_phylum_edges}
Human-related edges: {human_edges}
"""
    (RESULTS_DIR / "data_cleaning_summary.txt").write_text(summary, encoding="utf-8")

    print("Data cleaning completed.\n")
    print(f"Raw graph:\n{G_raw.number_of_nodes()} nodes, {G_raw.number_of_edges()} directed edges\n")
    print(f"Final analysis graph:\n{G.number_of_nodes()} nodes, {G.number_of_edges()} undirected edges")
    print(f"{components} connected component\n{isolates} isolates\n")
    print(f"Labels:\n{same_phylum_edges} same-phylum edges")
    print(f"{cross_phylum_edges} cross-phylum edges\n{human_edges} human-related edges\n")
    print("Saved:\ndata/processed/nodes_clean.csv\ndata/processed/edges_undirected.csv")
    print("data/processed/foodweb_undirected.gml\nresults/data_cleaning_summary.txt")


if __name__ == "__main__":
    main()
