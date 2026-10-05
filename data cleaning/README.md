# Data cleaning

This folder contains the raw Sanak intertidal food-web data and the preprocessing used to create the undirected graph for later analysis.

## What the script does

The raw GML is a directed food web, where an edge records which taxon consumes another. This project intentionally uses an undirected analysis graph, so the cleaning script:

1. loads the raw GML as a directed graph and checks its node metadata;
2. converts directed feeding records into undirected trophic interactions;
3. merges reciprocal directed edges into one undirected edge;
4. removes self-loops from the analysis graph without deleting any nodes;
5. preserves `label`, `TSN`, `name`, and `phylum` metadata;
6. adds an `is_human` flag for `Homo sapiens`;
7. labels observed edges as same-phylum, cross-phylum, and human-related.

The files in `raw/` are not changed.

## Run

The script requires `networkx` and `pandas`.

## Folder contents


`nodes_clean.csv` contains the original node metadata plus `is_human`. The literal phylum value `NA` is kept as text, so later pandas code should read it with `keep_default_na=False`.

`edges_undirected.csv` stores each undirected edge once as `min(node_id), max(node_id)`. It also includes `same_phylum` and `involves_human` labels for later evaluation.

`foodweb_undirected.gml` is the graph intended for later SBM and link-prediction work.

## Verified output

- Raw graph: 235 nodes and 1,804 directed edges
- Raw self-loops: 47
- Reciprocal non-self-loop pairs: 14
- Final graph: 235 nodes and 1,743 undirected edges
- Connected components: 1
- Isolated nodes: 0
- Same-phylum edges: 457
- Cross-phylum edges: 1,286
- Human-related edges: 70
- `Homo sapiens` degree: 70
