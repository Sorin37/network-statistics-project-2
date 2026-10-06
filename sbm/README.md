# sbm_weights.py

Scores a grouping of the taxa under the stochastic block model (SBM).

Given the food web and one way of sorting the taxa into groups, the code returns a single number saying how well that grouping explains the observed links. Higher is better. The Metropolis algorithm uses this number to decide whether to accept or reject a proposed change to a grouping.

The score is the weight f(T) from the lecture slides (slide 31), where T is a grouping:

    f(T) = product over pairs of groups (t, s) of
           1 / ((r + 1) * C(r, l))

For each pair of groups:

- `l` is the number of links observed between them
- `r` is the number of links possible between them
- `C(r, l)` is "r choose l", the number of ways to place l links among r possible ones

Each pair of groups is counted once, and a group is also paired with itself to cover the links inside it.

f(T) is too small to store as a number for a web of 235 taxa, so the code returns log f(T) instead.

## Inputs

- **`A`**: the adjacency matrix. A 235 x 235 numpy array of 0 and 1, symmetric, with a zero diagonal (undirected graph, no self-loops).
- **`groups`**: one grouping of the taxa. An integer numpy array of length 235, where `groups[i]` is the group of taxon `i`. Values run from 0 to `K - 1`.
- **`K`**: the number of groups allowed. Some groups may be empty.

Taxon numbers are the `id` column of `data cleaning/data/processed/nodes_clean.csv`.

## Functions

- **`block_counts(A, groups, K)`** returns two K x K tables: `L` (observed links) and `R` (possible links) between each pair of groups.
- **`log_weight(A, groups, K)`** returns log f(T) as a float. Higher means the grouping explains the links better.
- **`log_weight_from_counts(L, R)`** does the same from tables you already have.
- **`validate_inputs(A, groups, K)`** raises an error if the inputs are in the wrong format. Call it once before sampling (not in the loop)

## Example

Run from the repository root.

```python
import numpy as np
import pandas as pd
from sbm.sbm_weights import log_weight, validate_inputs

edges = pd.read_csv("data cleaning/data/processed/edges_undirected.csv")
A = np.zeros((235, 235), dtype=int)
A[edges.u, edges.v] = 1
A[edges.v, edges.u] = 1

 # every taxon in group 0
groups = np.zeros(235, dtype=int)
validate_inputs(A, groups, K=1)
print(log_weight(A, groups, K=1)) # result should be -6500.04

# every taxon in its own group
groups = np.arange(235)
print(log_weight(A, groups, K=235)) # -19058.08

# grouped by phylum
nodes = pd.read_csv("data cleaning/data/processed/nodes_clean.csv",
                    keep_default_na=False)
groups, phyla = pd.factorize(nodes.phylum)     # turns phylum names into 0, 1, 2, ...
print(len(phyla))                              # 28
print(log_weight(A, groups, K=len(phyla)))     # -6410.89
```

## Differences from the course code

This code builds on the course file `MetropolisAlgorithm.py`, which computes the same quantity inside its sampling loops. There are two differences. First, it computes the binomial term exactly, using the log-gamma function, whereas the course code uses an approximation for large blocks. Second, the course code loops over every pair of groups on each call, while this version uses matrix products and is faster.

## Tests

```
python -m pytest tests/sbm_weights_test.py
```

The tests cover a graph checked by hand, empty groups, agreement with a slow loop-based version, two properties that must hold for any grouping, and the input checker.
