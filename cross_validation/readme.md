# K-Fold Cross-Validation for SBM Link Prediction

## 1. Overview

This folder contains the implementation of **5-fold cross-validation** for evaluating the Stochastic Block Model (SBM) on the Sanak intertidal food web.

The goal is to evaluate how accurately the SBM can recover hidden links and whether prediction performance differs across taxonomic groups, particularly across phyla and for humans.

## 2. Methodology

The dataset contains **235 nodes and 1,743 observed undirected edges**, excluding self-loops.

We use edge-level 5-fold cross-validation:

1. Randomly divide all observed edges into five folds.
2. Remove one fold (20%) from the graph as the test positives.
3. Run the Metropolis algorithm on the remaining training graph to generate SBM partitions.
4. Calculate link reliability scores using the sampled partitions.
5. Evaluate the hidden positive edges against all 25,752 originally unobserved node pairs.
6. Repeat for all five folds and summarize the results.

The SBM and Metropolis sampling are rerun independently for each training fold to prevent test-edge information from leaking into model fitting.

### Evaluation Metrics

- **ROC-AUC:** Measures how well the model distinguishes hidden observed links from originally unobserved pairs.
- **Average Precision (AP):** Summarizes precision-recall performance, particularly relevant for sparse networks.

Additional subgroup evaluations include within-phylum, across-phylum, human-related, and non-human links.

## 3. Implementation

### `cross_validation.py`

The script supports three execution modes:

| Mode | Initial Steps | Metropolis Steps | Purpose |
|---|---:|---:|---|
| `split` | — | — | Verify the five-fold edge splitting |
| `quick` | 50 | 1,000 | Test the complete pipeline |
| `full` | 3,000 | 300,000 | Run the full evaluation |

Current parameters:

- Number of folds: 5
- SBM groups: 16
- NMI threshold: 0.95
- Random seed: 42

## 4. How to Run

Run from the project root:

```bash
python cross_validation/cross_validation.py --mode split
python cross_validation/cross_validation.py --mode quick
python cross_validation/cross_validation.py --mode full
```

Dependencies are listed in the project's root `requirements.txt`.

## 5. Outputs

Results are saved in `cross_validation/results/`.

| File | Description |
|---|---|
| `fold_metrics.csv` | ROC-AUC, AP, and subgroup metrics for each fold |
| `test_predictions.csv` | Test pairs, taxon metadata, true labels, and reliability scores |
| `roc_curves.png` | ROC curves for all five folds |
| `pr_curves.png` | Precision-recall curves for all five folds |

## 6. Preliminary Results

The full evaluation with 300,000 Metropolis steps per fold produced:

| Metric | Mean | Standard Deviation |
|---|---:|---:|
| ROC-AUC | 0.8804 | 0.0058 |
| Average Precision | 0.1239 | 0.0064 |

These results suggest that the SBM can rank hidden observed links above unobserved pairs substantially better than random ranking.

Further analysis of phylum-specific and human-related prediction performance will be added.

## 7. Limitations

- Originally unobserved links are treated as negative samples for evaluation, although they are not necessarily confirmed biological non-interactions.
- NMI-based partition retention may influence the Monte Carlo approximation of link reliability.
- The current number of SBM groups (K = 16) has not yet been optimized through cross-validation.
- Hidden test edges are removed from the training adjacency matrix and treated as absent during SBM fitting, rather than explicitly masked as unknown.

## 8. Related Modules

This implementation reuses the existing project modules:

- `sbm/sbm_weights.py`
- `metropolis_sorin/metropolis.py`
- `reliability_score/Link_reliability.py`

The original SBM, Metropolis, and link reliability implementations are not modified.