"""Tests for sbm/sbm_weights.py. Run from the repository root with:

    python -m pytest tests/sbm_weights_test.py
"""

import math

import numpy as np
import pytest

from sbm.sbm_weights import block_counts, log_weight, validate_inputs


# Build a symmetric 0/1 matrix from a list of (i, j) links.
def adjacency(n, links):
    A = np.zeros((n, n), dtype=int)
    for i, j in links:
        A[i, j] = A[j, i] = 1
    return A


def random_graph(n, p, rng):
    upper = np.triu(rng.random((n, n)) < p, k=1).astype(int)
    return upper + upper.T


# Reference version: plain loops and exact integer binomials.
def slow_log_weight(A, groups, K):
    total = 0.0
    for t in range(K):
        for s in range(t, K):
            in_t = np.where(groups == t)[0]
            in_s = np.where(groups == s)[0]
            if t == s:
                pairs = [(i, j) for a, i in enumerate(in_t) for j in in_t[a + 1:]]
            else:
                pairs = [(i, j) for i in in_t for j in in_s]
            r = len(pairs)
            l = sum(A[i, j] for i, j in pairs)
            total += -math.log(r + 1) - math.log(math.comb(r, l))
    return total


# A graph small enough to check by hand
def test_by_hand():
    # Nodes 0,1 in group 0 and nodes 2,3 in group 1. Links: 0-1, 0-2, 1-2.
    A = adjacency(4, [(0, 1), (0, 2), (1, 2)])
    groups = np.array([0, 0, 1, 1])

    L, R = block_counts(A, groups, K=2)
    assert L.tolist() == [[1, 2], [2, 0]]
    assert R.tolist() == [[1, 4], [4, 1]]

    # f = 1/(2*C(1,1)) * 1/(5*C(4,2)) * 1/(2*C(1,0)) = 1/2 * 1/30 * 1/2 = 1/120
    assert log_weight(A, groups, K=2) == pytest.approx(math.log(1 / 120))


# Empty groups must not crash or change the answer
def test_empty_groups():
    A = adjacency(4, [(0, 1), (0, 2), (1, 2)])
    groups = np.array([0, 0, 1, 1])
    assert log_weight(A, groups, K=6) == pytest.approx(log_weight(A, groups, K=2))



# Agreement with a slow, obviously-correct version
@pytest.mark.parametrize("seed", range(5))
def test_matches_slow_version(seed):
    rng = np.random.default_rng(seed)
    A = random_graph(30, 0.2, rng)
    groups = rng.integers(0, 4, size=30)
    assert log_weight(A, groups, K=4) == pytest.approx(slow_log_weight(A, groups, K=4))


# Properties that must hold for any grouping

def test_counts_add_up():
    rng = np.random.default_rng(0)
    A = random_graph(40, 0.1, rng)
    groups = rng.integers(0, 7, size=40)
    L, R = block_counts(A, groups, K=7)
    upper = np.triu_indices(7)
    assert L[upper].sum() == A.sum() // 2      # every link counted exactly once
    assert R[upper].sum() == 40 * 39 // 2      # every pair counted exactly once


def test_group_numbers_do_not_matter():
    rng = np.random.default_rng(1)
    A = random_graph(25, 0.2, rng)
    groups = rng.integers(0, 3, size=25)
    renamed = np.array([2, 0, 1])[groups]      # same grouping, different labels
    assert log_weight(A, groups, K=3) == pytest.approx(log_weight(A, renamed, K=3))


def test_validate_inputs_rejects_bad_input():
    A = adjacency(3, [(0, 1)])
    validate_inputs(A, np.array([0, 1, 1]), K=2)           # fine
    with pytest.raises(ValueError):
        validate_inputs(A, np.array([0, 1, 2]), K=2)       # group number too large
    B = A.copy(); B[2, 2] = 1
    with pytest.raises(ValueError):
        validate_inputs(B, np.array([0, 1, 1]), K=2)       # self-loop