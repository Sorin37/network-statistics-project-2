"""Scores a grouping of the taxa under the stochastic block model (SBM).

Given the food web and one way of sorting the taxa into groups, this file
returns a single number saying how well that grouping explains the observed
links, where higher is better. The Metropolis algorithm uses this number to decide
whether to accept or reject a proposed change to a grouping.

The score is the weight f(T) from the lecture slides (slide 31):

    f(T) = product over pairs of groups t <= s of
           1 / ((r_ts + 1) * C(r_ts, l_ts))

    l_ts = number of observed links between groups t and s
    r_ts = number of possible links between groups t and s
    C    = binomial coefficient

f(T) is too small to store as a number for a web of 235 taxa, so the
functions here return log f(T) instead.
"""

import numpy as np
from scipy.special import gammaln


def block_counts(A, groups, K):
    """Count the links between every pair of groups.

    For each pair of groups, it calculates how many links were observed
    between them, and how many links would be possible between them.

    Inputs
    A : the food web as a table of 0s and 1s, with one row and one
        column per taxon. A[i, j] is 1 if taxa i and j are linked.
        The graph is undirected and has no self-loops.
    groups : a list with one entry per taxon. groups[i] is the group
        number of taxon i. Group numbers run from 0 to K - 1.
    K : the number of groups allowed. Some of them may be empty.

    Outputs
    L : a K x K table. L[t, s] is the number of links observed
        between group t and group s.
    R : a K x K table. R[t, s] is the number of links possible
        between group t and group s.

    The entries L[t, t] and R[t, t] are for links inside group t.
    Both tables are symmetric, so L[t, s] equals L[s, t].
    """

    # convert to float to multiply faster, then convert back to int
    A = np.asarray(A, dtype=float)
    groups = np.asarray(groups)
    N = A.shape[0]  # number of taxa

    # build a membership table Z, with one row per taxon and one column per group.
    # Z[i, t] is 1 if taxon i is in group t, otherwise 0.
    Z = np.zeros((N, K))
    Z[np.arange(N), groups] = 1

    # Adding up each column of Z gives the number of taxa in each group.
    n = Z.sum(axis=0).astype(np.int64)

    # count the links between groups.
    # A @ Z gives, for each taxon, how many links it has into each group.
    # Multiplying by Z.T then adds those up over all the taxa in a group.
    # M[t, s] is the number of links going from group t to group s.
    M = np.rint(Z.T @ A @ Z).astype(np.int64)

    # L = observed links
    # Between two different groups, M is already the right count.
    # Inside one group, every link was counted twice (once from each end),
    # so the entries on the diagonal are halved.
    diag = np.diag_indices(K)  # the positions [0, 0], [1, 1], ... in a table
    L = M.copy()
    L[diag] = M[diag] // 2

    # R = possible links
    # Between two different groups, every taxon in one could link to every
    # taxon in the other: (size of group t) * (size of group s).
    # Inside one group of size n, the number of pairs is n * (n - 1) / 2.
    R = np.outer(n, n)
    R[diag] = n * (n - 1) // 2

    return L, R


def log_weight_from_counts(L, R):
    """Turn the link counts into the score, log f(T).

    Inputs
    L, R : the two tables returned by block_counts.

    Output
    The score, as a single number.
    """
    # Take each pair of groups once. The tables are symmetric, so only the
    # upper half (including the diagonal) is needed.
    upper = np.triu_indices(L.shape[0])
    l = L[upper].astype(float)  # links observed, one entry per pair of groups
    r = R[upper].astype(float)  # links possible, one entry per pair of groups

    # The log of "r choose l", for every pair of groups at once.
    # "r choose l" is r! / (l! * (r - l)!), and gammaln(x + 1) is log(x!),
    # so its log is log(r!) - log(l!) - log((r - l)!).
    # works for empty groups (r = 0) and for fully linked blocks (l = r).
    log_r_choose_l = gammaln(r + 1) - gammaln(l + 1) - gammaln(r - l + 1)

    # The formula has one term per pair of groups, so add them all up.
    terms = -np.log(r + 1) - log_r_choose_l
    return float(np.sum(terms))


def log_weight(A, groups, K):
    """Score one grouping of the taxa. Returns log f(T).

    Inputs are the same as for block_counts. The output is a single
    negative number. Closer to zero means the grouping explains the
    observed links better.
    """
    L, R = block_counts(A, groups, K)       # step 1: count the links
    return log_weight_from_counts(L, R)     # step 2: apply the formula


def validate_inputs(A, groups, K):
    """Check that the inputs are in the format the other functions expect.

    Stops with an error message if something is wrong.
    Call it once after loading the data.
    Don't call it inside the sampling loop, because the checks are slow.
    """
    A = np.asarray(A)
    groups = np.asarray(groups)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix")
    if not np.isin(A, (0, 1)).all():
        raise ValueError("A must contain only 0 and 1")
    if not (A == A.T).all():
        raise ValueError("A must be symmetric (undirected graph)")
    if A.diagonal().any():
        raise ValueError("A must have a zero diagonal (no self-loops)")
    if groups.shape != (A.shape[0],):
        raise ValueError("groups must have one entry per node")
    if not np.issubdtype(groups.dtype, np.integer):
        raise ValueError("groups must contain integers")
    if groups.min() < 0 or groups.max() >= K:
        raise ValueError("group numbers must be between 0 and K - 1")