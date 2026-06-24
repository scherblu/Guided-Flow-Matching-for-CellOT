import torch
import ot
import numpy as np

def compute_w2(x0, x1):
    """
    Compute the Wasserstein-2 distance between two distributions
    Args:
        x0: Tensor, shape (bs, dim)
            represents the source minibatch
        x1: Tensor, shape (bs, dim)
            represents the target minibatch
    Returns:
        w2: float
            the Wasserstein-2 distance between x0 and x1
    """
   # Compute the cost matrix (squared Euclidean distance)
    M = ot.dist(x0.detach().cpu().numpy(), x1.detach().cpu().numpy(), metric='sqeuclidean')

    # Compute the W2 distance
    W2_distance = np.sqrt(ot.emd2([], [], M))
    return W2_distance


def _finite_rows(x):
    """Rows of x with no inf/nan in any coordinate (the only ones a metric can score)."""
    return x[torch.isfinite(x).all(dim=-1)]


def safe_w2(samples, truth):
    """W2 to truth over the finite samples, plus the non-finite fraction.

    A diverged method produces inf/nan rows that make POT's LP unbounded and silently
    return 0 -- reading that as 'perfect' is the trap `compute_w2` falls into. Score W2 on
    the finite rows only and report how many were dropped.

    Args:
        samples: Tensor (n, dim) -- a method's generated samples (may contain inf/nan).
        truth:   Tensor (m, dim) -- ground-truth reference samples (assumed finite).
    Returns:
        (w2, nonfinite_frac): w2 is float('inf') if <2 finite rows survive.
    """
    finite = _finite_rows(samples)
    nonfinite_frac = 1.0 - finite.shape[0] / samples.shape[0]
    if finite.shape[0] < 2:
        return float("inf"), nonfinite_frac
    return compute_w2(finite, truth), nonfinite_frac


def diversity_trace_ratio(samples, truth):
    """trace(cov(pred)) / trace(cov(truth)).

    ~1 = well-calibrated spread; << 1 = over-collapse (CFG at high w, energy guidance at
    high lambda); >> 1 = over-dispersion. Computed over the finite rows only.

    Args:
        samples: Tensor (n, dim) -- generated samples (may contain inf/nan).
        truth:   Tensor (m, dim) -- ground-truth reference samples.
    Returns:
        ratio: float (0.0 if <2 finite rows survive).
    """
    finite = _finite_rows(samples)
    if finite.shape[0] < 2:
        return 0.0
    return (finite.var(0).sum() / truth.var(0).sum()).item()


def _mean_pairwise_dist(a, b):
    """Mean Euclidean distance over all pairs (V-statistic; a==b gives the within-set term)."""
    return torch.cdist(a, b).mean()


def energy_distance(pred, ref):
    """Szekely-Rizzo energy distance:  2*E||X-Y|| - E||X-X'|| - E||Y-Y'||.

    The single-cell-standard aggregate distribution distance (the scPerturb 'E-distance'),
    so the 2D toy phase diagram and the 4i phase share ONE headline metric. >= 0 always, and
    the V-statistic form makes it exactly 0 for an identical set. Scored over finite rows.
    O(n*m + n^2 + m^2) in memory via pairwise distances -- subsample large inputs first.

    Args:
        pred: Tensor (n, dim) -- generated samples (may contain inf/nan).
        ref:  Tensor (m, dim) -- reference samples (the tilted target in the toy).
    Returns:
        edist: float (inf if <2 finite pred rows survive).
    """
    x = _finite_rows(pred)
    y = ref
    if x.shape[0] < 2 or y.shape[0] < 2:
        return float("inf")
    d_xy = _mean_pairwise_dist(x, y)
    d_xx = _mean_pairwise_dist(x, x)
    d_yy = _mean_pairwise_dist(y, y)
    return (2 * d_xy - d_xx - d_yy).clamp(min=0.0).item()


def energy_test(pred, ref, n_permutations=200, seed=0):
    """Permutation E-test for ``energy_distance``: is pred drawn from ref's distribution?

    Pools the two finite sets, builds the pooled pairwise-distance matrix ONCE, then for each
    permutation re-partitions it into two groups of the original sizes and recomputes the
    energy distance by indexing submatrices (cheap -- no recomputed distances). The p-value
    is the fraction of permutations whose energy distance >= the observed one.

    Args:
        pred, ref: as in ``energy_distance``.
        n_permutations: number of random relabelings (>= 1).
        seed: RNG seed for reproducible permutations.
    Returns:
        (edist, pvalue): observed energy distance and its permutation p-value
        (both nan/inf-guarded: returns (inf, nan) if a set has <2 finite rows).
    """
    x = _finite_rows(pred)
    y = ref
    if x.shape[0] < 2 or y.shape[0] < 2:
        return float("inf"), float("nan")
    n = x.shape[0]
    pooled = torch.cat([x, y], dim=0)
    D = torch.cdist(pooled, pooled)
    N = D.shape[0]

    def ed_from_idx(idx_x, idx_y):
        d_xy = D[idx_x][:, idx_y].mean()
        d_xx = D[idx_x][:, idx_x].mean()
        d_yy = D[idx_y][:, idx_y].mean()
        return (2 * d_xy - d_xx - d_yy)

    full = torch.arange(N)
    observed = ed_from_idx(full[:n], full[n:])
    g = torch.Generator().manual_seed(seed)
    ge = 1  # count the observed labeling itself (standard +1 smoothing)
    for _ in range(n_permutations):
        perm = torch.randperm(N, generator=g)
        if ed_from_idx(perm[:n], perm[n:]) >= observed:
            ge += 1
    return observed.clamp(min=0.0).item(), ge / (n_permutations + 1)


def off_manifold_rate(pred, ref, q=0.99):
    """Fraction of pred that lands off the reference manifold (the 'fliers' / scatter term).

    Threshold = the q-quantile of ref's intra-set nearest-neighbour distances. A pred point
    is off-manifold if its nearest-neighbour distance to ref exceeds that threshold.
    Non-finite (diverged) pred rows are the ultimate fliers and are counted as off-manifold.

    Args:
        pred: Tensor (n, dim) -- generated samples (may contain inf/nan).
        ref:  Tensor (m, dim) -- reference samples defining the manifold.
        q: quantile of ref intra-NN distance used as the on-manifold cutoff (default 0.99).
    Returns:
        rate: float in [0, 1] (1.0 if no finite pred rows or ref too small to define a NN).
    """
    n_total = pred.shape[0]
    if n_total == 0 or ref.shape[0] < 2:
        return 1.0
    finite = _finite_rows(pred)
    n_nonfinite = n_total - finite.shape[0]
    if finite.shape[0] == 0:
        return 1.0
    d_ref = torch.cdist(ref, ref).fill_diagonal_(float("inf"))
    thresh = torch.quantile(d_ref.min(dim=1).values, q)
    pred_nn = torch.cdist(finite, ref).min(dim=1).values
    off = int((pred_nn > thresh).sum().item()) + n_nonfinite
    return off / n_total


def evaluate_samples(samples, truth, edist=False, off_manifold=False):
    """Bundle the per-sample-set metrics so drivers/notebooks score every cell consistently.

    Always returns W2 + diversity-trace + non-finite fraction (the cheap Phase-1 gate set).
    The phase diagram opts into the O(n^2) energy distance (its headline) and the
    off-manifold rate via the flags -- kept off by default so the toy sweeps stay fast.

    Returns:
        dict with keys 'w2', 'nonfinite', 'div_trace' (+ 'edist', 'off_manifold' if enabled).
    """
    w2, nonfinite = safe_w2(samples, truth)
    out = {"w2": w2, "nonfinite": nonfinite, "div_trace": diversity_trace_ratio(samples, truth)}
    if edist:
        out["edist"] = energy_distance(samples, truth)
    if off_manifold:
        out["off_manifold"] = off_manifold_rate(samples, truth)
    return out
