"""Ground-truth oracle for the tilted target p~1(x) proportional to p1(x) * exp(-J(x)).

In the 2D toy both p1 (e.g. the 8-Gaussian mixture) and the energy J are known, so we
can sample the analytic tilted target *exactly* by rejection sampling -- no estimator,
coupling-independent and unimpeachable. This is the oracle the guidance methods are
scored against (spec sec.3 / sec.7.4).
"""
import torch

from ..distributions.base import BaseDistribution


def rejection_sample_tilted(x1_dist: BaseDistribution, n: int, scale: float = 1.0,
                            device: str = 'cpu', max_iter: int = 10000) -> torch.Tensor:
    """Rejection-sample the tilted target  p~1 proportional to  p1 * exp(-scale * J).

    Proposes from p1 (``x1_dist.sample``) and accepts each proposal with probability
    exp(-scale * J(x)) / max(exp(-scale * J)) over the current proposal batch. Because the
    acceptance weight is bounded by 1, this draws exact samples from p~1.

    Args:
        x1_dist: target distribution with an energy configured (``get_J``) and ``sample``.
        n: number of accepted samples to return.
        scale: tilt strength multiplying J (matches the guidance ``scale``).
        device: device for the returned tensor.
        max_iter: safety cap on proposal rounds.

    Returns:
        Tensor (n, dim) of accepted samples from the tilted target.
    """
    accepted = None
    for _ in range(max_iter):
        proposals = x1_dist.sample(n).to(device)
        weights = torch.exp(-scale * x1_dist.get_J(proposals))
        acc_prob = weights / weights.max()
        keep = proposals[torch.rand(n, device=device) < acc_prob]
        accepted = keep if accepted is None else torch.cat([accepted, keep], 0)
        if accepted.shape[0] >= n:
            return accepted[:n]
    raise RuntimeError(
        f"rejection_sample_tilted: only {0 if accepted is None else accepted.shape[0]}/{n} "
        f"samples after {max_iter} rounds -- energy may be too sharp for this proposal."
    )
