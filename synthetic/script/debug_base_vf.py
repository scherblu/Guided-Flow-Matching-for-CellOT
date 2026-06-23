"""Why does g^cov vary across the coupling axis? Check the unguided base VF per coupling.

g^cov (additive -lambda*grad J at the one-step endpoint x_hat1 = x + (1-t)*v) relies on x_hat1
being an accurate endpoint prediction -- which holds only when the flow is ~straight. Curved /
uncertain flows (weak coupling) give a wrong x_hat1, so grad J points the wrong way and the
MAP-seeking pull collapses the samples. This measures, per coupling, the UNGUIDED base VF's own
error to its target (untilted 8gaussian) + a straightness proxy, so g^cov's coupling-dependence
(worst at independent, slight uptick at exact-OT) can be read against base-VF quality.

Straightness proxy: mean ||x1_hat(t=0) - x1_final|| -- how far the one-step endpoint prediction
from the START point is from where the trajectory actually lands. Small = straight = g^cov-friendly.

Run from synthetic/:  ../.pixi/envs/default/bin/python script/debug_base_vf.py
"""
import os
import sys

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from guided_flow.distributions.base import get_distribution
from guided_flow.utils.metrics import diversity_trace_ratio, energy_distance, safe_w2
from guided_flow.utils.misc import deterministic

import go_nogo_spike as spike
import phase_diagram as pd

DEVICE = "cpu"
N = 2000


def unguided_and_straightness(model, x0_dist, n):
    """Sample the model with zero guidance; also measure the t=0 one-step endpoint error."""
    deterministic(0)
    x = x0_dist.sample(n).to(DEVICE)
    # one-step endpoint prediction from the START: x_hat1 = x0 + v(x0,0)*(1-0)
    with torch.no_grad():
        v0 = model(torch.cat([x, torch.zeros(n, 1)], dim=-1))
    x1_hat0 = x + v0
    # integrate to the real endpoint (same Euler loop as the phase diagram, zero guide)
    zero = lambda t, xx, vv, m: torch.zeros_like(xx)
    x1 = spike.sample_guided(model, x0_dist, zero, n)
    straightness = (x1_hat0 - x1).norm(dim=-1).mean().item()
    return x1, straightness


def main():
    x0 = get_distribution("uniform")
    x1d = get_distribution("8gaussian")

    # untilted 8gaussian floor (the base VF's actual target)
    floor = sum(energy_distance(x1d.sample(N).to(DEVICE), x1d.sample(N).to(DEVICE)) for _ in range(5)) / 5
    print(f"\nuntilted 8gaussian edist floor (5 draws) = {floor:.4f}\n")
    print(f"{'coupling':12s}  {'base edist':>10s}  {'x floor':>8s}  {'base W2':>8s}  "
          f"{'base div':>8s}  {'endpoint err':>12s}")
    print("-" * 70)

    for label, cfm, reg in pd.COUPLINGS:
        model = pd.train_vf(cfm, reg, x0, x1d)          # cached -- no retraining
        base, straight = unguided_and_straightness(model, x0, N)
        ref = x1d.sample(N).to(DEVICE)
        ed = energy_distance(base, ref)
        w2, _ = safe_w2(base, ref)
        div = diversity_trace_ratio(base, ref)
        print(f"{label:12s}  {ed:10.4f}  {ed/floor:8.1f}  {w2:8.3f}  {div:8.2f}  {straight:12.3f}")

    print("\nRead against the g^cov grid: g^cov should track base-VF quality + straightness "
          "(small endpoint err = g^cov-friendly).")


if __name__ == "__main__":
    main()
