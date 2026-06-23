"""Verify the entropic-OT coupling axis (IMPLEMENTATION_SPEC_pivot.md sec.2/7.2).

The spec is explicit: "Don't assume the sign of the effect -- measure it." We need
reg -> infinity to approach the INDEPENDENT coupling and reg -> 0 to approach EXACT OT,
because that is the phase diagram's x-axis.

Cheap training-free proxy: the mean squared transport cost  E||x1 - x0||^2  under the
sampled coupling. The independent (product) coupling pays the full cost; exact OT pays the
minimum; the entropic plan must interpolate monotonically between them as reg decreases.
Because the OT-CFM target field is u_t = x1 - x0 averaged over the coupling, matching the
coupling's cost distribution means the trained VF matches too -- so this is a sufficient,
fast stand-in for "reg->inf approx independent VF".

Run from synthetic/:  ../.pixi/envs/default/bin/python script/test_entropic_coupling.py
"""
import os
import sys

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from guided_flow.distributions.base import get_distribution
from guided_flow.flow.conditional_flow_matching import (
    ConditionalFlowMatcher,
    EntropicOTConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
    get_cfm,
)
from guided_flow.utils.misc import deterministic

DEVICE = "cpu"
BS = 256
N_BATCH = 40          # average the cost over many minibatches to denoise
REGS = (10.0, 1.0, 0.3, 0.1, 0.03, 0.01)


def mean_coupling_cost(matcher, x0_dist, x1_dist, n_batch=N_BATCH, bs=BS, seed=0):
    """Mean ||x1 - x0||^2 of the (x0, x1) pairs the matcher couples, over n_batch minibatches.

    `ConditionalFlowMatcher` (independent) pairs the raw batches; the OT matchers re-pair via
    `ot_sampler.sample_plan`. We replicate the same coupling step the matcher uses at train.
    """
    deterministic(seed)
    costs = []
    for _ in range(n_batch):
        x0 = x0_dist.sample(bs, device=DEVICE)
        x1 = x1_dist.sample(bs, device=DEVICE)
        if hasattr(matcher, "ot_sampler"):
            x0, x1 = matcher.ot_sampler.sample_plan(x0, x1)
        costs.append(((x1 - x0) ** 2).sum(-1).mean().item())
    return sum(costs) / len(costs)


def main():
    x0_dist = get_distribution("uniform")
    x1_dist = get_distribution("8gaussian")

    indep = mean_coupling_cost(ConditionalFlowMatcher(sigma=0.0), x0_dist, x1_dist)
    exact = mean_coupling_cost(ExactOptimalTransportConditionalFlowMatcher(sigma=0.0), x0_dist, x1_dist)
    print(f"independent coupling cost = {indep:.4f}  (upper anchor)")
    print(f"exact-OT    coupling cost = {exact:.4f}  (lower anchor)\n")

    print("entropic-OT cost vs reg (expect: large reg -> independent, small reg -> exact):")
    costs = {}
    for reg in REGS:
        c = mean_coupling_cost(EntropicOTConditionalFlowMatcher(reg=reg, warn=False), x0_dist, x1_dist)
        costs[reg] = c
        # fraction of the way from exact (0) to independent (1)
        frac = (c - exact) / (indep - exact)
        print(f"  reg={reg:6.2f}  cost={c:.4f}  ({frac:5.0%} of the way to independent)")

    # get_cfm wiring works
    vianame = mean_coupling_cost(get_cfm("entropic_ot_cfm", sigma=0.0, reg=1.0), x0_dist, x1_dist)

    # ---- assertions: the sign + monotonicity of the effect ----
    PASS, FAIL = 0, 0

    def check(name, cond, detail=""):
        nonlocal PASS, FAIL
        ok = bool(cond)
        PASS += ok
        FAIL += not ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))

    print("\nchecks:")
    check("exact-OT cost < independent cost", exact < indep, f"{exact:.3f} < {indep:.3f}")
    cost_list = [costs[r] for r in REGS]                         # REGS is large -> small
    check("cost decreases monotonically as reg shrinks",
          all(cost_list[i] >= cost_list[i + 1] - 1e-3 for i in range(len(cost_list) - 1)),
          " -> ".join(f"{c:.2f}" for c in cost_list))
    check("largest reg approx independent (>=80% of the way)",
          (costs[REGS[0]] - exact) / (indep - exact) > 0.8, f"{costs[REGS[0]]:.3f} vs indep {indep:.3f}")
    check("smallest reg approx exact OT (<=25% of the way)",
          (costs[REGS[-1]] - exact) / (indep - exact) < 0.25, f"{costs[REGS[-1]]:.3f} vs exact {exact:.3f}")
    check("get_cfm('entropic_ot_cfm', reg=1.0) matches direct instance",
          abs(vianame - costs[1.0]) < 0.05 * indep, f"{vianame:.3f} vs {costs[1.0]:.3f}")

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
