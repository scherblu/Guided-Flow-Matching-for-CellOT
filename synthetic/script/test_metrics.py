"""Unit checks for the phase-diagram metrics (IMPLEMENTATION_SPEC_pivot.md sec.4).

Lightweight, self-contained (synthetic Gaussians -- no model/oracle needed). Verifies the
sanity properties the spec calls for: identical sets -> energy distance 0 / div-ratio 1,
a shifted set -> a known closed-form energy distance, off-manifold and E-test extremes, and
the non-finite handling the harness depends on.

Run from synthetic/:  ../.pixi/envs/default/bin/python script/test_metrics.py
"""
import math
import os
import sys

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from guided_flow.utils.metrics import (
    diversity_trace_ratio,
    energy_distance,
    energy_test,
    off_manifold_rate,
    safe_w2,
)

torch.manual_seed(0)
PASS, FAIL = 0, 0


def check(name, cond, detail=""):
    global PASS, FAIL
    ok = bool(cond)
    PASS += ok
    FAIL += not ok
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


# ---- fixtures ----
X = torch.randn(2000, 2)
Y = torch.randn(2000, 2)                  # same distribution, different draw
SHIFT = torch.tensor([10.0, 0.0])
Xshift = X + SHIFT                          # far-shifted copy

print("energy_distance:")
ed_same = energy_distance(X, X)
check("identical set -> 0", abs(ed_same) < 1e-5, f"got {ed_same:.2e}")
ed_floor = energy_distance(X, Y)
check("same distribution -> small (sampling floor)", ed_floor < 0.1, f"got {ed_floor:.4f}")
# For two copies translated by a vector of length L >> spread, 2*E||X-Y|| - E||X-X'|| - E||Y-Y'||
# -> 2*L - 2*E||X-X'|| ; with L=10 and unit-Gaussian within-set mean ~1.13, expect ~ 2*(10-1.13).
ed_shift = energy_distance(X, Xshift)
within = torch.cdist(X, X).mean().item()
expected = 2 * (SHIFT.norm().item() - within)
check("far-shifted -> ~2*(L - within)", abs(ed_shift - expected) < 0.5,
      f"got {ed_shift:.3f}, expected ~{expected:.3f}")
check("far-shifted >> floor", ed_shift > 50 * ed_floor, f"{ed_shift:.3f} vs {ed_floor:.4f}")

print("diversity_trace_ratio:")
check("identical set -> 1", abs(diversity_trace_ratio(X, X) - 1.0) < 1e-5)
check("half-variance set -> ~0.25", abs(diversity_trace_ratio(0.5 * X, X) - 0.25) < 1e-3,
      f"got {diversity_trace_ratio(0.5 * X, X):.4f}")

print("off_manifold_rate:")
omr_same = off_manifold_rate(X, Y, q=0.99)
check("same distribution -> small", omr_same < 0.05, f"got {omr_same:.3f}")
omr_far = off_manifold_rate(Xshift, X, q=0.99)
check("far-shifted -> ~1", omr_far > 0.95, f"got {omr_far:.3f}")

print("energy_test (permutation p-value):")
_, p_same = energy_test(X, Y, n_permutations=200, seed=0)
check("same distribution -> large p", p_same > 0.1, f"got {p_same:.3f}")
_, p_diff = energy_test(X, Xshift, n_permutations=200, seed=0)
check("different distribution -> small p", p_diff < 0.05, f"got {p_diff:.3f}")

print("non-finite handling:")
Xnan = X.clone()
Xnan[:200] = float("inf")                   # 10% diverged rows
w2, nf = safe_w2(Xnan, Y)
check("safe_w2 reports non-finite fraction", abs(nf - 0.1) < 1e-6, f"got {nf:.3f}")
check("energy_distance ignores non-finite rows", math.isfinite(energy_distance(Xnan, Y)))
omr_nan = off_manifold_rate(Xnan, Y, q=0.99)
check("off_manifold counts non-finite as off (>= their fraction)", omr_nan >= 0.1 - 1e-6,
      f"got {omr_nan:.3f}")
check("all-non-finite -> inf / rate 1",
      energy_distance(torch.full((100, 2), float("inf")), Y) == float("inf")
      and off_manifold_rate(torch.full((100, 2), float("inf")), Y) == 1.0)

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
