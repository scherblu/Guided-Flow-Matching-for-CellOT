"""Diagnose the GO/NO-GO spike's control-column gap (spec sec.7.6).

At independent coupling (P=1) g^MC must match the oracle floor -- if it doesn't, the
harness is suspect, not the science. The spike leaves it at ~2.6x floor (W2 0.199 vs 0.076).
This isolates the three candidate causes by sweeping ONE knob at a time off a baseline,
all on the cached *independent* model only (cheap -- no retraining, no exact-OT column):

  - MC_BATCH (B)   : finite-B ratio bias + estimator variance
  - ot_std (kernel): the MC kernel bandwidth (model trains at sigma=0, so this is a pure
                     estimator choice -- too-wide a kernel smears the tilted target)
  - num_steps      : Euler discretization error

Run from synthetic/:  ../.pixi/envs/default/bin/python script/debug_control.py
"""
import math
import os
import sys

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from guided_flow.distributions.base import get_distribution
from guided_flow.utils.metrics import compute_w2, diversity_trace_ratio, energy_distance, safe_w2
from guided_flow.utils.misc import deterministic
from guided_flow.utils.oracle import rejection_sample_tilted

import go_nogo_spike as spike

DEVICE = "cpu"
N_SAMPLE = 2000
SEEDS = (0, 1)

# baseline = the spike's current control-column config
BASE = dict(mc_batch=1000, ot_std=0.1, num_steps=100)


def _mean_ci(vals):
    """mean, 95% CI half-width over a list of scalars."""
    a = [v for v in vals]
    n = len(a)
    mean = sum(a) / n
    if n < 2:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in a) / (n - 1)
    return mean, 1.96 * math.sqrt(var) / math.sqrt(n)


def sample_mc(model, x0_dist, x1_dist, mc_batch, ot_std, num_steps, n, seed):
    """g^MC Euler integration with explicit knobs (mirrors spike.sample_guided)."""
    deterministic(seed)
    guide_fn = spike.get_mc_guide_fn(x0_dist, x1_dist, spike.SCALE, mc_batch, n, ot_std=ot_std)
    x = x0_dist.sample(n).to(DEVICE)
    t_span = torch.linspace(0, spike.T_END, num_steps)
    for k in range(num_steps - 1):
        t = t_span[k]
        dt = t_span[k + 1] - t_span[k]
        with torch.no_grad():
            v = model(torch.cat([x, t.expand(x.shape[0], 1)], dim=-1))
        g = guide_fn(t, x, v, model)
        x = x + dt * (v + g)
    return x.detach()


def score(model, x0_dist, x1_dist, truth, label, n=N_SAMPLE, seeds=SEEDS, **cfg):
    """Score g^MC at one config in BOTH W2 and energy distance (edist is the phase-diagram
    headline; W2 is the spike's companion). Returns (w2_mean, edist_mean)."""
    w2s, eds, dtrs, nfs = [], [], [], []
    for s in seeds:
        samp = sample_mc(model, x0_dist, x1_dist, n=n, seed=s, **cfg)
        w2, nf = safe_w2(samp, truth)
        w2s.append(w2)
        eds.append(energy_distance(samp, truth))
        nfs.append(nf)
        dtrs.append(diversity_trace_ratio(samp, truth))
    w2m, w2ci = _mean_ci(w2s)
    edm, edci = _mean_ci(eds)
    dtm = sum(dtrs) / len(dtrs)
    print(f"  {label:32s}  W2={w2m:.4f}+/-{w2ci:.3f}  edist={edm:.4f}+/-{edci:.3f}  "
          f"div={dtm:.2f}  nonfinite={max(nfs):.0%}")
    return w2m, edm


def floors(x1_dist, k=5):
    """Both-metric oracle floor with a CI: W2/edist between independent oracle draws."""
    w2s, eds = [], []
    for i in range(k):
        a = rejection_sample_tilted(x1_dist, N_SAMPLE, scale=spike.SCALE, device=DEVICE)
        b = rejection_sample_tilted(x1_dist, N_SAMPLE, scale=spike.SCALE, device=DEVICE)
        w2s.append(compute_w2(a, b))
        eds.append(energy_distance(a, b))
    return _mean_ci(w2s), _mean_ci(eds)


def main():
    deterministic(0)
    x0_dist = get_distribution("uniform")
    x1_dist = get_distribution("8gaussian")
    x1_dist.set_energy(spike.ENERGY_FN, target_indices=spike.TARGET_BLOBS, std_target=spike.SHARP_W)

    truth = rejection_sample_tilted(x1_dist, N_SAMPLE, scale=spike.SCALE, device=DEVICE)
    (w2_floor, w2_floor_ci), (ed_floor, ed_floor_ci) = floors(x1_dist)
    print(f"\noracle floor (n={N_SAMPLE}, 5 draws):  W2={w2_floor:.4f}+/-{w2_floor_ci:.3f}   "
          f"edist={ed_floor:.4f}+/-{ed_floor_ci:.3f}\n")

    # independent (control) model only -- it is cached from the spike
    model = spike.train_vf("cfm", x0_dist, x1_dist)

    # --- the decisive test: g^MC control at the baseline config, 5 seeds, both metrics ---
    print(f"control verdict -- g^MC at P=1, baseline config {BASE}, 5 seeds:")
    score(model, x0_dist, x1_dist, truth, "g^MC (control)", seeds=range(5), **BASE)
    print(f"  (W2 floor {w2_floor:.3f}, edist floor {ed_floor:.3f}) -- "
          f"harness is valid if edist control overlaps the edist floor.\n")

    # --- the memory's hypothesis: does more MC_BATCH tighten it? (it does not) ---
    print("[1] MC_BATCH sweep (tests the 'tighten with larger MC_BATCH' hypothesis):")
    for B in (1000, 2000, 4000):
        score(model, x0_dist, x1_dist, truth, f"mc_batch={B}", **{**BASE, "mc_batch": B})

    print(f"\n(goal: control edist toward floor {ed_floor:.3f}; W2 floor {w2_floor:.3f})")


if __name__ == "__main__":
    main()
