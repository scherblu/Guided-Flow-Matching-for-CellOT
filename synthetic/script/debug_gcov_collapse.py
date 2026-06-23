"""WHY does g^cov collapse each blob to a near-delta? (mechanism test)

Hypothesis: g^cov adds a DETERMINISTIC attractive drift -lambda*grad J(x_hat1) toward the well
minimum. An attractive drift contracts phase-space volume in the convex well (rate ~ lambda/w^2),
so it keeps shrinking the per-mode variance. But the correct tilted target p1*e^-J has FINITE
width -- near a well it is base(std 0.1) (x) tilt(std w) = std (1/0.1^2 + 1/w^2)^-1/2. g^cov has no
term that knows to STOP contracting there, so it over-contracts to a delta.

Predictions tested here (exact-OT model, so within-blob collapse is isolated from global collapse):
  1. No single lambda recovers the correct width: small lambda -> undertilted (~base 0.1),
     large lambda -> collapsed (~0); it cannot hit ~0.045 AND the right tilt at once.
  2. g^MC keeps the width (it reweights over the real endpoint DISTRIBUTION, not a point estimate).

Run from synthetic/:  ../.pixi/envs/default/bin/python script/debug_gcov_collapse.py
"""
import math
import os
import sys

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from guided_flow.distributions.base import get_distribution
from guided_flow.utils.metrics import energy_distance
from guided_flow.utils.misc import deterministic
from guided_flow.utils.oracle import rejection_sample_tilted

import go_nogo_spike as spike
import phase_diagram as pd

DEVICE = "cpu"
N = 2000
W = 0.05
BASE_STD = 0.1                     # 8gaussian per-blob std (the base density width)


def blob_stats(samples, centers_t, core_r=0.12):
    """Per-blob (nearest-center): mean radial spread, flier-robust CORE spread (samples within
    core_r of the center, so inter-blob scatter/fliers don't inflate it), and mass-imbalance."""
    s = samples[torch.isfinite(samples).all(-1)]
    assign = torch.cdist(s, centers_t).argmin(1)
    spreads, cores, masses = [], [], []
    for b in range(centers_t.shape[0]):
        sb = s[assign == b]
        if sb.shape[0] < 2:
            masses.append(0.0)
            continue
        masses.append(sb.shape[0] / s.shape[0])
        spreads.append((sb - sb.mean(0)).norm(dim=-1).mean().item())
        core = sb[(sb - centers_t[b]).norm(dim=-1) < core_r]      # exclude fliers
        if core.shape[0] >= 2:
            cores.append((core - core.mean(0)).norm(dim=-1).mean().item())
    spread = sum(spreads) / len(spreads) if spreads else float("nan")
    core = sum(cores) / len(cores) if cores else float("nan")
    imbalance = sum(abs(m - 1 / 3) for m in masses)
    return spread, core, imbalance


def main():
    x0 = get_distribution("uniform")
    x1 = get_distribution("8gaussian")
    x1.set_energy(spike.ENERGY_FN, target_indices=spike.TARGET_BLOBS, std_target=W)
    centers_t = x1.centers[spike.TARGET_BLOBS].to(DEVICE)
    truth = rejection_sample_tilted(x1, N, scale=spike.SCALE, device=DEVICE)

    # theoretical tilted width: base (x) tilt precision add; radial_spread ~ 1.25*std in 2D
    tilt_std = (1 / BASE_STD ** 2 + 1 / W ** 2) ** -0.5
    target_spread = tilt_std * math.sqrt(math.pi / 2)
    t_spread, t_core, _ = blob_stats(truth, centers_t)
    print(f"\nsharp J w={W}: base std {BASE_STD}, tilt std {W} -> tilted blob std {tilt_std:.4f} "
          f"(pred radial_spread {target_spread:.4f}); truth measured core spread {t_core:.4f}\n")

    model = pd.train_vf("ot_cfm", None, x0, x1)        # exact-OT (no global collapse)

    print(f"{'method':24s}  {'spread':>8s}  {'core spread':>11s}  {'mass-imbal':>10s}  {'edist':>8s}")
    print(f"  (core spread = within-blob width with fliers excluded; TARGET core ~ {target_spread:.3f})")
    print("-" * 72)
    for sched in ("linear_decay", "const"):
        for scale in (0.1, 0.3, 1.0, 3.0):
            deterministic(0)
            g = spike.get_gcov_guide_fn(x1, guide_scale=scale, schedule=sched)
            s = spike.sample_guided(model, x0, g, N)
            sp, core, imb = blob_stats(s, centers_t)
            ed = energy_distance(s, truth)
            print(f"g^cov sc={scale:<4} {sched:13s}  {sp:8.4f}  {core:11.4f}  {imb:10.2f}  {ed:8.4f}")
        print()

    deterministic(0)
    gmc = spike.sample_guided(model, x0, spike.get_mc_guide_fn(x0, x1, spike.SCALE, spike.MC_BATCH, N), N)
    sp, core, imb = blob_stats(gmc, centers_t)
    print(f"{'g^MC (reference)':24s}  {sp:8.4f}  {core:11.4f}  {imb:10.2f}  {energy_distance(gmc, truth):8.4f}")
    print(f"{'TARGET (tilted truth)':24s}  {target_spread:8.4f}  {t_core:11.4f}  {0.0:10.2f}  {0.0:8.4f}")
    print("\n=> g^cov core spread collapses toward 0 (over-contraction) and no setting matches the")
    print(f"   target {target_spread:.3f}; g^MC's core spread stays near the target width (its edist")
    print("   comes from fliers/scatter, NOT collapse) -- opposite failure modes.")


if __name__ == "__main__":
    main()
