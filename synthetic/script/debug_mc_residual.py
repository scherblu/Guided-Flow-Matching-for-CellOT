"""Is the g^MC control residual the MC estimator's fault, or the base VF's? (P=1 control)

At independent coupling g^MC should hit the oracle floor but sits ~10x it in energy distance.
Decompose the residual with a clean reference that has NO MC-estimator and NO ODE-guidance
error -- importance-resampling the base model's own unguided samples by exp(-scale*J):

  base            = unguided samples from the cfm model       (vs UNTILTED 8gaussian = its target)
  base_reweighted = base resampled with weights exp(-scale*J) (reference ONLY -- see caveat)
  g^MC            = the actual guided control                 (vs TILTED truth)

Result (2026-06-22): in ABSOLUTE energy distance g^MC (0.015) ~= the base VF's own error to its
untilted target (0.014) -- the MC guidance adds almost nothing on top of the base model, so the
control residual is BASE-VF training error, not MC-estimator error. The big "x floor" multiple is
inflated by the sharp tilted target's tiny floor (0.0009).

CAVEAT: base_reweighted is NOT a valid "optimum" -- importance-reweighting can only select existing
base samples (std ~0.1), it cannot SHARPEN them to the tilted target's std ~0.045, so it scores
WORSE (0.041) than active guidance. It is reported for reference, not as the bound it was meant to be.

Run from synthetic/:  ../.pixi/envs/default/bin/python script/debug_mc_residual.py
"""
import os
import sys

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from guided_flow.distributions.base import get_distribution
from guided_flow.utils.metrics import energy_distance, safe_w2
from guided_flow.utils.misc import deterministic
from guided_flow.utils.oracle import rejection_sample_tilted

import go_nogo_spike as spike

DEVICE = "cpu"
N = 2000


def floor_edist(draw, k=5):
    return sum(energy_distance(draw(), draw()) for _ in range(k)) / k


def reweight(samples, x1_dist, scale):
    """Importance-resample `samples` by exp(-scale*J): draws from (their dist) * e^{-J}."""
    w = torch.exp(-scale * x1_dist.get_J(samples))
    idx = torch.multinomial(w, samples.shape[0], replacement=True)
    return samples[idx]


def main():
    deterministic(0)
    x0 = get_distribution("uniform")
    x1 = get_distribution("8gaussian")
    x1.set_energy(spike.ENERGY_FN, target_indices=spike.TARGET_BLOBS, std_target=spike.SHARP_W)

    model = spike.train_vf("cfm", x0, x1)   # cached independent (control) model

    # ---- base model error: unguided samples vs the model's OWN target (untilted 8gaussian) ----
    zero_guide = lambda t, x, v, m: torch.zeros_like(x)
    base = spike.sample_guided(model, x0, zero_guide, N)
    f8 = floor_edist(lambda: x1.sample(N).to(DEVICE))
    base_ed = energy_distance(base, x1.sample(N).to(DEVICE))
    base_w2, _ = safe_w2(base, x1.sample(N).to(DEVICE))
    print(f"\nbase VF (unguided) vs UNTILTED 8gaussian:   "
          f"edist={base_ed:.4f} ({base_ed/f8:.1f}x floor {f8:.4f})   W2={base_w2:.3f}")

    # ---- tilted target + its floor ----
    truth = rejection_sample_tilted(x1, N, scale=spike.SCALE, device=DEVICE)
    ft = floor_edist(lambda: rejection_sample_tilted(x1, N, scale=spike.SCALE, device=DEVICE))

    # ---- best-possible tilt from THIS base model (no MC, no ODE guidance) ----
    base_rw = reweight(base, x1, spike.SCALE)
    rw_ed = energy_distance(base_rw, truth)
    print(f"base VF reweighted by e^-J vs TILTED truth: "
          f"edist={rw_ed:.4f} ({rw_ed/ft:.1f}x floor {ft:.4f})   <- base-VF-limited optimum")

    # ---- the actual g^MC control ----
    guide = spike.get_mc_guide_fn(x0, x1, spike.SCALE, spike.MC_BATCH, N)
    mc = spike.sample_guided(model, x0, guide, N)
    mc_ed = energy_distance(mc, truth)
    print(f"g^MC guided                vs TILTED truth: "
          f"edist={mc_ed:.4f} ({mc_ed/ft:.1f}x floor {ft:.4f})   <- actual control")

    print("\n" + "=" * 68)
    print(f"base VF error (abs edist, untilted)  : {base_ed:.4f}")
    print(f"g^MC control error (abs edist, tilted): {mc_ed:.4f}")
    print(f"naive reweight reference (can't sharpen): {rw_ed:.4f}  (worse than g^MC -- see caveat)")
    # the test: is g^MC's absolute error ~ the base VF's own absolute error?
    if mc_ed <= 1.5 * base_ed:
        print("=> g^MC abs error ~ base-VF abs error: the MC method works; the residual is the")
        print("   base velocity field (train it better to approach the floor), NOT the MC estimator.")
    else:
        print("=> g^MC abs error >> base-VF abs error: the MC estimator/ODE guidance adds real error.")
    print("=" * 68)


if __name__ == "__main__":
    main()
