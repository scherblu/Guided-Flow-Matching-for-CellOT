r"""Does adding a covariance factor (faithful PiGDM g^cov) cure the g^cov collapse?

The repo's g^cov is the simplified DPS form -grad J(x_hat1) (g_cov_A) with a hand-set schedule,
dropping the posterior-covariance coefficient of a faithful Feng/PiGDM g^cov:
    g = (d x_hat1 / d xt)^T  Sigma_t  grad(-J(x_hat1))
For isotropic blobs Sigma_t = r_t^2 I (scalar, commutes), so:
    g = r_t^2 * (d x_hat1 / d xt)^T grad(-J)
        \_______/   \__________________________/
        as_var          g_cov_G (grad through the model = the Jacobian term)
i.e. faithful cov-factored g^cov = g_cov_G x as_var, both already in the repo but never combined.

Test on the exact-OT model (within-blob collapse isolated): can any of
  {g_cov_A, g_cov_G} x {no cov (const), cov factor r_t^2} recover the target blob width ~0.055?
Key prior result: for ISOTROPIC Sigma_t the scalar factor is just a reschedule, and no schedule
fixed collapse -- so if a cov factor helps it must be via the anisotropic Jacobian (g_cov_G).

Run from synthetic/:  ../.pixi/envs/default/bin/python script/debug_gcov_cov.py
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
from debug_gcov_collapse import blob_stats

DEVICE = "cpu"
N = 2000
W = 0.05
BASE_STD = 0.1
CLAMP = 50.0           # final-guidance norm clamp (prevents Euler blowup; keeps direction)
EPS = 0.1


def r2(t):
    """PiGDM-style isotropic posterior variance r_t^2 ~ ((1-t)/t)^2 (= the 'as_var' schedule)."""
    return (1.0 / (t + EPS) - 1.0) ** 2


def make_gcov(x1_dist, variant, use_cov, scale):
    """variant 'A' = grad wrt x_hat1 (point est); 'G' = grad wrt x through the model (Jacobian).
    use_cov multiplies by r_t^2 (the covariance factor)."""
    def gf(t, x, dx_dt, model):
        if variant == "A":
            x1p = (x + dx_dt * (1 - t)).detach().requires_grad_(True)
            with torch.enable_grad():
                J = x1_dist.get_J(x1p)
                g = -torch.autograd.grad(J.sum(), x1p)[0]
        else:  # 'G' -- backprop through the velocity field (the Jacobian term)
            xr = x.detach().requires_grad_(True)
            with torch.enable_grad():
                v = model(torch.cat([xr, t.expand(xr.shape[0], 1)], dim=-1))
                x1p = xr + v * (1 - t)
                J = x1_dist.get_J(x1p)
                g = -torch.autograd.grad(J.sum(), xr)[0]
        if use_cov:
            g = g * r2(t)
        g = scale * g
        n = g.norm(dim=-1, keepdim=True)               # final clamp (keep direction)
        return g * (CLAMP / n).clamp(max=1.0)
    return gf


def main():
    x0 = get_distribution("uniform")
    x1 = get_distribution("8gaussian")
    x1.set_energy(spike.ENERGY_FN, target_indices=spike.TARGET_BLOBS, std_target=W)
    centers_t = x1.centers[spike.TARGET_BLOBS].to(DEVICE)
    truth = rejection_sample_tilted(x1, N, scale=spike.SCALE, device=DEVICE)

    tilt_std = (1 / BASE_STD ** 2 + 1 / W ** 2) ** -0.5
    target = tilt_std * math.sqrt(math.pi / 2)
    _, t_core, _ = blob_stats(truth, centers_t)
    print(f"\nsharp J w={W}: TARGET core width ~ {target:.4f} (truth measured {t_core:.4f})\n")

    model = pd.train_vf("ot_cfm", None, x0, x1)

    print(f"{'variant':28s}  {'core width':>10s}  {'mass-imbal':>10s}  {'edist':>8s}")
    print(f"  (faithful cov-factored g^cov = G + cov; target core ~ {target:.3f})")
    print("-" * 64)
    for variant in ("A", "G"):
        for use_cov in (False, True):
            best = None
            for scale in (0.01, 0.05, 0.2, 1.0):           # pick the scale with min edist
                deterministic(0)
                g = make_gcov(x1, variant, use_cov, scale)
                s = spike.sample_guided(model, x0, g, N)
                ed = energy_distance(s, truth)
                _, core, imb = blob_stats(s, centers_t)
                if best is None or ed < best[0]:
                    best = (ed, core, imb, scale)
            ed, core, imb, scale = best
            tag = f"g_cov_{variant}" + (" x cov(r_t^2)" if use_cov else " (no cov)")
            print(f"{tag:28s}  {core:10.4f}  {imb:10.2f}  {ed:8.4f}   (best scale={scale})")
        print()

    deterministic(0)
    gmc = spike.sample_guided(model, x0, spike.get_mc_guide_fn(x0, x1, spike.SCALE, spike.MC_BATCH, N), N)
    _, core, imb = blob_stats(gmc, centers_t)
    print(f"{'g^MC (reference)':28s}  {core:10.4f}  {imb:10.2f}  {energy_distance(gmc, truth):8.4f}")
    print(f"{'TARGET (tilted truth)':28s}  {target:10.4f}  {0.0:10.2f}  {0.0:8.4f}")


if __name__ == "__main__":
    main()
