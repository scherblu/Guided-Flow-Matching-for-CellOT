r"""Re-tune the faithful g^cov+Sigma scale when switching as_var -> Feng's pigdm schedule.

The covariance coefficient r_t^2 in get_gcov_cov_guide_fn can be either schedule:
    as_var = ((1-t)/t)^2          (flat-prior, eps-capped ~ up to 81 early -- the repo default)
    pigdm  = (1-t)^2/((1-t)^2+t^2)  (Feng 2502.02150's prior-corrected variance, bounded [0,1])
Because their magnitudes differ ~80x, the scale tuned for as_var is wrong for pigdm. Sweep both
on the exact-OT model (within-blob collapse isolated) and report the best scale by energy distance.

Run from synthetic/:  ../.pixi/envs/default/bin/python script/tune_pigdm.py
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
EPS = 0.1
CLAMP = pd.GCOV_CLAMP


def sched(name, t):
    if name == "as_var":
        return (1.0 / (t + EPS) - 1.0) ** 2
    if name == "pigdm":
        return (1 - t) ** 2 / ((1 - t) ** 2 + t ** 2 + 1e-8)
    raise ValueError(name)


def make_guide(x1_dist, schedule, scale):
    """g_cov_G (grad through model) x schedule(t) x scale, with final-norm clamp."""
    def gf(t, x, dx_dt, model):
        xr = x.detach().requires_grad_(True)
        with torch.enable_grad():
            v = model(torch.cat([xr, t.expand(xr.shape[0], 1)], dim=-1))
            x1p = xr + v * (1 - t)
            J = x1_dist.get_J(x1p)
            g = -torch.autograd.grad(J.sum(), xr)[0]
        g = scale * sched(schedule, t) * g
        n = g.norm(dim=-1, keepdim=True)
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

    print(f"{'schedule':10s} {'scale':>8s} | {'core':>8s} {'imbal':>7s} {'edist':>8s}")
    print(f"  (target core ~ {target:.3f}; lower edist better)")
    print("-" * 50)
    for schedule, scales in (
        ("as_var", (0.005, 0.01, 0.02, 0.05)),
        ("pigdm", (0.1, 0.3, 0.6, 1.0, 2.0, 4.0)),
    ):
        best = None
        for scale in scales:
            deterministic(0)
            s = spike.sample_guided(model, x0, make_guide(x1, schedule, scale), N)
            ed = energy_distance(s, truth)
            _, core, imb = blob_stats(s, centers_t)
            flag = ""
            if best is None or ed < best[0]:
                best = (ed, core, imb, scale)
                flag = " <-"
            print(f"{schedule:10s} {scale:8.3f} | {core:8.4f} {imb:7.2f} {ed:8.4f}{flag}")
        ed, core, imb, scale = best
        print(f"  >> best {schedule}: scale={scale}  edist={ed:.4f}  core={core:.4f}\n")

    deterministic(0)
    gmc = spike.sample_guided(model, x0, spike.get_mc_guide_fn(x0, x1, spike.SCALE, spike.MC_BATCH, N), N)
    _, core, imb = blob_stats(gmc, centers_t)
    print(f"{'g^MC ref':10s} {'':>8s} | {core:8.4f} {imb:7.2f} {energy_distance(gmc, truth):8.4f}")


if __name__ == "__main__":
    main()
