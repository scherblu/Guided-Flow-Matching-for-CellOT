"""Visualize WHY g^cov ticks up at exact-OT vs reg=0.1 (the field-geometry hypothesis).

g^cov adds -lambda*grad J at the one-step endpoint prediction x_hat1 = x + (1-t)*v_theta. Its
accuracy therefore depends on where x_hat1 lands -- i.e. on the base FIELD geometry, which differs
by coupling even when the final unguided distribution is equally good. Hypothesis: under the fully
committed exact-OT field the endpoint predictions are slightly mislocated vs the marginally softer
reg=0.1 field, so g^cov lands slightly off-target despite exact-OT having the better base VF.

Per coupling (reg=0.1, exact-OT), two rows:
  row 0: g^cov final samples vs the tilted truth + the 3 target centers  -> SEE the mislocation
  row 1: endpoint-prediction map x_hat1 at t=0.5 along the unguided trajectory + centers
         -> the CAUSE: do the predictions concentrate ON the 3 target blobs, or off them?

Run from synthetic/:  ../.pixi/envs/default/bin/python script/debug_gcov_field.py
"""
import os
import sys

import torch
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from guided_flow.distributions.base import get_distribution
from guided_flow.utils.metrics import energy_distance
from guided_flow.utils.misc import deterministic
from guided_flow.utils.oracle import rejection_sample_tilted

import go_nogo_spike as spike
import phase_diagram as pd

DEVICE = "cpu"
N = 2000
W = 0.05                       # sharp width where the exact-OT uptick is significant
COUPLINGS = [("reg=0.1 (best g^cov)", "entropic_ot_cfm", 0.1),
             ("exact-OT (uptick)", "ot_cfm", None)]


def gcov_and_endpoint_map(model, x0_dist, x1_dist, n):
    """g^cov final samples + the unguided endpoint prediction x_hat1 at t=0.5."""
    deterministic(0)
    samples = spike.sample_guided(model, x0_dist, spike.get_gcov_guide_fn(x1_dist), n)

    # unguided base trajectory to t~0.5, then its one-step endpoint prediction x_hat1
    deterministic(0)
    x = x0_dist.sample(n).to(DEVICE)
    t_span = torch.linspace(0, spike.T_END, spike.NUM_STEPS)
    half = spike.NUM_STEPS // 2
    for k in range(half):
        dt = t_span[k + 1] - t_span[k]
        with torch.no_grad():
            v = model(torch.cat([x, t_span[k].expand(n, 1)], dim=-1))
        x = x + dt * v
    t_half = t_span[half]
    with torch.no_grad():
        v = model(torch.cat([x, t_half.expand(n, 1)], dim=-1))
    x_hat1 = x + (1 - t_half) * v
    return samples, x_hat1, float(t_half)


def analyze_blobs(samples, centers_t, label):
    """Per target blob (nearest-center assignment): mass %, centroid offset, radial spread.
    Pins down whether g^cov error is mislocation (offset), mass imbalance, or wrong width."""
    s = samples[torch.isfinite(samples).all(-1)]
    d = torch.cdist(s, centers_t)
    assign = d.argmin(1)
    print(f"  {label}:")
    for b in range(centers_t.shape[0]):
        sb = s[assign == b]
        if sb.shape[0] == 0:
            print(f"    blob {b}: (empty)")
            continue
        mass = 100.0 * sb.shape[0] / s.shape[0]
        offset = (sb.mean(0) - centers_t[b]).norm().item()
        spread = (sb - sb.mean(0)).norm(dim=-1).mean().item()
        print(f"    blob {b}: mass={mass:4.1f}%  centroid_offset={offset:.4f}  radial_spread={spread:.4f}")


def main():
    x0 = get_distribution("uniform")
    x1 = get_distribution("8gaussian")
    x1.set_energy(spike.ENERGY_FN, target_indices=spike.TARGET_BLOBS, std_target=W)
    centers = x1.centers[spike.TARGET_BLOBS].cpu().numpy()
    centers_t = x1.centers[spike.TARGET_BLOBS].to(DEVICE)
    truth = rejection_sample_tilted(x1, N, scale=spike.SCALE, device=DEVICE)
    t_np = truth.cpu().numpy()
    print(f"\nper-blob breakdown (3 target blobs), sharp J w={W}:")
    analyze_blobs(truth, centers_t, "tilted truth (reference)")

    fig, axes = plt.subplots(2, len(COUPLINGS), figsize=(5 * len(COUPLINGS), 9), squeeze=False)
    for ci, (label, cfm, reg) in enumerate(COUPLINGS):
        model = pd.train_vf(cfm, reg, x0, x1)          # cached
        samples, xhat, t_half = gcov_and_endpoint_map(model, x0, x1, N)
        ed = energy_distance(samples, truth)
        analyze_blobs(samples, centers_t, f"{label}  (edist={ed:.4f})")
        s = samples[torch.isfinite(samples).all(-1)].cpu().numpy()
        xh = xhat[torch.isfinite(xhat).all(-1)].cpu().numpy()

        ax = axes[0][ci]
        ax.scatter(t_np[:, 0], t_np[:, 1], s=8, c="0.7", alpha=0.5, label="tilted truth")
        ax.scatter(s[:, 0], s[:, 1], s=8, c="tab:red", alpha=0.4, label="g^cov samples")
        ax.scatter(centers[:, 0], centers[:, 1], s=200, marker="x", c="k", linewidths=3,
                   label="target centers", zorder=5)
        ax.set_title(f"{label}\ng^cov samples vs truth  (edist={ed:.4f})", fontsize=10)
        ax.legend(loc="upper right", fontsize=7)
        ax.set_aspect("equal")

        ax = axes[1][ci]
        zc, win = centers[0], 0.22                       # zoom on one target blob
        ax.scatter(t_np[:, 0], t_np[:, 1], s=22, c="0.7", alpha=0.6, label="tilted truth (width~0.057)")
        ax.scatter(s[:, 0], s[:, 1], s=22, c="tab:red", alpha=0.5, label="g^cov samples")
        ax.scatter([zc[0]], [zc[1]], s=200, marker="x", c="k", linewidths=3, zorder=5)
        ax.set_xlim(zc[0] - win, zc[0] + win)
        ax.set_ylim(zc[1] - win, zc[1] + win)
        ax.set_title("zoom on one blob: g^cov collapses to a near-delta\n"
                     "(within-blob spread ~0 vs truth ~0.057)", fontsize=10)
        ax.legend(loc="upper right", fontsize=7)
        ax.set_aspect("equal")

    fig.suptitle(f"g^cov field-geometry check (uniform->8gaussian, sharp J w={W}): "
                 "reg=0.1 vs exact-OT", fontsize=12)
    fig.tight_layout()
    out = os.path.join(pd.LOGDIR, "debug_gcov_field.png")
    fig.savefig(out, dpi=150)
    print(f"[fig] {out}")
    print(f"endpoint-pred spread (std of x_hat1):  computed per panel above")


if __name__ == "__main__":
    main()
