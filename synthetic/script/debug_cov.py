"""Debug the g^cov (gradient-guidance) failure: strong collapse + 'lines' + 'triangular pattern'.

Two candidate mechanisms, separated here:
  (A) GEOMETRIC / energy-intrinsic: the 3-well logsumexp energy J has sharp Voronoi 'Y'
      separatrices between the 3 target modes; -grad J switches direction across them, so
      trajectories sort radially toward 3 vertices -> triangular / Y streak pattern.
  (B) NUMERICAL / Euler overshoot: |-grad J| ~ 1/w^2 (~400 at w=0.05); times dt=0.01 a single
      Euler step can exceed the distance to the mode -> overshoot/oscillation -> 'lines'.

We (1) visualize J + the guidance field, (2) draw g^cov trajectories over it, (3) ablate
num_steps / guide_scale / grad-clamp / width to attribute each artifact.

Run from synthetic/:  ../.pixi/envs/default/bin/python script/debug_cov.py
"""
import os
import sys

import torch
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from guided_flow.backbone.mlp import MLP
from guided_flow.distributions.base import get_distribution
from guided_flow.guidance.gradient_guidance import wrap_grad_fn
from guided_flow.utils.metrics import compute_w2
from guided_flow.utils.misc import deterministic
from guided_flow.utils.oracle import rejection_sample_tilted

DEVICE = "cpu"
WIDTH = 128
TARGETS = [0, 2, 5]
SCALE = 1.0
N = 1000
SPIKE = "logs/uniform-8gaussian/spike"


def load(cfm_name):
    m = MLP(dim=2, w=WIDTH, time_varying=True).to(DEVICE)
    m.load_state_dict(torch.load(f"{SPIKE}/{cfm_name}_w128_s15000.pth", map_location=DEVICE, weights_only=True))
    return m


def neg_grad_J(dist, x):
    """guidance direction -grad_x J(x)."""
    x = x.detach().requires_grad_(True)
    J = dist.get_J(x)
    g = -torch.autograd.grad(J.sum(), x)[0]
    return g.detach()


def gcov_guide(x1_dist, guide_scale, schedule, clamp=None):
    """g_cov_A: -grad J at the one-step prediction x1_pred = x + v*(1-t)."""
    def guide_fn(t, x, dx_dt, model):
        x1_pred = (x + dx_dt * (1 - t)).detach().requires_grad_(True)
        J = x1_dist.get_J(x1_pred)
        g = -torch.autograd.grad(J.sum(), x1_pred)[0]
        if clamp is not None:
            n = g.norm(dim=-1, keepdim=True)
            g = g * (clamp / n).clamp(max=1.0)
        return g
    return wrap_grad_fn(guide_scale, schedule, guide_fn)


def integrate_traj(model, x0_dist, guide_fn, n, num_steps, t_end=0.99):
    x = x0_dist.sample(n).to(DEVICE)
    ts = torch.linspace(0, t_end, num_steps)
    traj = [x.clone()]
    for k in range(num_steps - 1):
        t, dt = ts[k], ts[k + 1] - ts[k]
        with torch.no_grad():
            v = model(torch.cat([x, t.expand(x.shape[0], 1)], -1))
        g = guide_fn(t, x, v, model)
        x = x + dt * (v + g)
        traj.append(x.clone())
    return torch.stack(traj)  # (T, n, 2)


def field_panel(ax, dist, lim=1.5, grid=24):
    xs = torch.linspace(-lim, lim, 200)
    XX, YY = torch.meshgrid(xs, xs, indexing="ij")
    pts = torch.stack([XX.reshape(-1), YY.reshape(-1)], -1)
    with torch.no_grad():
        J = dist.get_J(pts).reshape(200, 200)
    ax.contourf(XX, YY, J, levels=30, cmap="viridis")
    gx = torch.linspace(-lim, lim, grid)
    GX, GY = torch.meshgrid(gx, gx, indexing="ij")
    gpts = torch.stack([GX.reshape(-1), GY.reshape(-1)], -1)
    g = neg_grad_J(dist, gpts)
    gn = g / (g.norm(dim=-1, keepdim=True) + 1e-8)  # unit dirs (show direction, not stiff magnitude)
    ax.quiver(gpts[:, 0], gpts[:, 1], gn[:, 0], gn[:, 1], color="w", alpha=0.6, scale=30)
    ax.set_aspect("equal")


def main():
    deterministic(0)
    x0_dist = get_distribution("uniform")
    x1 = get_distribution("8gaussian")
    model = load("cfm")  # independent model (same as toy notebook coupling)

    # ---------- 1. energy + guidance field for the candidate energies ----------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (name, ef, kw) in zip(axes, [
        ("single_blob w=0.1", "single_blob_quadratic_distance", dict(target_blob_idx=0, std_target=0.1)),
        ("3-blob multi_well w=0.05", "three_blobs_continuous_multi_well", dict(target_indices=TARGETS, std_target=0.05)),
        ("3-blob multi_well w=0.20", "three_blobs_continuous_multi_well", dict(target_indices=TARGETS, std_target=0.20)),
    ]):
        d = get_distribution("8gaussian"); d.set_energy(ef, **kw)
        field_panel(ax, d); ax.set_title(name)
    fig.suptitle("g^cov guidance field  -grad J  (white = direction)")
    fig.tight_layout(); fig.savefig(f"{SPIKE}/debug_cov_field.png", dpi=140)
    print(f"[fig] {SPIKE}/debug_cov_field.png")

    # ---------- 2. trajectories + ablations on the 3-blob energy ----------
    x1.set_energy("three_blobs_continuous_multi_well", target_indices=TARGETS, std_target=0.05)
    truth = rejection_sample_tilted(x1, N, scale=SCALE, device=DEVICE)
    floor = compute_w2(truth, rejection_sample_tilted(x1, N, scale=SCALE, device=DEVICE))
    print(f"\nfloor W2 = {floor:.4f}")

    # base shows the NaN 'lines'; clamp fixes those; schedule sweep tests whether the
    # COLLAPSE is intrinsic or an under-decayed-covariance artifact (stronger terminal decay
    # ~ true posterior cov (1-t)^2 should raise div_trace toward 1 if collapse is salvageable).
    ablations = [
        ("base scale1 steps100 (lines)", dict(guide_scale=1.0, schedule="linear_decay", clamp=None, steps=100, w=0.05)),
        ("clamp50 const (no decay)",     dict(guide_scale=1.0, schedule="const", clamp=50.0, steps=100, w=0.05)),
        ("clamp50 linear_decay",         dict(guide_scale=1.0, schedule="linear_decay", clamp=50.0, steps=100, w=0.05)),
        ("clamp50 cosine_decay",         dict(guide_scale=1.0, schedule="cosine_decay", clamp=50.0, steps=100, w=0.05)),
        ("clamp50 exp_decay",            dict(guide_scale=1.0, schedule="exp_decay", clamp=50.0, steps=100, w=0.05)),
    ]

    fig2, axes2 = plt.subplots(1, len(ablations) + 1, figsize=(4 * (len(ablations) + 1), 4), squeeze=False)
    tnp = truth.cpu().numpy()
    axes2[0][0].scatter(tnp[:, 0], tnp[:, 1], s=5, c="k", alpha=0.4)
    axes2[0][0].set_title("truth p~1"); axes2[0][0].set_aspect("equal")
    axes2[0][0].set_xlim(-1.5, 1.5); axes2[0][0].set_ylim(-1.5, 1.5)

    print("\n%-32s %8s %10s %10s" % ("ablation", "W2", "div_trace", "nonfinite"))
    for ax, (name, p) in zip(axes2[0][1:], ablations):
        d = get_distribution("8gaussian")
        d.set_energy("three_blobs_continuous_multi_well", target_indices=TARGETS, std_target=p["w"])
        deterministic(0)
        traj = integrate_traj(model, x0_dist, gcov_guide(d, p["guide_scale"], p["schedule"], p["clamp"]),
                              N, p["steps"])
        end = traj[-1]
        fin = end[torch.isfinite(end).all(-1)]
        nf = 1 - fin.shape[0] / end.shape[0]
        w2 = compute_w2(fin, truth) if fin.shape[0] > 1 else float("inf")
        dtr = (fin.var(0).sum() / truth.var(0).sum()).item() if fin.shape[0] > 1 else 0.0
        print("%-32s %8.4f %10.2f %9.0f%%" % (name, w2, dtr, 100 * nf))
        # draw a few trajectories (lines) + endpoints
        sub = traj[:, :150]  # (T,150,2)
        sub = torch.where(torch.isfinite(sub), sub, torch.nan)
        for j in range(sub.shape[1]):
            ax.plot(sub[:, j, 0], sub[:, j, 1], lw=0.3, alpha=0.25, color="tab:gray")
        e = fin.cpu().numpy()
        ax.scatter(tnp[:, 0], tnp[:, 1], s=4, c="k", alpha=0.12)
        ax.scatter(e[:, 0], e[:, 1], s=5, c="tab:red", alpha=0.5)
        ax.set_title(f"{name}\nW2={w2:.3f} div={dtr:.2f}", fontsize=8)
        ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5); ax.set_aspect("equal")
    fig2.suptitle("g^cov trajectories + endpoints (3-blob energy) — ablations")
    fig2.tight_layout(); fig2.savefig(f"{SPIKE}/debug_cov_traj.png", dpi=140)
    print(f"\n[fig] {SPIKE}/debug_cov_traj.png")


if __name__ == "__main__":
    main()
