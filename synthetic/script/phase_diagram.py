"""Guidance-failure phase diagram (IMPLEMENTATION_SPEC_pivot.md sec.5).

The one figure: a heatmap of error-to-ground-truth over (x = coupling strength, y = energy
sharpness) for each guidance method, where ground truth is the analytic rejection-sampled
tilted target p~1 propto p1 * exp(-J). Reveals the corner -- strong coupling x sharp J --
where additive/MC guidance diverges (Feng's P~1 breakdown). The GO/NO-GO spike already found
this corner at the two extremes; this maps the full plane.

Axes (both measured/validated, not assumed):
  x  coupling: independent -> entropic-OT(reg large->small) -> exact OT.
     reg semantics confirmed by script/test_entropic_coupling.py (large reg ~ independent).
  y  sharpness: energy width w (blob std is 0.1, so w<0.1 is a sharp J). Smaller = sharper.

Methods under test (NOT oracles): g^cov (expect collapse in corner), g^MC (expect residual
P-bias even at large N). CFG and FK-SMC are left as drop-in method slots (see METHODS).
The headline metric is energy distance to truth (shared with the 4i phase); W2, diversity-
trace and off-manifold rate are companion panels.

Run from synthetic/ (long; cache + background it):
    ../.pixi/envs/default/bin/python script/phase_diagram.py
Fast wiring check on a 2x2 grid:
    ../.pixi/envs/default/bin/python script/phase_diagram.py --smoke
"""
import json
import os
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from guided_flow.backbone.mlp import MLP
from guided_flow.distributions.base import get_distribution
from guided_flow.flow.conditional_flow_matching import get_cfm
from guided_flow.utils.metrics import (
    diversity_trace_ratio,
    energy_distance,
    off_manifold_rate,
    safe_w2,
)
from guided_flow.utils.misc import deterministic
from guided_flow.utils.oracle import rejection_sample_tilted

import go_nogo_spike as spike

DEVICE = "cpu"
WIDTH = spike.WIDTH            # MLP hidden width (match the spike's cached anchors)
TRAIN_STEPS = spike.TRAIN_STEPS
SCALE = spike.SCALE
TARGET_BLOBS = spike.TARGET_BLOBS
ENERGY_FN = spike.ENERGY_FN
N_SAMPLE = spike.N_SAMPLE      # samples per (method, cell)
MC_BATCH = spike.MC_BATCH      # baseline g^MC config validated in script/debug_control.py
LOGDIR = "logs/uniform-8gaussian/phase_diagram"
SEEDS = (0, 1, 2)              # per-cell seeds for mean +/- CI

# ----- the two axes -----
# x: coupling, ordered independent -> exact OT. (label, cfm_name, reg). reg=None for the
# non-entropic anchors -- 'cfm'/'ot_cfm' reuse the spike's cached models (same naming).
COUPLINGS = [
    ("independent", "cfm", None),
    ("reg=10", "entropic_ot_cfm", 10.0),
    ("reg=1", "entropic_ot_cfm", 1.0),
    ("reg=0.3", "entropic_ot_cfm", 0.3),
    ("reg=0.1", "entropic_ot_cfm", 0.1),
    ("exact-OT", "ot_cfm", None),
]
# y: energy width (broad -> sharp). Blob std 0.1, so <=0.05 is the sharp regime.
SHARPNESS = [0.2, 0.1, 0.05, 0.03]

RENDER_ONLY = "--render-only" in sys.argv   # re-render saved results with a corrected k-draw floor
AUGMENT = "--augment" in sys.argv           # compute only METHODS missing from the saved JSON, merge, re-render
SMOKE = "--smoke" in sys.argv
if SMOKE:                       # tiny grid for a fast wiring check
    COUPLINGS = [COUPLINGS[0], COUPLINGS[-1]]
    SHARPNESS = [0.2, 0.05]
    TRAIN_STEPS = 500
    N_SAMPLE = 400
    MC_BATCH = 200
    SEEDS = (0,)


# ----- training (one unconditional VF per coupling; cached by coupling key) -------------
def train_vf(cfm_name, reg, x0_dist, x1_dist, seed=0):
    """Train (or load) a 2D unconditional VF for one coupling. Mirrors spike.train_vf but
    threads `reg` through for the entropic matcher; 'cfm'/'ot_cfm' reuse the spike cache."""
    os.makedirs(LOGDIR, exist_ok=True)
    key = cfm_name if reg is None else f"{cfm_name}_reg{reg}"
    fname = f"{key}_w{WIDTH}_s{TRAIN_STEPS}.pth"
    # the two anchors are cached by the spike under its own logdir -- reuse them
    for d in (LOGDIR, spike.LOGDIR):
        path = os.path.join(d, fname)
        if os.path.exists(path):
            model = MLP(dim=2, w=WIDTH, time_varying=True).to(DEVICE)
            model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
            print(f"[load] {key}: {path}")
            return model

    deterministic(seed)
    model = MLP(dim=2, w=WIDTH, time_varying=True).to(DEVICE)
    FM = get_cfm(cfm_name, sigma=0.0, reg=(reg if reg is not None else 0.05))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    bs = 1024 if cfm_name == "cfm" else 256       # OT solves favour a moderate batch
    for k in range(TRAIN_STEPS):
        opt.zero_grad()
        x0 = x0_dist.sample(bs, device=DEVICE)
        x1 = x1_dist.sample(bs, device=DEVICE)
        t = torch.rand(x0.shape[0], device=DEVICE)
        t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1, t)
        vt = model(torch.cat([xt, t[:, None]], dim=-1))
        loss = ((vt - ut) ** 2).mean()
        loss.backward()
        opt.step()
        if (k + 1) % 5000 == 0:
            print(f"[train {key}] step {k+1}/{TRAIN_STEPS}  loss {loss.item():.4f}")
    path = os.path.join(LOGDIR, fname)
    torch.save(model.state_dict(), path)
    print(f"[save] {key}: {path}")
    return model


# ----- methods under test (drop in CFG / FK-SMC here as new entries) --------------------
GCOV_CLAMP = 50.0
GCOV_COV_SCALE = 0.01        # faithful-g^cov strength (r_t^2 is large early; see debug_gcov_cov.py)


def get_gcov_cov_guide_fn(x1_dist, guide_scale=GCOV_COV_SCALE, eps=0.1):
    """Faithful PiGDM-style covariance-factored g^cov:  r_t^2 * (d x_hat1/d x)^T grad(-J).

    Two pieces the simplified `spike.get_gcov_guide_fn` (g_cov_A, point estimate) drops:
      - grad taken w.r.t. x THROUGH the velocity field (g_cov_G = the Jacobian (d x_hat1/d x)^T),
      - times the isotropic posterior variance r_t^2 ~ ((1-t)/t)^2 (the 'as_var' coefficient).
    See script/debug_gcov_cov.py: this is the best deterministic g^cov but still under-spreads.
    """
    def guide_fn(t, x, dx_dt, model):
        xr = x.detach().requires_grad_(True)
        with torch.enable_grad():
            v = model(torch.cat([xr, t.expand(xr.shape[0], 1)], dim=-1))
            x1_pred = xr + v * (1 - t)
            J = x1_dist.get_J(x1_pred)
            g = -torch.autograd.grad(J.sum(), xr)[0]
        g = guide_scale * (1.0 / (t + eps) - 1.0) ** 2 * g       # x r_t^2 (covariance factor)
        nrm = g.norm(dim=-1, keepdim=True)
        return g * (GCOV_CLAMP / nrm).clamp(max=1.0)
    return guide_fn


def gcov_samples(model, x0_dist, x1_dist, n, seed):
    deterministic(seed)
    return spike.sample_guided(model, x0_dist, spike.get_gcov_guide_fn(x1_dist), n)


def gcov_cov_samples(model, x0_dist, x1_dist, n, seed):
    deterministic(seed)
    return spike.sample_guided(model, x0_dist, get_gcov_cov_guide_fn(x1_dist), n)


def mc_samples(model, x0_dist, x1_dist, n, seed):
    deterministic(seed)
    guide = spike.get_mc_guide_fn(x0_dist, x1_dist, SCALE, MC_BATCH, n)
    return spike.sample_guided(model, x0_dist, guide, n)


METHODS = {
    "g^cov": gcov_samples,           # simplified DPS (g_cov_A, point estimate, no covariance)
    "g^cov+Σ": gcov_cov_samples,     # faithful PiGDM (g_cov_G x r_t^2 covariance factor)
    "g^MC": mc_samples,
    # "CFG":  cfg_samples,    # TODO: needs the conditional (label-dropout) model
    # "FK":   fk_samples,     # TODO: SDE conversion + particle resampling (spec sec.7.4)
}
METRICS = ["edist", "w2", "div_trace", "off_manifold", "nonfinite"]


def score(samples, truth):
    w2, nf = safe_w2(samples, truth)
    return {
        "edist": energy_distance(samples, truth),
        "w2": w2,
        "div_trace": diversity_trace_ratio(samples, truth),
        "off_manifold": off_manifold_rate(samples, truth),
        "nonfinite": nf,
    }


def mean_ci(vals):
    a = np.asarray(vals, dtype=float)
    m = float(a.mean())
    ci = float(1.96 * a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.0
    return m, ci


def edist_floor_for_width(x1_dist, w, k=5):
    """k-draw energy-distance floor for the tilted target at width `w` (sets the energy).

    A SINGLE oracle pair gives a noisy floor (the bug the spike had); the tilted target's
    edist floor is tiny so a single draw swings the `x floor` normalization wildly. Average k.
    """
    x1_dist.set_energy(ENERGY_FN, target_indices=TARGET_BLOBS, std_target=w)
    return sum(
        energy_distance(rejection_sample_tilted(x1_dist, N_SAMPLE, scale=SCALE, device=DEVICE),
                        rejection_sample_tilted(x1_dist, N_SAMPLE, scale=SCALE, device=DEVICE))
        for _ in range(k)) / k


# ----- run -----------------------------------------------------------------------------
def main():
    deterministic(0)
    os.makedirs(LOGDIR, exist_ok=True)
    x0_dist = get_distribution("uniform")
    x1_dist = get_distribution("8gaussian")

    nC, nS = len(COUPLINGS), len(SHARPNESS)
    # results[metric][method] = (nS, nC) array of means; *_ci likewise
    res = {m: {meth: np.full((nS, nC), np.nan) for meth in METHODS} for m in METRICS}
    res_ci = {m: {meth: np.full((nS, nC), np.nan) for meth in METHODS} for m in METRICS}
    edist_floor = np.full((nS,), np.nan)   # floor depends on J width (the target), not coupling

    for ci, (clabel, cfm_name, reg) in enumerate(COUPLINGS):
        model = train_vf(cfm_name, reg, x0_dist, x1_dist)
        for si, w in enumerate(SHARPNESS):
            if ci == 0:
                edist_floor[si] = edist_floor_for_width(x1_dist, w)  # k-draw, before set_energy below
            x1_dist.set_energy(ENERGY_FN, target_indices=TARGET_BLOBS, std_target=w)
            truth = rejection_sample_tilted(x1_dist, N_SAMPLE, scale=SCALE, device=DEVICE)
            for meth, fn in METHODS.items():
                per_seed = {m: [] for m in METRICS}
                for s in SEEDS:
                    sc = score(fn(model, x0_dist, x1_dist, N_SAMPLE, s), truth)
                    for m in METRICS:
                        per_seed[m].append(sc[m])
                for m in METRICS:
                    mean, ci95 = mean_ci(per_seed[m])
                    res[m][meth][si, ci] = mean
                    res_ci[m][meth][si, ci] = ci95
                e = res["edist"][meth][si, ci]
                print(f"[{clabel:12s} | w={w:5.3f}] {meth:6s}  "
                      f"edist={e:.4f} ({e/edist_floor[si]:.1f}x floor)  "
                      f"W2={res['w2'][meth][si, ci]:.3f}  div={res['div_trace'][meth][si, ci]:.2f}  "
                      f"off={res['off_manifold'][meth][si, ci]:.2f}  nf={res['nonfinite'][meth][si, ci]:.0%}")

    save_results(res, res_ci, edist_floor)
    render(res, edist_floor)


def save_results(res, res_ci, edist_floor):
    out = {
        "couplings": [c[0] for c in COUPLINGS],
        "sharpness": SHARPNESS,
        "edist_floor": edist_floor.tolist(),
        "metrics": {m: {meth: res[m][meth].tolist() for meth in METHODS} for m in METRICS},
        "metrics_ci": {m: {meth: res_ci[m][meth].tolist() for meth in METHODS} for m in METRICS},
    }
    path = os.path.join(LOGDIR, "phase_diagram_results.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[results] {path}")


def render(res, edist_floor):
    """Rows = {edist/floor, div_trace, off_manifold}; cols = methods. Heatmaps over (x=coupling,
    y=sharpness). edist is shown as a multiple of the per-width oracle floor (1.0 = perfect)."""
    panels = [
        ("edist", "edist / floor", "magma"),
        ("div_trace", "diversity-trace (1=ok, <1 collapse)", "coolwarm"),
        ("off_manifold", "off-manifold rate", "viridis"),
    ]
    methods = list(METHODS)
    fig, axes = plt.subplots(len(panels), len(methods),
                             figsize=(4.2 * len(methods), 3.6 * len(panels)), squeeze=False)
    xt = [c[0] for c in COUPLINGS]
    yt = [f"{w}" for w in SHARPNESS]
    for r, (metric, title, cmap) in enumerate(panels):
        for c, meth in enumerate(methods):
            ax = axes[r][c]
            M = res[metric][meth].copy()
            if metric == "edist":
                M = M / edist_floor[:, None]           # normalize to the floor
                vmin, vmax, center = 1.0, max(2.0, np.nanpercentile(M, 95)), None
            elif metric == "div_trace":
                vmin, vmax, center = 0.0, 2.0, None
            else:
                vmin, vmax, center = 0.0, 1.0, None
            im = ax.imshow(M, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_xticks(range(len(xt)), xt, rotation=45, ha="right", fontsize=7)
            ax.set_yticks(range(len(yt)), yt, fontsize=8)
            for si in range(len(SHARPNESS)):
                for cidx in range(len(COUPLINGS)):
                    ax.text(cidx, si, f"{M[si, cidx]:.2f}", ha="center", va="center",
                            fontsize=6, color="w")
            if r == 0:
                ax.set_title(meth, fontsize=11)
            if c == 0:
                # y ticks run broad (bottom) -> sharp (top) since origin='lower'
                ax.set_ylabel(f"{title}\nenergy width w (sharp = top)", fontsize=8)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Guidance-failure phase diagram\n"
                 "x = coupling (independent -> exact OT),  y = energy width (broad -> sharp)",
                 fontsize=12)
    fig.supxlabel("coupling strength", fontsize=10)
    fig.tight_layout()
    path = os.path.join(LOGDIR, "phase_diagram.png")
    fig.savefig(path, dpi=150)
    print(f"[fig] {path}")


def render_from_json():
    """Re-render the heatmap from saved results with a freshly computed k-draw floor.
    Reuses the expensive MC metrics in the JSON -- only the (cheap) oracle floor is recomputed."""
    with open(os.path.join(LOGDIR, "phase_diagram_results.json")) as f:
        d = json.load(f)
    res = {m: {meth: np.array(arr) for meth, arr in d["metrics"][m].items()} for m in d["metrics"]}
    x1_dist = get_distribution("8gaussian")
    floor = np.array([edist_floor_for_width(x1_dist, w) for w in d["sharpness"]])
    print("recomputed k-draw edist floors:", [f"{f:.5f}" for f in floor])
    render(res, floor)


def augment():
    """Compute only the METHODS absent from the saved JSON (e.g. a newly added g^cov+Σ),
    merge them into the existing results, recompute a k-draw floor, save + re-render.
    Reuses cached models AND the expensive cached MC results -- no g^MC re-run."""
    path = os.path.join(LOGDIR, "phase_diagram_results.json")
    with open(path) as f:
        d = json.load(f)
    nC, nS = len(COUPLINGS), len(SHARPNESS)
    assert d["couplings"] == [c[0] for c in COUPLINGS] and d["sharpness"] == list(SHARPNESS), \
        "saved grid does not match current COUPLINGS/SHARPNESS"

    def load(store):
        return {m: {meth: (np.array(store[m][meth]) if meth in store[m] else np.full((nS, nC), np.nan))
                    for meth in METHODS} for m in METRICS}
    res, res_ci = load(d["metrics"]), load(d["metrics_ci"])
    missing = [meth for meth in METHODS if meth not in d["metrics"]["edist"]]
    if not missing:
        print("nothing to augment; all METHODS already in the JSON.")
        return render_from_json()
    print(f"augmenting with: {missing}")

    x0_dist = get_distribution("uniform")
    x1_dist = get_distribution("8gaussian")
    edist_floor = np.full((nS,), np.nan)
    for ci, (clabel, cfm_name, reg) in enumerate(COUPLINGS):
        model = train_vf(cfm_name, reg, x0_dist, x1_dist)
        for si, w in enumerate(SHARPNESS):
            if ci == 0:
                edist_floor[si] = edist_floor_for_width(x1_dist, w)
            x1_dist.set_energy(ENERGY_FN, target_indices=TARGET_BLOBS, std_target=w)
            truth = rejection_sample_tilted(x1_dist, N_SAMPLE, scale=SCALE, device=DEVICE)
            for meth in missing:
                per_seed = {m: [] for m in METRICS}
                for s in SEEDS:
                    sc = score(METHODS[meth](model, x0_dist, x1_dist, N_SAMPLE, s), truth)
                    for m in METRICS:
                        per_seed[m].append(sc[m])
                for m in METRICS:
                    mean, ci95 = mean_ci(per_seed[m])
                    res[m][meth][si, ci] = mean
                    res_ci[m][meth][si, ci] = ci95
                e = res["edist"][meth][si, ci]
                print(f"[{clabel:12s} | w={w:5.3f}] {meth:8s}  edist={e:.4f} "
                      f"({e/edist_floor[si]:.1f}x floor)  W2={res['w2'][meth][si, ci]:.3f}  "
                      f"div={res['div_trace'][meth][si, ci]:.2f}  off={res['off_manifold'][meth][si, ci]:.2f}")
    save_results(res, res_ci, edist_floor)
    render(res, edist_floor)


if __name__ == "__main__":
    if RENDER_ONLY:
        render_from_json()
    elif AUGMENT:
        augment()
    else:
        main()
