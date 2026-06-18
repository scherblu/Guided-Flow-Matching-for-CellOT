"""GO/NO-GO spike (IMPLEMENTATION_SPEC_pivot.md sec.7.0).

Question: does the guidance-failure corner even exist? At the *strongest* coupling
(ExactOptimalTransportConditionalFlowMatcher) and the *sharpest* energy J, do g^cov and
large-N g^MC visibly diverge from the analytic ground-truth tilted target
p~1 propto p1 * exp(-J)?

  GO    -- both diverge from truth here -> build the full phase diagram.
  NO-GO -- they match truth even at this extreme -> re-scope.

We also run the *independent-coupling* model as the control column (sec.7.6): there P=1
and every method must match truth -- if it doesn't, the harness is wrong, not the science.

Self-contained: trains the two 2D models inline (no tyro/wandb), caches them, then
samples g^cov + large-N g^MC and scores both against the rejection-sampled oracle.
Run from synthetic/:  ../.pixi/envs/default/bin/python script/go_nogo_spike.py
"""
import os
import sys

import torch
import matplotlib.pyplot as plt
from torch.distributions import Normal, Independent

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from guided_flow.backbone.mlp import MLP
from guided_flow.distributions.base import get_distribution
from guided_flow.flow.conditional_flow_matching import get_cfm
from guided_flow.guidance.gradient_guidance import wrap_grad_fn
from guided_flow.utils.metrics import compute_w2
from guided_flow.utils.misc import deterministic
from guided_flow.utils.oracle import rejection_sample_tilted

# ----- spike configuration: the extreme corner -----------------------------------------
DEVICE = "cpu"
WIDTH = 128                 # MLP hidden width
TRAIN_STEPS = 15000         # 2D, seconds-minutes on CPU
SHARP_W = 0.05              # energy width; blob std is 0.1, so 0.05 = sharp J
TARGET_BLOBS = [0, 2, 5]    # 3 of the 8 modes (spec)
SCALE = 1.0                 # tilt strength e^{-SCALE * J}; shared by oracle + methods
N_SAMPLE = 2000             # samples per method / oracle
MC_BATCH = 1000             # large N for g^MC so residual corner-error is P-bias, not variance
NUM_STEPS = 100             # ODE steps
T_END = 0.99               # avoid the t->1 blow-up in the conditional VF
ENERGY_FN = "three_blobs_continuous_multi_well"
LOGDIR = "logs/uniform-8gaussian/spike"


# ----- training (inline, strongest + independent coupling) -----------------------------
def train_vf(cfm_name, x0_dist, x1_dist, seed=0):
    """Train an unconditional 2D velocity field; cache by coupling name."""
    os.makedirs(LOGDIR, exist_ok=True)
    path = os.path.join(LOGDIR, f"{cfm_name}_w{WIDTH}_s{TRAIN_STEPS}.pth")
    model = MLP(dim=2, w=WIDTH, time_varying=True).to(DEVICE)
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
        print(f"[load] {cfm_name}: {path}")
        return model

    deterministic(seed)
    FM = get_cfm(cfm_name, sigma=0.0)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    bs = 256 if cfm_name == "ot_cfm" else 1024
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
            print(f"[train {cfm_name}] step {k+1}/{TRAIN_STEPS}  loss {loss.item():.4f}")
    torch.save(model.state_dict(), path)
    print(f"[save] {cfm_name}: {path}")
    return model


# ----- guidance (copied from notebooks/mc.ipynb harness) -------------------------------
GCOV_CLAMP = 50.0  # per-sample gradient-norm clamp: keep Euler stable so g^cov's failure
                   # shows as collapse/bias, not as an integrator NaN under the sharp J.


def get_gcov_guide_fn(x1_dist, guide_scale=1.0, schedule="linear_decay"):
    """g^cov (method C): gradient of J at the one-step x1 prediction (additive guidance)."""
    def guide_fn(t, x, dx_dt, model):
        x1_pred = x + dx_dt * (1 - t)
        with torch.enable_grad():
            x1_pred = x1_pred.requires_grad_(True)
            J = x1_dist.get_J(x1_pred)
            try:
                g = -torch.autograd.grad(J.sum(), x1_pred, create_graph=True)[0]
            except Exception:
                return torch.zeros_like(x)
        norm = g.norm(dim=-1, keepdim=True)
        return g * (GCOV_CLAMP / norm).clamp(max=1.0)  # clip large gradients, keep direction
    return wrap_grad_fn(guide_scale, schedule, guide_fn)


def get_mc_guide_fn(x0_dist, x1_dist, scale, mc_batch, batch, ot_std=0.1, ep=1e-2):
    """g^MC (method D), ot_cfm branch: exact Monte-Carlo guidance over the coupling."""
    def ot_cfm_log_p_tz(x0, x1, xt, t, std):
        mean = t * x1 + (1 - t) * x0
        return Independent(Normal(loc=mean, scale=std), 1).log_prob(xt)

    def draw_coupling():
        x0 = x0_dist.sample(mc_batch).to(DEVICE).unsqueeze(0).repeat(batch, 1, 1).permute(1, 0, 2).reshape(-1, 2)
        x1 = x1_dist.sample(mc_batch).to(DEVICE).unsqueeze(0).repeat(batch, 1, 1).permute(1, 0, 2).reshape(-1, 2)
        Jx1 = torch.exp(-scale * x1_dist.get_J(x1))
        return x0, x1, Jx1

    def guide_fn(t, x, dx_dt, model):
        # IMPORTANT: re-draw the coupling set EVERY step. A fixed set makes the finite-B
        # estimator error a deterministic drift bias that compounds along the trajectory
        # (smeared/structured samples); resampling makes per-step errors average out.
        # See script/debug_mc.py: fixed vs resampled at identical ESS -> W2 0.40 vs 0.14.
        x0, x1, Jx1 = draw_coupling()
        b = x.shape[0]
        B = mc_batch
        x_ = x.repeat(B, 1)
        t_ = t.repeat(B * b, 1)
        log_p_tz_x = ot_cfm_log_p_tz(x0, x1, x_, t_, std=ot_std)
        log_p_t_x = log_p_tz_x.reshape(B, b, 1).logsumexp(0) - torch.log(torch.tensor(B, device=x.device))
        log_p_tz_x_times_J = (log_p_tz_x + torch.log(Jx1)).reshape(B, b, 1)
        logZ = torch.logsumexp(log_p_tz_x_times_J, 0) - torch.log(torch.tensor(B, device=x.device)) - log_p_t_x
        Z = torch.exp(logZ)
        u = x1 - x0
        g = (log_p_tz_x.reshape(B, b, 1) - log_p_t_x.unsqueeze(0)).exp() * (
            Jx1.reshape(B, b, 1) / Z - 1) * u.reshape(B, b, 2)
        return g.mean(0)

    return guide_fn


def safe_w2(samples, truth):
    """W2 to truth over the finite samples; also report the non-finite fraction.

    A diverged method produces inf/nan rows that make POT's LP unbounded and silently
    return 0 -- we must not read that as 'perfect'. Returns (w2_over_finite, nonfinite_frac)."""
    finite = samples[torch.isfinite(samples).all(dim=-1)]
    nf = 1.0 - finite.shape[0] / samples.shape[0]
    if finite.shape[0] < 2:
        return float("inf"), nf
    return compute_w2(finite, truth), nf


def diversity_trace_ratio(samples, truth):
    """trace(cov(pred)) / trace(cov(truth)); << 1 = collapse, >> 1 = scatter."""
    finite = samples[torch.isfinite(samples).all(dim=-1)]
    if finite.shape[0] < 2:
        return 0.0
    return (finite.var(0).sum() / truth.var(0).sum()).item()


def sample_guided(model, x0_dist, guide_fn, n):
    """Explicit Euler integration of  dx/dt = v_theta(x,t) + guide(t,x,v,model)
    (mirrors GuidedMLPWrapper + torchdyn's euler solver, without the torchdyn dep)."""
    x = x0_dist.sample(n).to(DEVICE)
    t_span = torch.linspace(0, T_END, NUM_STEPS)
    for k in range(NUM_STEPS - 1):
        t = t_span[k]
        dt = t_span[k + 1] - t_span[k]
        with torch.no_grad():
            v = model(torch.cat([x, t.expand(x.shape[0], 1)], dim=-1))
        g = guide_fn(t, x, v, model)
        x = x + dt * (v + g)
    return x.detach()


# ----- run -----------------------------------------------------------------------------
def main():
    deterministic(0)
    x0_dist = get_distribution("uniform")
    x1_dist = get_distribution("8gaussian")
    x1_dist.set_energy(ENERGY_FN, target_indices=TARGET_BLOBS, std_target=SHARP_W)

    # analytic ground truth p~1 propto p1 * exp(-J)
    truth = rejection_sample_tilted(x1_dist, N_SAMPLE, scale=SCALE, device=DEVICE)
    floor = compute_w2(truth, rejection_sample_tilted(x1_dist, N_SAMPLE, scale=SCALE, device=DEVICE))
    print(f"\noracle floor  W2(truth, truth') = {floor:.4f}  (sampling-noise baseline)\n")

    couplings = {"independent (control, P=1)": "cfm", "exact-OT (strongest)": "ot_cfm"}
    results = {}
    for cname, cfm in couplings.items():
        model = train_vf(cfm, x0_dist, x1_dist)
        gcov = sample_guided(model, x0_dist, get_gcov_guide_fn(x1_dist), N_SAMPLE)
        mc = sample_guided(model, x0_dist,
                           get_mc_guide_fn(x0_dist, x1_dist, SCALE, MC_BATCH, N_SAMPLE), N_SAMPLE)
        res = {}
        for m, s in [("g^cov", gcov), ("g^MC", mc)]:
            w2, nf = safe_w2(s, truth)
            res[m] = (s, w2, nf, diversity_trace_ratio(s, truth))
        results[cname] = res
        for m in ("g^cov", "g^MC"):
            _, w2, nf, dtr = res[m]
            print(f"[{cname:28s}] {m:6s}  W2->truth={w2:.4f}  nonfinite={nf:.0%}  div_trace={dtr:.2f}")

    # ----- figure -----
    ncol = len(couplings)
    fig, axes = plt.subplots(1, ncol, figsize=(5 * ncol, 5), squeeze=False)
    t = truth.cpu().numpy()
    for ax, (cname, res) in zip(axes[0], results.items()):
        ax.scatter(t[:, 0], t[:, 1], s=6, c="k", alpha=0.25, label="truth p~1")
        for m, c in [("g^cov", "tab:red"), ("g^MC", "tab:blue")]:
            samp, w2, nf, dtr = res[m]
            s = samp[torch.isfinite(samp).all(dim=-1)].cpu().numpy()
            ax.scatter(s[:, 0], s[:, 1], s=6, c=c, alpha=0.35,
                       label=f"{m} (W2={w2:.3f}, div={dtr:.2f})")
        ax.set_title(cname)
        ax.legend(loc="upper right", fontsize=8)
        ax.set_aspect("equal")
    fig.suptitle(f"GO/NO-GO spike: sharpest J (w={SHARP_W}, blobs {TARGET_BLOBS})  vs ground truth")
    fig.tight_layout()
    out = os.path.join(LOGDIR, "go_nogo_spike.png")
    fig.savefig(out, dpi=150)
    print(f"\n[fig] {out}")

    # ----- verdict -----
    corner = results["exact-OT (strongest)"]
    ctrl = results["independent (control, P=1)"]
    print("\n" + "=" * 70)
    print(f"oracle floor W2(truth,truth') = {floor:.4f}")
    # per-method independent -> exact-OT progression (the P-bias / collapse signature)
    go_signals = []
    for m in ("g^cov", "g^MC"):
        ci, cc = ctrl[m][1], corner[m][1]       # W2 indep, W2 exact-OT
        di, dc = ctrl[m][3], corner[m][3]       # div_trace indep, exact-OT
        worsens = (cc > max(2 * floor, 1.3 * ci)) or (dc < 0.5)  # error grows OR collapses
        go_signals.append(worsens)
        print(f"  {m:6s}: W2 {ci:.3f} (indep) -> {cc:.3f} (exact-OT) [{cc/floor:.1f}x floor]   "
              f"div_trace {di:.2f} -> {dc:.2f}   {'<-- fails in corner' if worsens else 'robust'}")
    print("-" * 70)
    if any(go_signals):
        print("VERDICT: GO -- at strong coupling x sharp J, additive/MC guidance diverges")
        print("         from the analytic truth (error grows and/or distribution collapses).")
        print("         => build the full phase diagram (spec sec.7.1-7.7).")
    else:
        print("VERDICT: NO-GO -- guidance tracks truth even at the extreme; re-scope (spec sec.7.0).")
    print("=" * 70)


if __name__ == "__main__":
    main()
