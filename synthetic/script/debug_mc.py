"""Debug the g^MC overdispersion + 'weird structure' seen in the GO/NO-GO spike.

Hypothesis: the ot_cfm MC guide approximates the conditional path density p_t(xt|z) with a
FIXED-std Gaussian kernel (ot_std=0.1) over a FIXED set of B coupling samples. Over the
[-1,1]^2 box that kernel is narrow -> only a few coupling samples carry weight -> effective
sample size (ESS) collapses -> guidance is dominated by a handful of fixed x1 attractors ->
reproducible filaments/clumps + overdispersion.

Test on the INDEPENDENT model (where g^MC *should* recover truth, P=1):
  (1) instrument ESS(t) for the kernel estimator at several ot_std,
  (2) compare against the ANALYTIC cfm-branch estimator (uses exact source density, no kernel),
  (3) plot samples for each so the structure is visible.

Run from synthetic/:  ../.pixi/envs/default/bin/python script/debug_mc.py
"""
import os
import sys

import torch
import matplotlib.pyplot as plt
from torch.distributions import Normal, Independent

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from guided_flow.backbone.mlp import MLP
from guided_flow.distributions.base import get_distribution
from guided_flow.utils.metrics import compute_w2
from guided_flow.utils.misc import deterministic
from guided_flow.utils.oracle import rejection_sample_tilted

DEVICE = "cpu"
WIDTH = 128
SHARP_W = 0.05
TARGETS = [0, 2, 5]
SCALE = 1.0
N = 2000
MC_BATCH = 1000
NUM_STEPS = 100
T_END = 0.99
EP = 1e-2
SPIKE = "logs/uniform-8gaussian/spike"

ess_log = {}  # (tag) -> list of (t, mean_ESS)


def load(cfm_name):
    m = MLP(dim=2, w=WIDTH, time_varying=True).to(DEVICE)
    m.load_state_dict(torch.load(f"{SPIKE}/{cfm_name}_w128_s15000.pth", map_location=DEVICE, weights_only=True))
    return m


def mc_guide_kernel(x0_dist, x1_dist, ot_std, tag, resample=False):
    """ot_cfm-branch MC guide: Gaussian kernel of width ot_std around t*x1+(1-t)*x0.
    Logs effective sample size = 1 / sum_z w_z^2 (averaged over points)."""
    ess_log[tag] = []
    state = {}

    def draw():
        x0 = x0_dist.sample(MC_BATCH).to(DEVICE).unsqueeze(0).repeat(N, 1, 1).permute(1, 0, 2).reshape(-1, 2)
        x1 = x1_dist.sample(MC_BATCH).to(DEVICE).unsqueeze(0).repeat(N, 1, 1).permute(1, 0, 2).reshape(-1, 2)
        Jx1 = torch.exp(-SCALE * x1_dist.get_J(x1))
        return x0, x1, Jx1

    state["coupling"] = draw()

    def guide(t, x, dx_dt, model):
        if resample:
            state["coupling"] = draw()
        x0, x1, Jx1 = state["coupling"]
        b, B = x.shape[0], MC_BATCH
        x_ = x.repeat(B, 1)
        t_ = t.repeat(B * b, 1)
        mean = t_ * x1 + (1 - t_) * x0
        log_p = Independent(Normal(mean, ot_std), 1).log_prob(x_)          # (B*b,)
        log_pt = log_p.reshape(B, b, 1).logsumexp(0) - torch.log(torch.tensor(B))
        # ESS from normalized weights w_z = softmax_B(log_p)
        logw = log_p.reshape(B, b, 1) - log_p.reshape(B, b, 1).logsumexp(0, keepdim=True)
        w = logw.exp()
        ess = (1.0 / (w ** 2).sum(0)).mean().item()
        ess_log[tag].append((t.item(), ess))
        log_pJ = (log_p + torch.log(Jx1)).reshape(B, b, 1)
        Z = torch.exp(torch.logsumexp(log_pJ, 0) - torch.log(torch.tensor(B)) - log_pt)
        u = x1 - x0
        g = (log_p.reshape(B, b, 1) - log_pt.unsqueeze(0)).exp() * (Jx1.reshape(B, b, 1) / Z - 1) * u.reshape(B, b, 2)
        return g.mean(0)  # RAW, no schedule -- matches the spike's MC guide
    return guide


def mc_guide_analytic(x0_dist, x1_dist, tag):
    """cfm-branch MC guide: exact source density p(x0) via the deterministic inverse
    x0 = (xt - t*x1)/(1-t). No kernel std. Exact for the independent-coupling marginal.
    Logs ESS from the analytic weights."""
    ess_log[tag] = []
    x1 = x1_dist.sample(MC_BATCH).to(DEVICE).unsqueeze(0).repeat(N, 1, 1).permute(1, 0, 2).reshape(-1, 2)
    Jx1 = torch.exp(-SCALE * x1_dist.get_J(x1))

    def guide(t, x, dx_dt, model):
        b, B = x.shape[0], MC_BATCH
        x_ = x.repeat(B, 1)
        t_ = t.repeat(B * b, 1)
        x0 = x_ / (1 - t_ + EP) - (t_ + EP) / (1 - t_ + EP) * x1
        p1t = x0_dist.prob(x0).clamp(1e-8) / (1 - t + EP) ** 2
        log_p = p1t.log()
        log_pt = log_p.reshape(B, b, 1).logsumexp(0) - torch.log(torch.tensor(B))
        logw = log_p.reshape(B, b, 1) - log_p.reshape(B, b, 1).logsumexp(0, keepdim=True)
        ess_log[tag].append((t.item(), (1.0 / (logw.exp() ** 2).sum(0)).mean().item()))
        log_pJ = (log_p + torch.log(Jx1)).reshape(B, b, 1)
        Z = torch.exp(torch.logsumexp(log_pJ, 0) - torch.log(torch.tensor(B)) - log_pt)
        u = (x1 - x_) / (1 - t_ + EP)
        g = (log_p.reshape(B, b, 1) - log_pt.unsqueeze(0)).exp() * (Jx1.reshape(B, b, 1) / (Z + 1e-8) - 1) * u.reshape(B, b, 2)
        return g.mean(0)  # RAW, no schedule
    return guide


def integrate(x0_dist, guide):
    x = x0_dist.sample(N).to(DEVICE)
    ts = torch.linspace(0, T_END, NUM_STEPS)
    for k in range(NUM_STEPS - 1):
        t, dt = ts[k], ts[k + 1] - ts[k]
        with torch.no_grad():
            v = MODEL(torch.cat([x, t.expand(x.shape[0], 1)], -1))
        x = x + dt * (v + guide(t, x, v, MODEL))
    return x.detach()


def main():
    deterministic(0)
    global MODEL
    x0_dist = get_distribution("uniform")
    x1_dist = get_distribution("8gaussian")
    x1_dist.set_energy("three_blobs_continuous_multi_well", target_indices=TARGETS, std_target=SHARP_W)
    truth = rejection_sample_tilted(x1_dist, N, scale=SCALE, device=DEVICE)
    floor = compute_w2(truth, rejection_sample_tilted(x1_dist, N, scale=SCALE, device=DEVICE))

    MODEL = load("cfm")  # INDEPENDENT model: g^MC should recover truth here

    runs = [
        ("kernel std=0.1", mc_guide_kernel(x0_dist, x1_dist, 0.1, "k0.1")),
        ("kernel std=0.3", mc_guide_kernel(x0_dist, x1_dist, 0.3, "k0.3")),
        ("kernel std=0.1 resampled", mc_guide_kernel(x0_dist, x1_dist, 0.1, "k0.1rs", resample=True)),
        ("analytic (exact src)", mc_guide_analytic(x0_dist, x1_dist, "analytic")),
    ]
    samples = {}
    print(f"floor W2 = {floor:.4f}\n")
    for name, g in runs:
        deterministic(0)
        s = integrate(x0_dist, g)
        nf = 1 - torch.isfinite(s).all(-1).float().mean().item()
        sf = s[torch.isfinite(s).all(-1)]
        w2 = compute_w2(sf, truth)
        dtr = (sf.var(0).sum() / truth.var(0).sum()).item()
        samples[name] = sf
        print(f"{name:26s}  W2={w2:.4f}  div_trace={dtr:.2f}  nonfinite={nf:.0%}")

    # ESS summary
    print("\nmean ESS over trajectory (out of B=1000 coupling samples):")
    for tag, log in ess_log.items():
        vals = torch.tensor([e for _, e in log])
        print(f"  {tag:10s}  mean={vals.mean():.1f}  min={vals.min():.1f}  max={vals.max():.1f}")

    # figure: samples
    fig, axes = plt.subplots(1, len(runs) + 1, figsize=(4 * (len(runs) + 1), 4), squeeze=False)
    t = truth.cpu().numpy()
    axes[0][0].scatter(t[:, 0], t[:, 1], s=6, c="k", alpha=0.4); axes[0][0].set_title("truth p~1")
    for ax, (name, _) in zip(axes[0][1:], runs):
        s = samples[name].cpu().numpy()
        ax.scatter(t[:, 0], t[:, 1], s=5, c="k", alpha=0.15)
        ax.scatter(s[:, 0], s[:, 1], s=5, c="tab:blue", alpha=0.4)
        ax.set_title(name, fontsize=9)
    for ax in axes[0]:
        ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.6, 1.6); ax.set_aspect("equal")
    fig.suptitle("g^MC debug (independent model): kernel vs analytic estimator")
    fig.tight_layout(); fig.savefig(f"{SPIKE}/debug_mc_samples.png", dpi=140)

    # figure: ESS vs t
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    for tag, log in ess_log.items():
        ts = [tt for tt, _ in log]; es = [e for _, e in log]
        ax2.plot(ts, es, label=tag)
    ax2.set_xlabel("t"); ax2.set_ylabel("mean ESS"); ax2.set_yscale("log"); ax2.legend(); ax2.set_title("effective sample size")
    fig2.tight_layout(); fig2.savefig(f"{SPIKE}/debug_mc_ess.png", dpi=140)
    print(f"\n[fig] {SPIKE}/debug_mc_samples.png  +  debug_mc_ess.png")


if __name__ == "__main__":
    main()
