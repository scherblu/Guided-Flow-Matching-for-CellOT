"""Strength-response + NFE-matched evaluation drivers for the Phase-1 toy gate.

These wrap `utils.metrics` (W2 + diversity-trace) around a caller-supplied sample function
so they stay agnostic to *how* a method generates samples (the spike's inline Euler loop,
`GuidedMLPWrapper` + torchdyn, ...). The only contract is the sample callback signature,
documented per driver below.

Two metrics drivers (PHASE1_CHECKLIST.md sec.2):
  - `strength_response`: sweep a method's knob, score W2 + diversity-trace vs the knob
    (this doubles as the controllability demo). knobs: B -> w in {0,1,2,4};
    C (g_cov_A) -> lambda-peak schedule; D (g^MC) -> N in {32,128,1024}.
  - `nfe_matched`: re-evaluate a method at fixed *total function-evaluation* budgets so the
    visual grid isn't misleading (a 1-eval/step method and a 2-eval/step method at the same
    `num_steps` are not a fair comparison).

NFE bookkeeping (evals per ODE step, per method):
  A conditional OT-CFM = 1 (one network eval)         -> T = NFE
  B CFG               = 2 (cond + uncond)             -> T = NFE / 2
  C g_cov_A           = 1 (+ closed-form grad J, ~0)  -> T ~ NFE
  D g^MC              = 1 + K (K = extra MC passes)   -> T = NFE / (1 + K)
"""
import numpy as np

from .metrics import evaluate_samples


# ----- NFE accounting ------------------------------------------------------------------
# Network forward evals per integration step, for the knob-free part of each method.
# Method D (g^MC) is 1 + K and depends on the MC config, so it is not a constant here --
# pass its value through `evals_per_step(..., mc_k=K)`.
EVALS_PER_STEP = {
    "A": 1,   # conditional OT-CFM
    "B": 2,   # CFG: cond + uncond
    "C": 1,   # g_cov_A: one eval, closed-form energy gradient is negligible
}


def evals_per_step(method, mc_k=None):
    """Function evals per ODE step for `method` ('A'|'B'|'C'|'D').

    'D' (g^MC) costs 1 + `mc_k`, where mc_k is the number of extra forward/log-prob passes
    the MC estimator makes per step; it must be supplied explicitly.
    """
    if method == "D":
        if mc_k is None:
            raise ValueError("method 'D' (g^MC) needs mc_k (the K in 1+K evals/step)")
        return 1 + mc_k
    if method not in EVALS_PER_STEP:
        raise ValueError(f"unknown method {method!r}; expected one of A/B/C/D")
    return EVALS_PER_STEP[method]


def steps_for_budget(nfe_budget, per_step):
    """Number of integration steps that spends ~`nfe_budget` total evals (>= 1)."""
    return max(1, int(round(nfe_budget / per_step)))


def total_nfe(num_steps, per_step):
    """Total function evaluations for `num_steps` steps at `per_step` evals each."""
    return num_steps * per_step


# ----- summary over seeds --------------------------------------------------------------
def _summarize(values):
    """mean / std / 95% CI half-width over a list of per-seed scalars.

    Inf survives the mean (a diverged seed makes the cell inf -- intended, not hidden);
    std/ci of a set containing inf is nan, which flags the instability.
    """
    a = np.asarray(values, dtype=float)
    n = a.shape[0]
    mean = float(a.mean())
    std = float(a.std(ddof=1)) if n > 1 else 0.0
    ci95 = 1.96 * std / np.sqrt(n) if n > 1 else 0.0
    return mean, std, float(ci95)


# ----- drivers -------------------------------------------------------------------------
def strength_response(sample_fn, knob_values, truth, seeds=(0,), knob_name="knob"):
    """Sweep a method's controllability knob; score W2 + diversity-trace (mean +/- CI over seeds).

    Args:
        sample_fn: callable(value, seed) -> Tensor (n, dim). Generates the method's samples
                   for one knob `value` under random `seed`.
        knob_values: iterable of knob settings to sweep (e.g. (0, 1, 2, 4) for CFG `w`).
        truth: Tensor (m, dim) ground-truth reference for the metrics.
        seeds: iterable of integer seeds; >=5 recommended for the gate's +/- CI.
        knob_name: label for the swept knob in the output rows.
    Returns:
        list of dict rows, one per knob value, with keys:
          {knob_name, w2_mean, w2_std, w2_ci95, div_mean, div_std, div_ci95,
           nonfinite_mean, n_seeds}.
    """
    seeds = list(seeds)
    rows = []
    for v in knob_values:
        w2s, divs, nfs = [], [], []
        for s in seeds:
            m = evaluate_samples(sample_fn(v, s), truth)
            w2s.append(m["w2"])
            divs.append(m["div_trace"])
            nfs.append(m["nonfinite"])
        w2_mean, w2_std, w2_ci = _summarize(w2s)
        div_mean, div_std, div_ci = _summarize(divs)
        rows.append({
            knob_name: v,
            "w2_mean": w2_mean, "w2_std": w2_std, "w2_ci95": w2_ci,
            "div_mean": div_mean, "div_std": div_std, "div_ci95": div_ci,
            "nonfinite_mean": float(np.mean(nfs)),
            "n_seeds": len(seeds),
        })
    return rows


def nfe_matched(sample_fn, budgets, truth, per_step, seeds=(0,)):
    """Re-score a method at fixed total-NFE budgets so the comparison grid is fair.

    For each budget the integration step count is `steps_for_budget(budget, per_step)`, i.e.
    methods that cost more per step take fewer steps for the same total compute.

    Args:
        sample_fn: callable(num_steps, seed) -> Tensor (n, dim). Generates samples using
                   exactly `num_steps` integration steps.
        budgets: iterable of total function-evaluation budgets (e.g. (50, 200, 500)).
        truth: Tensor (m, dim) ground-truth reference.
        per_step: this method's evals/step (use `evals_per_step(method, mc_k=...)`).
        seeds: iterable of integer seeds.
    Returns:
        list of dict rows, one per budget, with keys:
          {budget, num_steps, actual_nfe, w2_mean, w2_std, w2_ci95,
           div_mean, div_std, div_ci95, nonfinite_mean, n_seeds}.
    """
    seeds = list(seeds)
    rows = []
    for budget in budgets:
        num_steps = steps_for_budget(budget, per_step)
        w2s, divs, nfs = [], [], []
        for s in seeds:
            m = evaluate_samples(sample_fn(num_steps, s), truth)
            w2s.append(m["w2"])
            divs.append(m["div_trace"])
            nfs.append(m["nonfinite"])
        w2_mean, w2_std, w2_ci = _summarize(w2s)
        div_mean, div_std, div_ci = _summarize(divs)
        rows.append({
            "budget": budget,
            "num_steps": num_steps,
            "actual_nfe": total_nfe(num_steps, per_step),
            "w2_mean": w2_mean, "w2_std": w2_std, "w2_ci95": w2_ci,
            "div_mean": div_mean, "div_std": div_std, "div_ci95": div_ci,
            "nonfinite_mean": float(np.mean(nfs)),
            "n_seeds": len(seeds),
        })
    return rows
