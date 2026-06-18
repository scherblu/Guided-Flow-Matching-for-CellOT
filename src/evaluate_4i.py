"""
evaluate_4i.py -- evaluation suite for conditional / guided flow-matching on the
CellOT 4i dataset (the single, mixed-line file `8h.h5ad`).

Reference implementation of the evaluation in `PLAN_4i_fm_feasibility_prototype.md`
sec.5 -- seed `bio4i/eval.py` from this. Don't-implement-yet note: this is a
self-contained, runnable scaffold; wire `load_generated()` to your model output.

A small, biology-aware diagnostic set, reported PER CELL LINE and PER FEATURE
GROUP (protein-intensity vs morphology). The grouping/splitting is the point:
it stops easy, near-inert morphology features and an easy cell-line axis from
inflating an aggregate score.

Figures / tables produced (per cell line, per feature group):
  1. real_effect_size   -- energy distance(real drug, real control). The CEILING:
                           how much each drug actually does. A drug that barely
                           moves cells cannot show a big modelling win.
  2. per_feature_delta  -- predicted vs real mean shift-from-control, per feature.
                           Reported over ALL features AND over RESPONDERS only.
                           The gap between the two = "are results riding on a few
                           easy features". Known target markers are annotated.
  3. gap_closed         -- (E[ctrl,real] - E[pred,real]) / (E[ctrl,real] - E[noise])
                           per drug: fraction of the explainable gap the model
                           closes (1 = at the sampling noise floor; 0 = no better
                           than doing nothing). Bootstrap CIs.
  4. specificity_matrix -- energy distance gen_i x real_j; the diagonal should be
                           the row-minimum (does gen-for-drug-i look like real-i?).
  5. umap_overlay       -- per drug: grey = other cells, + real, + generated,
                           generated PROJECTED THROUGH the reducer fit on real only.

CELL LINES ARE NOT A COLUMN. `8h.h5ad` mixes two melanoma lines (M130219 +
M130429) that are separable by marker proteins SOX9 / MelA (one mesenchymal:
SOX9-high / MelA-low; one melanocytic: the reverse). We gate them with a
2-component Gaussian mixture on [SOX9, MelA] fit on the REAL cells, and apply the
same fitted gate to the generated cells.

------------------------------------------------------------------------------
HOW TO USE
  1. Point REAL_PATH at 8h.h5ad.
  2. Fill in `load_generated()` so it returns {drug_name: ndarray [n_cells, n_features]}
     with columns in the SAME ORDER as adata.var_names. (Your model already emits
     cells in feature space; just collect them per drug.)
  3. `python evaluate_4i.py`            # real run
     `python evaluate_4i.py --demo`     # smoke-test on fabricated data (no h5ad needed)

Core metrics use only numpy / sklearn / matplotlib / pandas. scanpy+anndata are
needed only to LOAD the real data; umap-learn is optional (falls back to PCA-2D).
"""

from __future__ import annotations
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------- config
REAL_PATH = "data/unprocessed/8h.h5ad"   # adjust to your path
CONTROL_NAMES = ("control", "cl")        # both treated as untreated control
OUT_DIR = "eval_4i_out"
N_SUB = 600                              # cells per group for O(n^2) energy distance
N_BOOT = 200                             # bootstrap reps for CIs / responder noise
RESPONDER_K = 2.0                        # responder if |dReal| > K * control-noise sd
RNG = np.random.default_rng(0)

# Known primary marker responses (marker, sign). Used only to ANNOTATE plots --
# extend freely. sign = +1 expected up, -1 expected down after treatment.
EXPECTED_RESPONDERS = {
    "trametinib":    [("pERK", -1)],                                 # MEK inhibitor
    "everolimus":    [("pS6k1", -1)],                                # mTOR inhibitor
    "staurosporine": [("ClCasp3", +1), ("Ki67", -1), ("PCNA", -1)],  # pan-kinase / cytotoxic
    "dabrafenib":    [("pERK", -1)],                                 # BRAF inhibitor
    "vemurafenib":   [("pERK", -1)],
}

# ----------------------------------------------------------------------------- feature parsing
def parse_var(var_names):
    """Return per-feature (group, marker, compartment). group in {protein, morphology}."""
    info = []
    for n in var_names:
        p = n.split("-")
        if n.startswith("morphology"):
            info.append(dict(name=n, group="morphology", marker=None, comp=p[1] if len(p) > 1 else None))
        elif n.startswith("intensity"):
            # intensity-<comp>-<marker>-<stat>
            comp = p[1] if len(p) > 1 else None
            marker = p[2] if len(p) > 2 else None
            info.append(dict(name=n, group="protein", marker=marker, comp=comp))
        else:
            info.append(dict(name=n, group="other", marker=None, comp=None))
    return pd.DataFrame(info)


def feature_groups(vinfo):
    return {
        "protein": np.where(vinfo["group"].values == "protein")[0],
        "morphology": np.where(vinfo["group"].values == "morphology")[0],
    }


def marker_col(vinfo, marker, comp="cell", stat="mean"):
    """Index of intensity-<comp>-<marker>-<stat>; falls back to any compartment."""
    names = vinfo["name"].values
    want = f"intensity-{comp}-{marker}-{stat}"
    hit = np.where(names == want)[0]
    if len(hit):
        return int(hit[0])
    cand = [i for i, n in enumerate(names) if f"-{marker}-" in n and n.startswith("intensity")]
    return cand[0] if cand else None


# ----------------------------------------------------------------------------- cell-line gating
def fit_cell_line_gate(X, vinfo):
    """2-component GMM on [SOX9, MelA]; returns (gmm, scaler, cols, label_map)."""
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler
    sox9 = marker_col(vinfo, "Sox9")
    mela = marker_col(vinfo, "MelA")
    if sox9 is None or mela is None:
        raise ValueError("Need SOX9 and MelA intensity features to gate cell lines.")
    cols = [sox9, mela]
    Z = StandardScaler().fit(X[:, cols])
    gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=0).fit(Z.transform(X[:, cols]))
    # label the component with higher SOX9 mean as 'mesenchymal' (SOX9+/MelA-)
    comp_sox9 = gmm.means_[:, 0]
    mesen = int(np.argmax(comp_sox9))
    label_map = {mesen: "mesenchymal", 1 - mesen: "melanocytic"}
    return gmm, Z, cols, label_map


def assign_cell_line(X, gate):
    gmm, Z, cols, label_map = gate
    comp = gmm.predict(Z.transform(X[:, cols]))
    return np.array([label_map[c] for c in comp])


# ----------------------------------------------------------------------------- metrics
def energy_distance(a, b, n_sub=N_SUB, rng=RNG):
    """sqrt(2 E||x-y|| - E||x-x'|| - E||y-y'||). Equal subsample for an n-fair estimate."""
    if len(a) == 0 or len(b) == 0:
        return np.nan
    n = min(len(a), len(b), n_sub)
    a = a[rng.choice(len(a), n, replace=len(a) < n)]
    b = b[rng.choice(len(b), n, replace=len(b) < n)]
    from scipy.spatial.distance import cdist
    dab = cdist(a, b).mean()
    daa = cdist(a, a).mean()
    dbb = cdist(b, b).mean()
    return float(np.sqrt(max(2 * dab - daa - dbb, 0.0)))


def energy_test(a, b, n_perm=200, n_sub=N_SUB, rng=RNG):
    """Permutation p-value for E-distance > 0."""
    obs = energy_distance(a, b, n_sub, rng)
    pooled = np.vstack([a, b]); na = len(a)
    cnt = 0
    for _ in range(n_perm):
        idx = rng.permutation(len(pooled))
        if energy_distance(pooled[idx[:na]], pooled[idx[na:]], n_sub, rng) >= obs:
            cnt += 1
    return obs, (cnt + 1) / (n_perm + 1)


def boot_ci(fn, reps=N_BOOT, rng=RNG, alpha=0.05):
    vals = np.array([fn(rng) for _ in range(reps)])
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return np.nan, np.nan
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def responder_mask(real_drug, real_ctrl, k=RESPONDER_K, n_boot=N_BOOT, rng=RNG):
    """A feature responds if |mean(drug)-mean(ctrl)| exceeds k * bootstrap-sd of the
    control mean (i.e. clearly beyond the noise of estimating a mean from controls)."""
    d = real_drug.mean(0) - real_ctrl.mean(0)
    nc = len(real_ctrl)
    boot = np.array([real_ctrl[rng.integers(0, nc, nc)].mean(0) for _ in range(n_boot)])
    sd = boot.std(0) + 1e-12
    return np.abs(d) > k * sd, d, sd


# ----------------------------------------------------------------------------- per-drug evaluation
def evaluate(real_X, real_drug, real_line, gen, gen_line, vinfo, control_mask_fn,
             out_dir=OUT_DIR, rng=RNG):
    groups = feature_groups(vinfo)
    lines = sorted(set(real_line))
    drugs = [d for d in sorted(set(real_drug)) if d in gen]
    rows = []
    for line in lines:
        for gname, gidx in groups.items():
            ctrl = real_X[(real_line == line) & control_mask_fn(real_drug)][:, gidx]
            if len(ctrl) < 20:
                continue
            for d in drugs:
                rd = real_X[(real_line == line) & (real_drug == d)][:, gidx]
                gd = gen[d][gen_line[d] == line][:, gidx]
                if len(rd) < 20 or len(gd) < 20:
                    continue
                e_pred = energy_distance(gd, rd, rng=rng)
                e_base = energy_distance(ctrl, rd, rng=rng)           # = real effect size
                h = len(rd) // 2
                e_noise = energy_distance(rd[:h], rd[h:], rng=rng)    # sampling floor
                denom = e_base - e_noise
                gap = (e_base - e_pred) / denom if denom > 1e-9 else np.nan
                lo, hi = boot_ci(lambda r: (energy_distance(ctrl, rd, rng=r) - energy_distance(gd, rd, rng=r))
                                 / max(energy_distance(ctrl, rd, rng=r) - e_noise, 1e-9), rng=rng)
                mask, dreal, _ = responder_mask(rd, ctrl, rng=rng)
                dpred = gd.mean(0) - ctrl.mean(0)
                r_all = np.corrcoef(dreal, dpred)[0, 1] if len(dreal) > 2 else np.nan
                r_resp = (np.corrcoef(dreal[mask], dpred[mask])[0, 1]
                          if mask.sum() > 2 else np.nan)
                rows.append(dict(line=line, group=gname, drug=d,
                                 real_effect=e_base, e_pred=e_pred, e_noise=e_noise,
                                 gap_closed=gap, gap_lo=lo, gap_hi=hi,
                                 n_responders=int(mask.sum()), n_feat=len(gidx),
                                 r_all=r_all, r_resp=r_resp))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- plots
def plot_gap_and_effect(df, out_dir):
    for (line, group), sub in df.groupby(["line", "group"]):
        sub = sub.sort_values("real_effect", ascending=False)
        fig, ax = plt.subplots(1, 2, figsize=(11, 0.5 + 0.5 * len(sub)))
        ax[0].barh(sub["drug"], sub["real_effect"], color="0.6")
        ax[0].set_title("real effect size  E(real drug, real control)")
        ax[0].invert_yaxis()
        err = np.vstack([sub["gap_closed"] - sub["gap_lo"], sub["gap_hi"] - sub["gap_closed"]])
        ax[1].barh(sub["drug"], sub["gap_closed"], xerr=np.abs(err), color="#4c78a8")
        ax[1].axvline(1.0, ls="--", c="k", lw=1); ax[1].axvline(0.0, ls=":", c="0.5", lw=1)
        ax[1].set_xlim(min(-0.1, sub["gap_lo"].min()), 1.1)
        ax[1].set_title("gap closed  (1=noise floor, 0=no better than control)")
        ax[1].invert_yaxis()
        fig.suptitle(f"{line} | {group}")
        fig.tight_layout(); fig.savefig(f"{out_dir}/01_effect_gap_{line}_{group}.png", dpi=110)
        plt.close(fig)


def plot_per_feature_delta(real_X, real_drug, real_line, gen, gen_line, vinfo,
                           control_mask_fn, out_dir, rng=RNG):
    groups = feature_groups(vinfo)
    names = vinfo["name"].values
    for line in sorted(set(real_line)):
        for d in [x for x in sorted(set(real_drug)) if x in gen]:
            fig, axes = plt.subplots(1, len(groups), figsize=(6 * len(groups), 5.2))
            for ax, (gname, gidx) in zip(np.atleast_1d(axes), groups.items()):
                ctrl = real_X[(real_line == line) & control_mask_fn(real_drug)][:, gidx]
                rd = real_X[(real_line == line) & (real_drug == d)][:, gidx]
                gd = gen[d][gen_line[d] == line][:, gidx]
                if len(rd) < 20 or len(gd) < 20 or len(ctrl) < 20:
                    ax.set_visible(False); continue
                mask, dreal, _ = responder_mask(rd, ctrl, rng=rng)
                dpred = gd.mean(0) - ctrl.mean(0)
                ax.scatter(dreal[~mask], dpred[~mask], s=18, c="0.7", label="inert")
                ax.scatter(dreal[mask], dpred[mask], s=30, c="#e45756", label="responder")
                lim = np.abs(np.concatenate([dreal, dpred])).max() * 1.1 + 1e-9
                ax.plot([-lim, lim], [-lim, lim], "k--", lw=1)
                ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
                ax.set_xlabel("real shift-from-control"); ax.set_ylabel("predicted shift")
                r_all = np.corrcoef(dreal, dpred)[0, 1]
                r_resp = np.corrcoef(dreal[mask], dpred[mask])[0, 1] if mask.sum() > 2 else np.nan
                ax.set_title(f"{gname}: r_all={r_all:.2f}  r_resp={r_resp:.2f}  (n_resp={mask.sum()})")
                # annotate known target markers
                gnames = names[gidx]
                for mk, sign in EXPECTED_RESPONDERS.get(d, []):
                    hit = [j for j, nm in enumerate(gnames) if f"-{mk}-" in nm]
                    for j in hit:
                        ax.annotate(mk, (dreal[j], dpred[j]), fontsize=8, color="navy")
                ax.legend(fontsize=8)
            fig.suptitle(f"per-feature shift recovery | {line} | {d}")
            fig.tight_layout(); fig.savefig(f"{out_dir}/02_delta_{line}_{d}.png", dpi=110)
            plt.close(fig)


def plot_specificity(real_X, real_drug, real_line, gen, gen_line, vinfo, out_dir, rng=RNG):
    gidx = feature_groups(vinfo)["protein"]   # specificity on the biological signal
    for line in sorted(set(real_line)):
        drugs = [x for x in sorted(set(real_drug)) if x in gen]
        M = np.full((len(drugs), len(drugs)), np.nan)
        for i, di in enumerate(drugs):
            gi = gen[di][gen_line[di] == line][:, gidx]
            for j, dj in enumerate(drugs):
                rj = real_X[(real_line == line) & (real_drug == dj)][:, gidx]
                if len(gi) >= 20 and len(rj) >= 20:
                    M[i, j] = energy_distance(gi, rj, rng=rng)
        valid = ~np.all(np.isnan(M), axis=1)
        if valid.sum() < 2:
            continue
        argmin_diag = np.nanmean([np.nanargmin(M[i]) == i for i in range(len(drugs)) if valid[i]])
        fig, ax = plt.subplots(figsize=(1.2 + 0.7 * len(drugs), 1.2 + 0.7 * len(drugs)))
        im = ax.imshow(M, cmap="viridis")
        ax.set_xticks(range(len(drugs))); ax.set_xticklabels(drugs, rotation=45, ha="right")
        ax.set_yticks(range(len(drugs))); ax.set_yticklabels(drugs)
        ax.set_xlabel("real drug j"); ax.set_ylabel("generated drug i")
        for i in range(len(drugs)):
            for j in range(len(drugs)):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                            color="w", fontsize=7)
        ax.set_title(f"specificity (protein) | {line}\ndiagonal is row-min in {argmin_diag:.0%} of rows")
        fig.colorbar(im); fig.tight_layout()
        fig.savefig(f"{out_dir}/03_specificity_{line}.png", dpi=110); plt.close(fig)


def plot_umap_overlay(real_X, real_drug, real_line, gen, gen_line, vinfo, out_dir, rng=RNG):
    gidx = feature_groups(vinfo)["protein"]
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    for line in sorted(set(real_line)):
        rmask = real_line == line
        Xr = real_X[rmask][:, gidx]
        sc = StandardScaler().fit(Xr)
        pca = PCA(n_components=min(20, Xr.shape[1])).fit(sc.transform(Xr))
        try:
            import umap
            reducer = umap.UMAP(random_state=0).fit(pca.transform(sc.transform(Xr)))
            emb_r = reducer.embedding_
            proj = lambda Z: reducer.transform(pca.transform(sc.transform(Z)))
            tag = "UMAP"
        except Exception:
            emb_r = pca.transform(sc.transform(Xr))[:, :2]
            proj = lambda Z: pca.transform(sc.transform(Z))[:, :2]
            tag = "PCA-2D"
        rdrug = real_drug[rmask]
        for d in [x for x in sorted(set(real_drug)) if x in gen]:
            gd = gen[d][gen_line[d] == line]
            if len(gd) < 20:
                continue
            emb_g = proj(gd[:, gidx])
            fig, ax = plt.subplots(figsize=(6, 5.5))
            ax.scatter(emb_r[:, 0], emb_r[:, 1], s=4, c="0.82", label="other real")
            sel = rdrug == d
            ax.scatter(emb_r[sel, 0], emb_r[sel, 1], s=8, c="#4c78a8", label=f"real {d}")
            ax.scatter(emb_g[:, 0], emb_g[:, 1], s=8, c="#e45756", label=f"generated {d}")
            ax.set_title(f"{tag} overlay | {line} | {d}"); ax.legend(fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
            fig.tight_layout(); fig.savefig(f"{out_dir}/04_umap_{line}_{d}.png", dpi=110)
            plt.close(fig)


# ----------------------------------------------------------------------------- data loading
def load_real(path):
    import scanpy as sc
    adata = sc.read_h5ad(path)
    X = np.asarray(adata.X.todense()) if hasattr(adata.X, "todense") else np.asarray(adata.X)
    return X.astype(np.float32), adata.obs["drug"].astype(str).values, list(adata.var_names)


def load_generated(var_names):
    """TODO: return {drug_name: ndarray [n_cells, len(var_names)]} from your model,
    columns in adata.var_names order. Raise until implemented."""
    raise NotImplementedError("Fill load_generated() with your model's per-drug samples.")


def make_demo(n_per=800, seed=0):
    """Fabricate a 2-line, multi-drug dataset + a plausible 'model' to exercise the pipeline."""
    rng = np.random.default_rng(seed)
    markers = ["CD45", "ClCasp3", "DAPI", "Ki67", "MelA", "PCNA", "Sox9", "aTUB",
               "pAKT", "pEGFR", "pERK", "pMET", "pS6k1"]
    var = ([f"intensity-cell-{m}-mean" for m in markers]
           + [f"intensity-nuclei-{m}-mean" for m in markers]
           + [f"morphology-cell-{a}" for a in ["area", "perimeter", "eccentricity", "roundness",
                                               "circularity", "convexity", "extent", "elongation",
                                               "equivalent_diameter", "mean_radius", "major_axis_length"]]
           + [f"morphology-nuclei-{a}" for a in ["area", "perimeter", "eccentricity", "roundness",
                                                 "circularity", "convexity", "extent", "elongation",
                                                 "equivalent_diameter", "mean_radius", "major_axis_length"]])
    vinfo = parse_var(var); nf = len(var)
    idx = {n: i for i, n in enumerate(var)}

    def base(line):
        x = rng.normal(0, 1, (n_per, nf)).astype(np.float32)
        # encode the two lines via SOX9/MelA (cell + nuclei)
        for comp in ("cell", "nuclei"):
            if line == "mesenchymal":
                x[:, idx[f"intensity-{comp}-Sox9-mean"]] += 3.0
                x[:, idx[f"intensity-{comp}-MelA-mean"]] -= 2.0
            else:
                x[:, idx[f"intensity-{comp}-Sox9-mean"]] -= 2.0
                x[:, idx[f"intensity-{comp}-MelA-mean"]] += 3.0
        return x

    effects = {  # true per-drug marker shifts (cell-mean), morphology mostly inert
        "control": {},
        "trametinib": {"pERK": -1.5},
        "everolimus": {"pS6k1": -1.4},
        "staurosporine": {"ClCasp3": +2.5, "Ki67": -1.8, "PCNA": -1.5, "morphology-cell-area": -1.2},
    }
    real_X, real_drug, real_line, gen, gen_line = [], [], [], {}, {}
    for line in ("mesenchymal", "melanocytic"):
        for d, eff in effects.items():
            x = base(line)
            for k, v in eff.items():
                col = idx.get(k, idx.get(f"intensity-cell-{k}-mean"))
                if col is not None:
                    x[:, col] += v
            real_X.append(x); real_drug += [d] * n_per; real_line += [line] * n_per
    real_X = np.vstack(real_X)
    real_drug = np.array(real_drug); real_line = np.array(real_line)
    # a 'decent model': reproduce real per (line,drug) with noise + slight under-shoot
    for d, eff in effects.items():
        if d == "control":
            continue
        parts, lines_g = [], []
        for line in ("mesenchymal", "melanocytic"):
            r = real_X[(real_line == line) & (real_drug == d)]
            g = 0.85 * r + 0.15 * r.mean(0) + rng.normal(0, 0.4, r.shape)
            parts.append(g.astype(np.float32)); lines_g += [line] * len(g)
        gen[d] = np.vstack(parts); gen_line[d] = np.array(lines_g)
    return real_X, real_drug, var, vinfo, gen, gen_line


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="run on fabricated data")
    ap.add_argument("--out", default=OUT_DIR)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    control_mask_fn = lambda dr: np.isin(dr, CONTROL_NAMES)

    if args.demo:
        real_X, real_drug, var_names, vinfo, gen, gen_line = make_demo()
        gate = fit_cell_line_gate(real_X, vinfo)
        real_line = assign_cell_line(real_X, gate)
        gen_line = {d: assign_cell_line(g, gate) for d, g in gen.items()}
    else:
        real_X, real_drug, var_names = load_real(REAL_PATH)
        vinfo = parse_var(var_names)
        gate = fit_cell_line_gate(real_X, vinfo)          # fit on real
        real_line = assign_cell_line(real_X, gate)
        gen = load_generated(var_names)
        gen_line = {d: assign_cell_line(g, gate) for d, g in gen.items()}  # same gate
        print("real cells per line:", dict(zip(*np.unique(real_line, return_counts=True))))

    df = evaluate(real_X, real_drug, real_line, gen, gen_line, vinfo, control_mask_fn, args.out)
    df.to_csv(f"{args.out}/metrics.csv", index=False)
    print("\n=== per (line, group, drug) metrics ===")
    with pd.option_context("display.width", 160, "display.max_columns", None):
        print(df.round(3).to_string(index=False))

    plot_gap_and_effect(df, args.out)
    plot_per_feature_delta(real_X, real_drug, real_line, gen, gen_line, vinfo, control_mask_fn, args.out)
    plot_specificity(real_X, real_drug, real_line, gen, gen_line, vinfo, args.out)
    plot_umap_overlay(real_X, real_drug, real_line, gen, gen_line, vinfo, args.out)
    print(f"\nFigures + metrics.csv written to {args.out}/")


if __name__ == "__main__":
    main()