"""Test 1: paired (within-model) permutation test on the curvature trajectories, 4-shot vs 0-shot,
run SEPARATELY per span (I, PU, U, QUD), pointwise across the 40 relative-depth points with a
Bonferroni correction over the 40 points within each span.

Design
  - Per model i, per span, the paired difference trajectory at depth d:
        D_i(d) = (curv4(d) - surro4(d)) - (curv0(d) - surro0(d))     [deg]
    (surrogate-normalized to match the replotted figures; for a paired diff the null ~cancels.)
  - Statistic at depth d: mean paired difference Dbar(d) across models, two-sided.
  - Null: sign-flip permutation (paired/one-sample exact scheme). Each permutation draws one sign
    vector s in {+1,-1}^n and applies it to every model's whole D_i (shared across depths, i.e. a
    permutation is one relabeling), then recomputes the per-depth mean. NPERM permutations plus the
    identity. p(d) = (1 + #{|perm(d)| >= |obs(d)|}) / (1 + NPERM).
  - Bonferroni within each span: a depth is significant if p(d) < 0.05 / 40.
Model set: all models with 0- and 4-shot curvature + surrogate (within-model test, no quartile split).
Runnable: python test1_paired_curvature_perm.py
Writes plots/test1_paired_curvature_perm.png and appends to RESULTS.md.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
import data as D

HERE = Path(__file__).resolve().parent
PLOTS = HERE / "plots"; PLOTS.mkdir(exist_ok=True)
SPANS = ["I", "PU", "U", "QUD_polar"]; SPANLAB = {"I": "I", "PU": "PU", "U": "U", "QUD_polar": "QUD"}
GRID = D.GRID; NGRID = GRID.size
NPERM = 20000
ALPHA = 0.05
BONF = ALPHA / NGRID
RNG = np.random.default_rng(0)


def paired_matrix(span):
    """[n_models, 40] paired diff D_i(d) = (curv4-surro4) - (curv0-surro0); models missing any
    piece are dropped. Returns (D, model_names)."""
    rows, names = [], []
    for a in D.SIZE:
        c0, c4 = D.curv_prof(a, 0, span), D.curv_prof(a, 4, span)
        s0, s4 = D.surro_prof(a, 0, span), D.surro_prof(a, 4, span)
        if any(x is None for x in (c0, c4, s0, s4)):
            continue
        rows.append((c4 - s4) - (c0 - s0)); names.append(a)
    return np.array(rows), names


def perm_test(Dmat):
    """Two-sided sign-flip permutation. Returns (obs, pvals) each length 40."""
    n = Dmat.shape[0]
    obs = Dmat.mean(0)
    ge = np.zeros(NGRID)
    for _ in range(NPERM):
        s = RNG.integers(0, 2, size=n) * 2 - 1          # +-1 per model
        pstat = (s[:, None] * Dmat).mean(0)
        ge += (np.abs(pstat) >= np.abs(obs) - 1e-12)
    pvals = (1.0 + ge) / (1.0 + NPERM)
    return obs, pvals


def main():
    results = {}
    for span in SPANS:
        Dmat, names = paired_matrix(span)
        obs, pvals = perm_test(Dmat)
        sig = pvals < BONF
        results[span] = dict(Dmat=Dmat, names=names, obs=obs, p=pvals, sig=sig)
        rel_sig = GRID[sig]
        print(f"[{SPANLAB[span]:3s}] n={len(names)}  sig depths (Bonf p<{BONF:.2e}): "
              f"{sig.sum()}/{NGRID}"
              + (f"  rel-depth {rel_sig.min():.2f}-{rel_sig.max():.2f}" if sig.any() else "")
              + f"  min p={pvals.min():.2e}  peak|Dbar|={np.abs(obs).max():.2f} deg", flush=True)

    # ---- figure: paired mean difference per span with Bonferroni-significant depths marked
    fig, axes = plt.subplots(2, 2, figsize=(12, 9)); axes = axes.ravel()
    for j, span in enumerate(SPANS):
        ax = axes[j]; r = results[span]
        obs, sig, Dmat = r["obs"], r["sig"], r["Dmat"]
        se = Dmat.std(0, ddof=1) / np.sqrt(Dmat.shape[0])
        ax.axhline(0, ls=":", c="#999", lw=1)
        ax.fill_between(GRID, obs - se, obs + se, color="#4C72B0", alpha=0.20, lw=0)
        ax.plot(GRID, obs, color="#4C72B0", lw=2, label="mean 4-shot - 0-shot")
        if sig.any():
            ax.plot(GRID[sig], obs[sig], "o", ms=5, color="#C44E52", zorder=5,
                    label=f"Bonferroni sig (p<{BONF:.1e})")
            for d in GRID[sig]:
                ax.axvspan(d - 0.006, d + 0.006, color="#C44E52", alpha=0.06, lw=0)
        ax.set_title(f"{SPANLAB[span]}  (n={Dmat.shape[0]}, {r['sig'].sum()}/{NGRID} sig)",
                     fontweight="bold")
        ax.grid(alpha=0.3)
        if j % 2 == 0:
            ax.set_ylabel("Δ curvature (deg): 4-shot - 0-shot")
        if j >= 2:
            ax.set_xlabel("relative depth")
        if j == 0:
            ax.legend(fontsize=9, loc="lower left")
    fig.suptitle("Paired sign-flip permutation test of the 4-shot vs 0-shot curvature trajectory, "
                 f"per span\n(pointwise, {NPERM} permutations, Bonferroni over {NGRID} depths, n={results['I']['Dmat'].shape[0]} models)",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = PLOTS / "test1_paired_curvature_perm.png"
    fig.savefig(out, dpi=150, facecolor="white", bbox_inches="tight")
    print("wrote", out)

    # ---- results table
    tbl = pd.DataFrame([{
        "span": SPANLAB[s], "n_models": results[s]["Dmat"].shape[0],
        "n_sig_depths": int(results[s]["sig"].sum()),
        "sig_reldepth_lo": float(GRID[results[s]["sig"]].min()) if results[s]["sig"].any() else np.nan,
        "sig_reldepth_hi": float(GRID[results[s]["sig"]].max()) if results[s]["sig"].any() else np.nan,
        "min_p": float(results[s]["p"].min()),
        "peak_abs_Dbar_deg": float(np.abs(results[s]["obs"]).max()),
        "bonf_thresh": BONF,
    } for s in SPANS])
    tbl.to_csv(HERE / "test1_results.csv", index=False)
    print("\n" + tbl.to_string(index=False))


if __name__ == "__main__":
    main()
