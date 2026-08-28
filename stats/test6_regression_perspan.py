"""Test 6: per-span version of the H1 regression -- does the magnitude of curvature change in each
span predict behavioural pragmatic uptake, controlling for model size? Bonferroni over the 4 spans.

Same model as the composite regression, run separately per span:
    dprag ~ curv_change_span + log(size)
  curv_change_span = mean over depth of |curv4 - curv0|  (magnitude, deg; per span)
  dprag            = pragmaticity(4-shot) - pragmaticity(0-shot)
Report the curvature coefficient beta (per deg), its parametric p (multiple-regression t-test) and
incremental R^2 (variance added beyond a size-only model). p is Bonferroni-adjusted across the 4
spans (min(1, 4*p); sig at 0.05). A permutation p on the size-partialled association is added as a
robustness check (rank-based Spearman-partial too), since the H1 effect is tail-sensitive.
41-model reported panel. Runnable: python test6_regression_perspan.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
import data as D

HERE = Path(__file__).resolve().parent
PLOTS = HERE / "plots"; PLOTS.mkdir(exist_ok=True)
SPANS = ["I", "QUD_polar", "PU", "U"]; SPANLAB = {"I": "I", "QUD_polar": "QUD", "PU": "PU", "U": "U"}
NSPAN = len(SPANS)
NPERM = 200000
RNG = np.random.default_rng(0)


def curv_change_span(a, span):
    c0, c4 = D.curv_prof(a, 0, span), D.curv_prof(a, 4, span)
    if c0 is None or c4 is None:
        return None
    return float(np.mean(np.abs(c4 - c0)))


def resid(v, z):
    Z = np.column_stack([np.ones_like(z), z]); b, *_ = np.linalg.lstsq(Z, v, rcond=None)
    return v - Z @ b


def perm_p_partial(x, y, z):
    """Permutation p for the size-partialled x-y association (residualise on z, shuffle y-resid)."""
    xr, yr = resid(x, z), resid(y, z)
    xc = xr - xr.mean(); yc = yr - yr.mean()
    tgt = abs(xc @ yc); ge = 0
    for _ in range(NPERM):
        yp = RNG.permutation(yr); ypc = yp - yp.mean()
        if abs(xc @ ypc) >= tgt - 1e-12:
            ge += 1
    r = float(xc @ yc / (np.linalg.norm(xc) * np.linalg.norm(yc)))
    return r, (1 + ge) / (1 + NPERM)


def main():
    mt = D.master().set_index("model")
    panel = [a for a in mt.index if a not in D.PANEL_EXCLUDE]
    rows, store = [], {}
    for span in SPANS:
        recs = [(a, curv_change_span(a, span), mt.loc[a, "dprag"], mt.loc[a, "logsize"]) for a in panel]
        recs = [r for r in recs if r[1] is not None]
        x = np.array([r[1] for r in recs]); y = np.array([r[2] for r in recs]); ls = np.array([r[3] for r in recs])
        n = len(recs)
        # multiple regression y ~ 1 + x + logsize
        X = np.column_stack([np.ones(n), x, ls])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        yh = X @ beta; rss = float(((y - yh) ** 2).sum()); dfres = n - 3
        se = np.sqrt(rss / dfres * np.diag(np.linalg.inv(X.T @ X)))
        tval = beta / se; pval = 2 * stats.t.sf(np.abs(tval), dfres)
        b_curv, p_curv = beta[1], pval[1]
        tss = float(((y - y.mean()) ** 2).sum()); r2 = 1 - rss / tss
        Xs = np.column_stack([np.ones(n), ls]); bs, *_ = np.linalg.lstsq(Xs, y, rcond=None)
        r2_size = 1 - float(((y - Xs @ bs) ** 2).sum()) / tss
        incr = r2 - r2_size
        # robustness
        r_par, p_perm = perm_p_partial(x, y, ls)
        rho, p_sp = stats.spearmanr(resid(x, ls), resid(y, ls))
        rows.append(dict(span=SPANLAB[span], n=n, beta_curv=b_curv, p_reg=p_curv,
                         p_reg_bonf=min(1, NSPAN * p_curv), incrR2=incr,
                         p_perm_bonf=min(1, NSPAN * p_perm), spearman_partial=rho,
                         p_spear_bonf=min(1, NSPAN * p_sp)))
        store[span] = (x, y, ls, b_curv, min(1, NSPAN * p_curv), incr)
    tbl = pd.DataFrame(rows)
    tbl.to_csv(HERE / "test6_results.csv", index=False)
    with pd.option_context("display.width", 220):
        print(f"per-span regression  dprag ~ curv_change + logsize,  n={rows[0]['n']}, Bonferroni /{NSPAN}\n")
        print(tbl.to_string(index=False,
              formatters={"beta_curv": "{:+.4f}".format, "p_reg": "{:.3f}".format,
                          "p_reg_bonf": "{:.3f}".format, "incrR2": "{:.3f}".format,
                          "p_perm_bonf": "{:.3f}".format, "spearman_partial": "{:+.2f}".format,
                          "p_spear_bonf": "{:.3f}".format}))

    # ---- added-variable (size-residualised) scatter per span
    fig, axes = plt.subplots(2, 2, figsize=(11, 9)); axes = axes.ravel()
    for j, span in enumerate(SPANS):
        ax = axes[j]; x, y, ls, b, pb, incr = store[span]
        xr, yr = resid(x, ls), resid(y, ls)
        ax.scatter(xr, yr, s=42, color="#C44E52", edgecolor="white", linewidth=.6, zorder=3)
        sl, ic = np.polyfit(xr, yr, 1); xs = np.linspace(xr.min(), xr.max(), 50)
        ax.plot(xs, sl * xs + ic, "--", color="#333", lw=1.4, zorder=2)
        ax.axhline(0, ls=":", c="#bbb", lw=1); ax.axvline(0, ls=":", c="#bbb", lw=1)
        sig = "*" if pb < 0.05 else "ns"
        ax.text(0.04, 0.96, f"β={b:+.4f}/deg\nΔR²={incr:.2f}\nBonf p={pb:.2g} {sig}",
                transform=ax.transAxes, va="top", fontsize=10,
                bbox=dict(boxstyle="round,pad=.35", fc="white", ec="#ccc"))
        ax.set_title(SPANLAB[span], fontweight="bold")
        if j % 2 == 0: ax.set_ylabel("Δ pragmaticity (size-residualised)")
        if j >= 2: ax.set_xlabel("curvature change |Δc| (size-residualised)")
    fig.suptitle(f"Per-span regression: pragmatic uptake vs curvature change ({rows[0]['n']} models, size-controlled)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = PLOTS / "test6_regression_perspan.png"
    fig.savefig(out, dpi=150, facecolor="white", bbox_inches="tight")
    print("\nwrote", out)


if __name__ == "__main__":
    main()
