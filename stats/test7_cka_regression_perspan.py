"""Test 7: per-span version of the H1 regression for CKA -- does the magnitude of within-bracket CKA
change in each span predict behavioural pragmatic uptake, controlling for model size? Bonferroni over
the 4 spans. This is the CKA analogue of Test 6 (curvature); CKA is reported first per paper order.

Same model as the composite regression, run separately per span:
    dprag ~ cka_change_span + log(size)
  cka_change_span = off-diagonal mean of |CKA4 - CKA0|  (within-bracket, per span)
  dprag           = pragmaticity(4-shot) - pragmaticity(0-shot)
Report the CKA coefficient beta, its parametric p (multiple-regression t-test) and incremental R^2
(variance added beyond a size-only model). p is Bonferroni-adjusted across the 4 spans
(min(1, 4*p); sig at 0.05). A permutation p on the size-partialled association is added as a
robustness check (rank-based Spearman-partial too). 41-model reported panel.
Runnable: python test7_cka_regression_perspan.py
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
POOL = D.POOL
NSPAN = len(SPANS)
NPERM = 200000
RNG = np.random.default_rng(0)


def cka_change_span(a, span):
    """Off-diagonal mean of |CKA4 - CKA0| for model a, span (within-bracket). None if missing/NaN."""
    m0, m4 = D._load_cka(a, 0), D._load_cka(a, 4)
    if m0 is None or m4 is None or POOL not in m0 or POOL not in m4:
        return None
    try:
        C0 = np.asarray(m0[POOL]["within"][span], float)
        C4 = np.asarray(m4[POOL]["within"][span], float)
    except KeyError:
        return None
    if C0.shape != C4.shape or np.isnan(C0).any() or np.isnan(C4).any():
        return None
    Dm = np.abs(C4 - C0)
    off = ~np.eye(Dm.shape[0], dtype=bool)
    return float(Dm[off].mean())


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
        recs = [(a, cka_change_span(a, span), mt.loc[a, "dprag"], mt.loc[a, "logsize"]) for a in panel]
        recs = [r for r in recs if r[1] is not None]
        x = np.array([r[1] for r in recs]); y = np.array([r[2] for r in recs]); ls = np.array([r[3] for r in recs])
        n = len(recs)
        # multiple regression y ~ 1 + x + logsize
        X = np.column_stack([np.ones(n), x, ls])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        yh = X @ beta; rss = float(((y - yh) ** 2).sum()); dfres = n - 3
        se = np.sqrt(rss / dfres * np.diag(np.linalg.inv(X.T @ X)))
        tval = beta / se; pval = 2 * stats.t.sf(np.abs(tval), dfres)
        b_cka, p_cka = beta[1], pval[1]
        tss = float(((y - y.mean()) ** 2).sum()); r2 = 1 - rss / tss
        Xs = np.column_stack([np.ones(n), ls]); bs, *_ = np.linalg.lstsq(Xs, y, rcond=None)
        r2_size = 1 - float(((y - Xs @ bs) ** 2).sum()) / tss
        incr = r2 - r2_size
        # robustness
        r_par, p_perm = perm_p_partial(x, y, ls)
        rho, p_sp = stats.spearmanr(resid(x, ls), resid(y, ls))
        rows.append(dict(span=SPANLAB[span], n=n, beta_cka=b_cka, p_reg=p_cka,
                         p_reg_bonf=min(1, NSPAN * p_cka), incrR2=incr,
                         p_perm_bonf=min(1, NSPAN * p_perm), spearman_partial=rho,
                         p_spear_bonf=min(1, NSPAN * p_sp)))
        store[span] = (x, y, ls, b_cka, min(1, NSPAN * p_cka), incr)
    tbl = pd.DataFrame(rows)
    tbl.to_csv(HERE / "test7_results.csv", index=False)
    with pd.option_context("display.width", 220):
        print(f"per-span regression  dprag ~ cka_change + logsize,  n={rows[0]['n']}, Bonferroni /{NSPAN}\n")
        print(tbl.to_string(index=False,
              formatters={"beta_cka": "{:+.4f}".format, "p_reg": "{:.3f}".format,
                          "p_reg_bonf": "{:.3f}".format, "incrR2": "{:.3f}".format,
                          "p_perm_bonf": "{:.3f}".format, "spearman_partial": "{:+.2f}".format,
                          "p_spear_bonf": "{:.3f}".format}))

    # ---- added-variable (size-residualised) scatter per span
    fig, axes = plt.subplots(2, 2, figsize=(11, 9)); axes = axes.ravel()
    for j, span in enumerate(SPANS):
        ax = axes[j]; x, y, ls, b, pb, incr = store[span]
        xr, yr = resid(x, ls), resid(y, ls)
        ax.scatter(xr, yr, s=42, color="#55A868", edgecolor="white", linewidth=.6, zorder=3)
        sl, ic = np.polyfit(xr, yr, 1); xs = np.linspace(xr.min(), xr.max(), 50)
        ax.plot(xs, sl * xs + ic, "--", color="#333", lw=1.4, zorder=2)
        ax.axhline(0, ls=":", c="#bbb", lw=1); ax.axvline(0, ls=":", c="#bbb", lw=1)
        sig = "*" if pb < 0.05 else "ns"
        ax.text(0.04, 0.96, f"β={b:+.4f}\nΔR²={incr:.2f}\nBonf p={pb:.2g} {sig}",
                transform=ax.transAxes, va="top", fontsize=10,
                bbox=dict(boxstyle="round,pad=.35", fc="white", ec="#ccc"))
        ax.set_title(SPANLAB[span], fontweight="bold")
        if j % 2 == 0: ax.set_ylabel("Δ pragmaticity (size-residualised)")
        if j >= 2: ax.set_xlabel("CKA change |ΔCKA| (size-residualised)")
    fig.suptitle(f"Per-span regression: pragmatic uptake vs CKA change ({rows[0]['n']} models, size-controlled)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = PLOTS / "test7_cka_regression_perspan.png"
    fig.savefig(out, dpi=150, facecolor="white", bbox_inches="tight")
    print("\nwrote", out)


if __name__ == "__main__":
    main()
