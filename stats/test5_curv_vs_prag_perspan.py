"""Test 5: is the per-span curvature straightening related to behavioural pragmatic uptake, across
models -- tested SEPARATELY per span with Bonferroni over the four spans.

Per model, per span:
  x = signed straightening = mean over depth of (curv4 - surro4) - (curv0 - surro0)   [deg, <0 = straighter]
  y = Delta pragmaticity  = pragmaticity(4-shot) - pragmaticity(0-shot)
Per span we test the x-y association two ways:
  raw     Pearson r(x, y)
  partial partial correlation controlling log(model size): residualise x and y on logsize, correlate.
Null by permutation (shuffle y / y-residual, NPERM), two-sided; p Bonferroni-adjusted across 4 spans
(adjusted p = min(1, 4*p_raw); significant at alpha=0.05). 41-model reported panel.
Runnable: python test5_curv_vs_prag_perspan.py
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
GRID = D.GRID
NPERM = 200000
NSPAN = len(SPANS)
RNG = np.random.default_rng(0)


def resid(v, z):
    """residual of v after regressing on z (with intercept)."""
    Z = np.column_stack([np.ones_like(z), z])
    beta, *_ = np.linalg.lstsq(Z, v, rcond=None)
    return v - Z @ beta


def straighten(a, span):
    c0, c4 = D.curv_prof(a, 0, span), D.curv_prof(a, 4, span)
    s0, s4 = D.surro_prof(a, 0, span), D.surro_prof(a, 4, span)
    if any(x is None for x in (c0, c4, s0, s4)):
        return None
    return float(((c4 - s4) - (c0 - s0)).mean())


def perm_p_corr(x, y):
    """Two-sided permutation p for Pearson r(x,y), shuffling y. Returns (r, p)."""
    xc = x - x.mean(); yc = y - y.mean()
    denom = np.linalg.norm(xc) * np.linalg.norm(yc)
    obs = float(xc @ yc / denom)
    tgt = abs(xc @ yc)
    ge = 0
    for _ in range(NPERM):
        yp = RNG.permutation(y)
        if abs(xc @ (yp - yp.mean())) >= tgt - 1e-12:
            ge += 1
    return obs, (1 + ge) / (1 + NPERM)


def main():
    mt = D.master().set_index("model")
    panel = [a for a in mt.index if a not in D.PANEL_EXCLUDE]
    rows, store = [], {}
    for span in SPANS:
        recs = [(a, straighten(a, span), mt.loc[a, "dprag"], mt.loc[a, "logsize"]) for a in panel]
        recs = [(a, s, dp, ls) for (a, s, dp, ls) in recs if s is not None]
        names = [a for a, *_ in recs]
        x = np.array([r[1] for r in recs]); y = np.array([r[2] for r in recs]); ls = np.array([r[3] for r in recs])
        r_raw, p_raw = perm_p_corr(x, y)
        xr, yr = resid(x, ls), resid(y, ls)
        r_par, p_par = perm_p_corr(xr, yr)
        rows.append(dict(span=SPANLAB[span], n=len(names),
                         r_raw=r_raw, p_raw=p_raw, p_raw_bonf=min(1, NSPAN * p_raw),
                         r_partial=r_par, p_partial=p_par, p_partial_bonf=min(1, NSPAN * p_par)))
        store[span] = (x, y, ls, r_raw, min(1, NSPAN * p_raw))
    tbl = pd.DataFrame(rows)
    tbl.to_csv(HERE / "test5_results.csv", index=False)
    with pd.option_context("display.width", 200):
        print(f"per-span curvature straightening vs Delta-pragmaticity, n={rows[0]['n']}, "
              f"Bonferroni /{NSPAN} (adjusted p, sig at 0.05)\n")
        print(tbl.to_string(index=False,
              formatters={"r_raw": "{:+.3f}".format, "p_raw": "{:.3f}".format,
                          "p_raw_bonf": "{:.3f}".format, "r_partial": "{:+.3f}".format,
                          "p_partial": "{:.3f}".format, "p_partial_bonf": "{:.3f}".format}))

    # ---- figure: raw scatter per span
    fig, axes = plt.subplots(2, 2, figsize=(11, 9)); axes = axes.ravel()
    for j, span in enumerate(SPANS):
        ax = axes[j]; x, y, ls, r_raw, pb = store[span]
        ax.scatter(x, y, s=42, color="#4C72B0", edgecolor="white", linewidth=.6, zorder=3)
        sl, ic = np.polyfit(x, y, 1); xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, sl * xs + ic, "--", color="#333", lw=1.4, zorder=2)
        ax.axhline(0, ls=":", c="#bbb", lw=1); ax.axvline(0, ls=":", c="#bbb", lw=1)
        sig = "*" if pb < 0.05 else "ns"
        ax.text(0.04, 0.96, f"r={r_raw:+.2f}\nBonf p={pb:.2g} {sig}", transform=ax.transAxes,
                va="top", fontsize=11, bbox=dict(boxstyle="round,pad=.35", fc="white", ec="#ccc"))
        ax.set_title(SPANLAB[span], fontweight="bold")
        if j % 2 == 0: ax.set_ylabel("Δ pragmaticity (4 - 0 shot)")
        if j >= 2: ax.set_xlabel("curvature straightening (deg; <0 = straighter)")
    fig.suptitle(f"Per-span curvature straightening vs pragmatic uptake ({rows[0]['n']} models, raw)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = PLOTS / "test5_curv_vs_prag_perspan.png"
    fig.savefig(out, dpi=150, facecolor="white", bbox_inches="tight")
    print("\nwrote", out)


if __name__ == "__main__":
    main()
