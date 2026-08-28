"""Test 11: between-group comparison of the few-shot curvature trough-LOCATION shift (argmin), top vs
bottom pragmatic-uptake quartiles, per span, Bonferroni over the 4 spans. Argmin analogue of Test 9.

On the 41-model reported panel (v7) we rank models by pragmatic uptake
Dprag = pragmaticity(4-shot) - pragmaticity(0-shot) and split the EXTREME quartiles:
  Q4 = top 25% (strongest uptake)   vs   Q1 = bottom 25% (weakest / negative uptake);
the middle 50% is dropped (~11 per group). Per model the representational quantity is the shift in the
straightening-trough location on the [0,1] relative-depth axis:
  shift_i = GRID[argmin(curv4 - surro4)] - GRID[argmin(curv0 - surro0)]   (<0 = trough earlier at 4-sh)
Per span the statistic is the difference in group means  obs = mean(shift|Q4) - mean(shift|Q1). The null
is INDEPENDENT-GROUPS: shuffle the Q4/Q1 labels across the pooled models (NPERM), recompute; two-sided
p = (1 + #|perm| >= |obs|) / (1 + NPERM). p is Bonferroni-adjusted across the 4 spans
(min(1, 4*p); significant at 0.05). Welch t as a parametric cross-check. Argmin/curvature only.
Runnable: python test11_group_argmin_quartiles.py
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
GRID = D.GRID
SPANS = ["I", "QUD_polar", "PU", "U"]; SPANLAB = {"I": "I", "QUD_polar": "QUD", "PU": "PU", "U": "U"}
NSPAN = len(SPANS)
NPERM = 200000
RNG = np.random.default_rng(0)


def argmin_shift(a, span):
    """GRID[argmin(curv4-surro4)] - GRID[argmin(curv0-surro0)]; None if any profile missing."""
    c0, c4 = D.curv_prof(a, 0, span), D.curv_prof(a, 4, span)
    s0, s4 = D.surro_prof(a, 0, span), D.surro_prof(a, 4, span)
    if any(x is None for x in (c0, c4, s0, s4)):
        return None
    l0 = GRID[int(np.argmin(c0 - s0))]; l4 = GRID[int(np.argmin(c4 - s4))]
    return float(l4 - l0)


def perm_group_diff(x_hi, x_lo):
    """Two-sided label-permutation p for mean(hi) - mean(lo)."""
    obs = x_hi.mean() - x_lo.mean()
    pool = np.concatenate([x_hi, x_lo]); n_hi = x_hi.size; n = pool.size
    ge = 0
    for _ in range(NPERM):
        idx = RNG.permutation(n)
        d = pool[idx[:n_hi]].mean() - pool[idx[n_hi:]].mean()
        if abs(d) >= abs(obs) - 1e-15:
            ge += 1
    return obs, (1 + ge) / (1 + NPERM)


def main():
    mt = D.master().set_index("model")
    panel = [a for a in mt.index if a not in D.PANEL_EXCLUDE]
    dprag = mt.loc[panel, "dprag"]
    q1, q3 = np.quantile(dprag.values, 0.25), np.quantile(dprag.values, 0.75)
    HIGH = [a for a in panel if mt.loc[a, "dprag"] >= q3]          # Q4 top 25%
    LOW = [a for a in panel if mt.loc[a, "dprag"] <= q1]           # Q1 bottom 25%
    print(f"panel n={len(panel)}  |  Q4 (Dprag>={q3:+.3f}) n={len(HIGH)}  "
          f"vs  Q1 (Dprag<={q1:+.3f}) n={len(LOW)}\n")

    rows, store = [], {}
    for span in SPANS:
        xhi = np.array([v for v in (argmin_shift(a, span) for a in HIGH) if v is not None])
        xlo = np.array([v for v in (argmin_shift(a, span) for a in LOW) if v is not None])
        obs, p_perm = perm_group_diff(xhi, xlo)
        t, p_t = stats.ttest_ind(xhi, xlo, equal_var=False)
        rows.append(dict(span=SPANLAB[span], n_hi=xhi.size, n_lo=xlo.size,
                         mean_hi=xhi.mean(), mean_lo=xlo.mean(), diff=obs,
                         p_perm=p_perm, p_perm_bonf=min(1, NSPAN * p_perm),
                         p_welch=float(p_t)))
        store[span] = (xhi, xlo)
    tbl = pd.DataFrame(rows)
    tbl.to_csv(HERE / "test11_results.csv", index=False)
    with pd.option_context("display.width", 220):
        print(f"Q4 vs Q1 trough-location (argmin) shift, label-permutation, Bonferroni /{NSPAN}\n")
        print(tbl.to_string(index=False,
              formatters={"mean_hi": "{:+.3f}".format, "mean_lo": "{:+.3f}".format,
                          "diff": "{:+.3f}".format, "p_perm": "{:.4f}".format,
                          "p_perm_bonf": "{:.4f}".format, "p_welch": "{:.4f}".format}))

    # ---- figure: per-model trough shift by group, per span
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    for j, span in enumerate(SPANS):
        xhi, xlo = store[span]
        r = tbl[tbl.span == SPANLAB[span]].iloc[0]
        for X, col, off in [(xhi, "#C44E52", -0.16), (xlo, "#4C72B0", 0.16)]:
            xx = np.full(X.size, j + off) + RNG.normal(0, 0.03, X.size)
            ax.scatter(xx, X, s=26, color=col, alpha=0.55, edgecolor="none", zorder=2)
            m = X.mean(); se = X.std(ddof=1) / np.sqrt(X.size)
            ax.errorbar(j + off, m, yerr=1.96 * se, fmt="o", ms=8, color=col, capsize=4, lw=2, zorder=4)
        sig = "*" if r.p_perm_bonf < 0.05 else "ns"
        ax.text(j, ax.get_ylim()[1], f"p={r.p_perm_bonf:.3f} {sig}", ha="center", va="bottom",
                fontsize=9, color=("#C44E52" if r.p_perm_bonf < 0.05 else "#888"))
    ax.axhline(0, ls=":", c="#999", lw=1)
    ax.set_xticks(range(NSPAN)); ax.set_xticklabels([SPANLAB[s] for s in SPANS])
    ax.set_ylabel("per-model trough-location shift (rel depth)\n4-shot - 0-shot  (<0 = earlier)")
    ax.set_xlabel("span")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0], [0], marker="o", ls="", color="#C44E52", label="Q4 top uptake"),
                       Line2D([0], [0], marker="o", ls="", color="#4C72B0", label="Q1 bottom uptake")],
              loc="lower right", frameon=False)
    ax.set_title("Test 11: few-shot trough-location shift, top vs bottom pragmatic-uptake quartiles\n"
                 f"(41-model panel, label-permutation, Bonferroni /{NSPAN}, "
                 f"n={store['I'][0].size} vs {store['I'][1].size})", fontsize=12, fontweight="bold")
    ax.margins(y=0.15)
    fig.tight_layout()
    out = PLOTS / "test11_group_argmin_quartiles.png"
    fig.savefig(out, dpi=150, facecolor="white", bbox_inches="tight")
    print("\nwrote", out)


if __name__ == "__main__":
    main()
