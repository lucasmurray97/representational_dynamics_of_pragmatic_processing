"""Test 10: between-group comparison of few-shot within-bracket CKA change, top vs bottom pragmatic-
uptake quartiles, per span, Bonferroni over the 4 spans. CKA analogue of Test 9 (curvature).

On the 41-model reported panel (v7) we rank models by pragmatic uptake
Dprag = pragmaticity(4-shot) - pragmaticity(0-shot) and split the EXTREME quartiles:
  Q4 = top 25% (strongest uptake)   vs   Q1 = bottom 25% (weakest / negative uptake);
the middle 50% is dropped (~11 per group). Per model the representational quantity is the signed
within-bracket CKA change, off-diagonal mean of the L x L difference matrix:
  DCKA_i = offdiag_mean( CKA4 - CKA0 )      (>0 = more cross-layer similarity at 4-shot)
Per span the statistic is the difference in group means  obs = mean(DCKA|Q4) - mean(DCKA|Q1). The null
is INDEPENDENT-GROUPS: shuffle the Q4/Q1 labels across the pooled models (NPERM), recompute; two-sided
p = (1 + #|perm| >= |obs|) / (1 + NPERM). p is Bonferroni-adjusted across the 4 spans
(min(1, 4*p); significant at 0.05). Welch t as a parametric cross-check. CKA only.
Runnable: python test10_group_cka_quartiles.py
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


def cka_delta(a, span):
    """Signed off-diagonal mean of (CKA4 - CKA0) for model a, span. None if missing/NaN."""
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
    Dm = C4 - C0
    off = ~np.eye(Dm.shape[0], dtype=bool)
    return float(Dm[off].mean())


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
        xhi = np.array([v for v in (cka_delta(a, span) for a in HIGH) if v is not None])
        xlo = np.array([v for v in (cka_delta(a, span) for a in LOW) if v is not None])
        obs, p_perm = perm_group_diff(xhi, xlo)
        t, p_t = stats.ttest_ind(xhi, xlo, equal_var=False)
        rows.append(dict(span=SPANLAB[span], n_hi=xhi.size, n_lo=xlo.size,
                         mean_hi=xhi.mean(), mean_lo=xlo.mean(), diff=obs,
                         p_perm=p_perm, p_perm_bonf=min(1, NSPAN * p_perm),
                         p_welch=float(p_t)))
        store[span] = (xhi, xlo)
    tbl = pd.DataFrame(rows)
    tbl.to_csv(HERE / "test10_results.csv", index=False)
    with pd.option_context("display.width", 220):
        print(f"Q4 vs Q1 within-bracket CKA change (off-diagonal mean), label-permutation, Bonferroni /{NSPAN}\n")
        print(tbl.to_string(index=False,
              formatters={"mean_hi": "{:+.4f}".format, "mean_lo": "{:+.4f}".format,
                          "diff": "{:+.4f}".format, "p_perm": "{:.4f}".format,
                          "p_perm_bonf": "{:.4f}".format, "p_welch": "{:.4f}".format}))

    # ---- figure: per-model CKA change by group, per span
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
    ax.set_ylabel("per-model within-bracket CKA change\noff-diagonal mean, 4-shot - 0-shot")
    ax.set_xlabel("span")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0], [0], marker="o", ls="", color="#C44E52", label="Q4 top uptake"),
                       Line2D([0], [0], marker="o", ls="", color="#4C72B0", label="Q1 bottom uptake")],
              loc="lower right", frameon=False)
    ax.set_title("Test 10: few-shot CKA change, top vs bottom pragmatic-uptake quartiles\n"
                 f"(41-model panel, label-permutation, Bonferroni /{NSPAN}, "
                 f"n={store['I'][0].size} vs {store['I'][1].size})", fontsize=12, fontweight="bold")
    ax.margins(y=0.15)
    fig.tight_layout()
    out = PLOTS / "test10_group_cka_quartiles.png"
    fig.savefig(out, dpi=150, facecolor="white", bbox_inches="tight")
    print("\nwrote", out)


if __name__ == "__main__":
    main()
