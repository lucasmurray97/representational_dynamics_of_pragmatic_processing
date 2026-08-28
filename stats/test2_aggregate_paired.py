"""Test 2: aggregate-per-model paired test of the 4-shot vs 0-shot curvature change, per span.

Each model's paired difference trajectory D_i(d) = (curv4-surro4) - (curv0-surro0) is collapsed to a
single scalar S_i, then ONE paired test per span across models (4 tests total, Bonferroni /4). No
depth grid, no correlated multiplicity -- at the cost of a pre-specified summary statistic.

Summary statistics (S_i):
  PRIMARY  mid     mean of D_i over the mid-depth window [0.30, 0.60]. Directional; the window is
                   pre-specified from the PAPER's straightening location (curvature minimum ~0.45-0.48
                   mid-depth), NOT tuned to our Test-1 peaks -> no double-dipping. Does not cancel.
  robust   full    mean of D_i over the whole depth (assumption-free but cancels sign-varying spans).
  robust   dipmin  min over depth of D_i (deepest straightening; negatively biased under H0, but the
                   permutation null aggregates identically so the test stays valid).

Test per (span, summary):
  obs = mean_i S_i;  two-sided sign-flip permutation null (NPERM perms, aggregation done inside the
  loop) -> p_perm;  plus Wilcoxon signed-rank -> p_wilcox. Effect size dz = mean(S)/sd(S).
  Significant if p < 0.05/4 = 0.0125 (Bonferroni over the 4 spans).
n = 44 models (all with 0- & 4-shot curvature + surrogate). Script pairs with test1 (per-depth).
Runnable: python test2_aggregate_paired.py
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
SPANS = ["I", "PU", "U", "QUD_polar"]; SPANLAB = {"I": "I", "PU": "PU", "U": "U", "QUD_polar": "QUD"}
GRID = D.GRID
MIDWIN = (0.30, 0.60)                 # pre-specified from the paper's mid-depth straightening
MIDMASK = (GRID >= MIDWIN[0]) & (GRID <= MIDWIN[1])
NPERM = 200000
NSPAN = len(SPANS)
BONF = 0.05 / NSPAN
RNG = np.random.default_rng(0)


def paired_matrix(span):
    rows, names = [], []
    for a in D.SIZE:
        if a in D.PANEL_EXCLUDE:                 # 41-model reported panel (v7)
            continue
        c0, c4 = D.curv_prof(a, 0, span), D.curv_prof(a, 4, span)
        s0, s4 = D.surro_prof(a, 0, span), D.surro_prof(a, 4, span)
        if any(x is None for x in (c0, c4, s0, s4)):
            continue
        rows.append((c4 - s4) - (c0 - s0)); names.append(a)
    return np.array(rows), names


SUMMARIES = {
    "mid":    lambda Dm: Dm[:, MIDMASK].mean(1),
    "full":   lambda Dm: Dm.mean(1),
    "dipmin": lambda Dm: Dm.min(1),
}


def signflip_p(S):
    """Two-sided sign-flip permutation p for mean(S) == 0."""
    obs = S.mean(); n = S.size
    ge = 0
    for _ in range(NPERM):
        s = RNG.integers(0, 2, size=n) * 2 - 1
        if abs((s * S).mean()) >= abs(obs) - 1e-15:
            ge += 1
    return (1 + ge) / (1 + NPERM), obs


def main():
    rows = []
    prim = {}
    for span in SPANS:
        Dm, names = paired_matrix(span)
        for key, fn in SUMMARIES.items():
            S = fn(Dm)
            p_perm, obs = signflip_p(S)
            # Wilcoxon signed-rank (two-sided); guard the all-zero edge case
            try:
                w = stats.wilcoxon(S, alternative="two-sided", zero_method="wilcox")
                p_wil = float(w.pvalue)
            except ValueError:
                p_wil = np.nan
            dz = obs / S.std(ddof=1) if S.std(ddof=1) > 0 else np.nan
            frac_neg = float((S < 0).mean())
            p_t = float(stats.ttest_1samp(S, 0).pvalue)      # parametric cross-check
            rows.append(dict(span=SPANLAB[span], summary=key, n=len(names),
                             mean_S=obs, dz=dz, frac_neg=frac_neg, skew=float(stats.skew(S)),
                             p_perm=p_perm, p_wilcox=p_wil, p_ttest=p_t,
                             sig_bonf=p_perm < BONF))
            if key == "mid":
                prim[span] = S
    tbl = pd.DataFrame(rows)
    tbl.to_csv(HERE / "test2_results.csv", index=False)

    order = {"mid": 0, "full": 1, "dipmin": 2}
    tbl_sorted = tbl.sort_values(["summary", "span"], key=lambda c: c.map(order) if c.name == "summary" else c)
    with pd.option_context("display.width", 200):
        print(f"Bonferroni threshold (4 spans): p < {BONF:.4f}\n")
        print(tbl_sorted.to_string(index=False,
              formatters={"mean_S": "{:+.3f}".format, "dz": "{:+.2f}".format,
                          "frac_neg": "{:.2f}".format, "p_perm": "{:.2e}".format,
                          "p_wilcox": "{:.2e}".format}))

    # ---- figure: per-model PRIMARY summary (mid-depth window mean) per span, strip + mean +/- CI
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for j, span in enumerate(SPANS):
        S = prim[span]
        x = np.full(S.size, j) + RNG.normal(0, 0.05, S.size)
        ax.scatter(x, S, s=22, color="#4C72B0", alpha=0.5, edgecolor="none", zorder=2)
        m = S.mean(); se = S.std(ddof=1) / np.sqrt(S.size)
        r = tbl[(tbl.span == SPANLAB[span]) & (tbl.summary == "mid")].iloc[0]
        col = "#C44E52" if r.sig_bonf else "#888"
        ax.errorbar(j, m, yerr=1.96 * se, fmt="o", ms=9, color=col, capsize=5, lw=2, zorder=4)
        star = "*" if r.sig_bonf else "ns"
        ax.text(j, ax.get_ylim()[1], f"p={r.p_perm:.1e} {star}", ha="center", va="bottom",
                fontsize=9, color=col)
    ax.axhline(0, ls=":", c="#999", lw=1)
    ax.set_xticks(range(NSPAN)); ax.set_xticklabels([SPANLAB[s] for s in SPANS])
    ax.set_ylabel(f"per-model Δcurv, mid-depth [{MIDWIN[0]:.2f},{MIDWIN[1]:.2f}] mean (deg)\n(4-shot - 0-shot)")
    ax.set_xlabel("span")
    ax.set_title("Test 2: aggregate paired 4-shot vs 0-shot curvature change per span\n"
                 f"(mid-depth window mean, sign-flip permutation, Bonferroni /{NSPAN}, n={prim['I'].size})",
                 fontsize=12, fontweight="bold")
    ax.margins(y=0.15)
    fig.tight_layout()
    out = PLOTS / "test2_aggregate_paired.png"
    fig.savefig(out, dpi=150, facecolor="white", bbox_inches="tight")
    print("\nwrote", out)


if __name__ == "__main__":
    main()
