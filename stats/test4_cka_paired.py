"""Test 4: aggregate-per-model paired test of the 4-shot vs 0-shot within-bracket CKA change, per
span -- the CKA analogue of Test 2 (curvature).

Per model, per span, the signed CKA difference matrix DeltaCKA = CKA4 - CKA0 (L x L, same L within a
model) is collapsed to one scalar by averaging its OFF-DIAGONAL entries (the diagonal is the trivial
layer-with-itself similarity). Then ONE paired test per span across models:
  obs = mean_i S_i;  two-sided sign-flip permutation (NPERM) -> p_perm;  Wilcoxon signed-rank cross
  check;  effect size dz = mean(S)/sd(S);  frac_neg = share of models with S_i < 0 (CKA dropped).
  Significant if p < 0.05/4 = 0.0125 (Bonferroni over the 4 spans).
Signed (like curvature): tests whether few-shot systematically raises or lowers within-bracket CKA in
each span. n = 44 models (all with 0- & 4-shot CKA). Pool = bertscore.
Runnable: python test4_cka_paired.py
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
SPANS = ["QUD_polar", "I", "PU", "U"]; SPANLAB = {"QUD_polar": "QUD", "I": "I", "PU": "PU", "U": "U"}
POOL = D.POOL
NPERM = 200000
NSPAN = len(SPANS)
BONF = 0.05 / NSPAN
RNG = np.random.default_rng(0)


def offdiag_mean_delta(a, span):
    """Mean off-diagonal signed (CKA4 - CKA0) for model a, span. None if missing/NaN."""
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


def paired_scalars(span):
    S, names = [], []
    for a in D.SIZE:
        if a in D.PANEL_EXCLUDE:                 # 41-model reported panel (v7)
            continue
        v = offdiag_mean_delta(a, span)
        if v is not None:
            S.append(v); names.append(a)
    return np.array(S), names


def signflip_p(S):
    obs = S.mean(); n = S.size
    ge = 0
    for _ in range(NPERM):
        s = RNG.integers(0, 2, size=n) * 2 - 1
        if abs((s * S).mean()) >= abs(obs) - 1e-15:
            ge += 1
    return (1 + ge) / (1 + NPERM), obs


def main():
    rows, store = [], {}
    for span in SPANS:
        S, names = paired_scalars(span)
        p_perm, obs = signflip_p(S)
        try:
            p_wil = float(stats.wilcoxon(S, alternative="two-sided", zero_method="wilcox").pvalue)
        except ValueError:
            p_wil = np.nan
        dz = obs / S.std(ddof=1) if S.std(ddof=1) > 0 else np.nan
        p_t = float(stats.ttest_1samp(S, 0).pvalue)
        rows.append(dict(span=SPANLAB[span], n=len(names), mean_dCKA=obs, dz=dz,
                         frac_neg=float((S < 0).mean()), p_perm=p_perm, p_wilcox=p_wil,
                         p_ttest=p_t, bonf_sig=p_perm < BONF))
        store[span] = S
    tbl = pd.DataFrame(rows)
    tbl.to_csv(HERE / "test4_results.csv", index=False)
    with pd.option_context("display.width", 200):
        print(f"CKA paired 4-shot vs 0-shot (signed, off-diagonal mean). Bonferroni /{NSPAN}: p<{BONF}\n")
        print(tbl.to_string(index=False,
              formatters={"mean_dCKA": "{:+.4f}".format, "dz": "{:+.2f}".format,
                          "frac_neg": "{:.2f}".format, "p_perm": "{:.2e}".format,
                          "p_wilcox": "{:.2e}".format}))

    # ---- figure
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for j, span in enumerate(SPANS):
        S = store[span]
        x = np.full(S.size, j) + RNG.normal(0, 0.05, S.size)
        ax.scatter(x, S, s=22, color="#55A868", alpha=0.5, edgecolor="none", zorder=2)
        m = S.mean(); se = S.std(ddof=1) / np.sqrt(S.size)
        r = tbl[tbl.span == SPANLAB[span]].iloc[0]
        col = "#C44E52" if r.bonf_sig else "#888"
        ax.errorbar(j, m, yerr=1.96 * se, fmt="o", ms=9, color=col, capsize=5, lw=2, zorder=4)
        star = "*" if r.bonf_sig else "ns"
        ax.text(j, ax.get_ylim()[1], f"p={r.p_perm:.1e} {star}", ha="center", va="bottom",
                fontsize=9, color=col)
    ax.axhline(0, ls=":", c="#999", lw=1)
    ax.set_xticks(range(NSPAN)); ax.set_xticklabels([SPANLAB[s] for s in SPANS])
    ax.set_ylabel("per-model mean off-diagonal ΔCKA (4-shot - 0-shot)")
    ax.set_xlabel("span")
    ax.set_title("Test 4: aggregate paired 4-shot vs 0-shot within-bracket CKA change per span\n"
                 f"(off-diagonal mean, signed, sign-flip permutation, Bonferroni /{NSPAN}, n={store['I'].size})",
                 fontsize=12, fontweight="bold")
    ax.margins(y=0.15)
    fig.tight_layout()
    out = PLOTS / "test4_cka_paired.png"
    fig.savefig(out, dpi=150, facecolor="white", bbox_inches="tight")
    print("\nwrote", out)


if __name__ == "__main__":
    main()
