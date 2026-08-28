"""Test 3: does the curvature straightening trough occur at an EARLIER (shallower) relative depth at
4-shot than at 0-shot? Paired, per span, on the [0,1] normalized relative-depth axis.

Per model i, per span, from y(d) = curv(d) - surro(d) on the 40-pt normalized grid GRID in [0,1]:
  loc_argmin(y)   = GRID[argmin(y)]                         (colleague's proposal; noisy on flat troughs)
  loc_centroid(y) = sum_d w(d) d / sum_d w(d), w=max(0,-y)  (robust "center of straightening mass")
Statistic per model = Delta = loc(y4) - loc(y0). Directional hypothesis: earlier => Delta < 0.

Test per (span, estimator): one-sided paired sign-flip permutation (NPERM, lower tail) + one-sided
Wilcoxon signed-rank (alternative='less'). Effect: mean Delta, frac earlier (<0). PRIMARY spans
I and QUD (where a trough exists); Bonferroni over the 2 primary spans (p<0.025). U, PU computed but
exploratory (trough ill-defined -- they do not straighten, Tests 1-2). n = 44.
Runnable: python test3_min_depth_shift.py
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
SPANS = ["I", "QUD_polar", "U", "PU"]; SPANLAB = {"I": "I", "QUD_polar": "QUD", "U": "U", "PU": "PU"}
PRIMARY = {"I", "QUD_polar"}
NPERM = 50000
BONF = 0.05 / 2                     # 2 primary spans
RNG = np.random.default_rng(0)


def loc_argmin(y):
    return GRID[int(np.argmin(y))]


def loc_centroid(y):
    w = np.maximum(0.0, -y)
    return float(np.sum(w * GRID) / np.sum(w)) if w.sum() > 0 else np.nan


def deltas(span, locfn):
    out, names = [], []
    for a in D.SIZE:
        c0, c4 = D.curv_prof(a, 0, span), D.curv_prof(a, 4, span)
        s0, s4 = D.surro_prof(a, 0, span), D.surro_prof(a, 4, span)
        if any(x is None for x in (c0, c4, s0, s4)):
            continue
        l0, l4 = locfn(c0 - s0), locfn(c4 - s4)
        if np.isnan(l0) or np.isnan(l4):
            continue
        out.append(l4 - l0); names.append(a)
    return np.array(out), names


def signflip_p_less(delta):
    """One-sided (lower-tail) sign-flip permutation p for mean(delta) < 0."""
    obs = delta.mean(); n = delta.size
    le = 0
    for _ in range(NPERM):
        s = RNG.integers(0, 2, size=n) * 2 - 1
        if (s * delta).mean() <= obs + 1e-15:
            le += 1
    return (1 + le) / (1 + NPERM), obs


def main():
    rows = []
    store = {}
    for span in SPANS:
        for est, fn in [("argmin", loc_argmin), ("centroid", loc_centroid)]:
            dlt, names = deltas(span, fn)
            p_perm, obs = signflip_p_less(dlt)
            try:
                p_wil = float(stats.wilcoxon(dlt, alternative="less", zero_method="wilcox").pvalue)
            except ValueError:
                p_wil = np.nan
            med = float(np.median(dlt)); frac_e = float((dlt < 0).mean())
            dz = obs / dlt.std(ddof=1) if dlt.std(ddof=1) > 0 else np.nan
            rows.append(dict(span=SPANLAB[span], estimator=est, primary=span in PRIMARY,
                             n=len(names), mean_delta=obs, median_delta=med, dz=dz,
                             frac_earlier=frac_e, p_perm_1s=p_perm, p_wilcox_1s=p_wil,
                             bonf_sig=(span in PRIMARY and p_perm < BONF)))
            store[(span, est)] = dlt
    tbl = pd.DataFrame(rows)
    tbl.to_csv(HERE / "test3_results.csv", index=False)
    with pd.option_context("display.width", 220):
        print(f"one-sided (earlier => Delta<0); Bonferroni over 2 primary spans: p<{BONF}\n")
        print(tbl.to_string(index=False,
              formatters={"mean_delta": "{:+.3f}".format, "median_delta": "{:+.3f}".format,
                          "dz": "{:+.2f}".format, "frac_earlier": "{:.2f}".format,
                          "p_perm_1s": "{:.3f}".format, "p_wilcox_1s": "{:.3f}".format}))

    # ---- figure: per-model Delta for the centroid estimator, per span
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for j, span in enumerate(SPANS):
        dlt = store[(span, "centroid")]
        x = np.full(dlt.size, j) + RNG.normal(0, 0.05, dlt.size)
        ax.scatter(x, dlt, s=22, color="#4C72B0", alpha=0.5, edgecolor="none", zorder=2)
        m = dlt.mean(); se = dlt.std(ddof=1) / np.sqrt(dlt.size)
        r = tbl[(tbl.span == SPANLAB[span]) & (tbl.estimator == "centroid")].iloc[0]
        col = "#C44E52" if r.bonf_sig else ("#4C72B0" if r.primary else "#999")
        ax.errorbar(j, m, yerr=1.96 * se, fmt="o", ms=9, color=col, capsize=5, lw=2, zorder=4)
        tag = "*" if r.bonf_sig else ("ns" if r.primary else "expl")
        ax.text(j, ax.get_ylim()[1], f"p={r.p_perm_1s:.2f} {tag}", ha="center", va="bottom",
                fontsize=9, color=col)
    ax.axhline(0, ls=":", c="#999", lw=1)
    ax.set_xticks(range(len(SPANS))); ax.set_xticklabels([SPANLAB[s] for s in SPANS])
    ax.set_ylabel("Δ trough location (rel depth): 4-shot - 0-shot\n(<0 = earlier at 4-shot)")
    ax.set_xlabel("span   (I, QUD primary; U, PU exploratory)")
    ax.set_title("Test 3: does the straightening trough move earlier under few-shot?\n"
                 f"(centroid location on [0,1] axis, one-sided paired sign-flip perm, n={store[('I','centroid')].size})",
                 fontsize=12, fontweight="bold")
    ax.margins(y=0.15)
    fig.tight_layout()
    out = PLOTS / "test3_min_depth_shift.png"
    fig.savefig(out, dpi=150, facecolor="white", bbox_inches="tight")
    print("\nwrote", out)


if __name__ == "__main__":
    main()
