"""Re-correct the three between-group quartile tests (curvature=Test 9, CKA=Test 10, argmin=Test 11)
with Benjamini-Hochberg FDR across the 4 spans, replacing Bonferroni. Reads the saved raw permutation
p-values (p_perm) from testN_results.csv -- no re-run of the permutations. Writes group_fdr_<name>.csv
with an added p_perm_fdr column and prints a combined table (CKA first, then curvature, then argmin).
Runnable: python group_fdr.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
SPAN_ORDER = ["QUD", "I", "PU", "U"]                    # display order (matches tab:signed)
TESTS = [("CKA (Test 10)", "test10_results.csv"),      # paper order: CKA first
         ("Curvature (Test 9)", "test9_results.csv"),
         ("Trough-loc / argmin (Test 11)", "test11_results.csv")]


def bh_fdr(p):
    """Benjamini-Hochberg adjusted p-values for a vector p (monotone, capped at 1)."""
    p = np.asarray(p, float); m = p.size
    order = np.argsort(p); ranks = np.arange(1, m + 1)
    adj_sorted = p[order] * m / ranks
    adj_sorted = np.minimum.accumulate(adj_sorted[::-1])[::-1]   # enforce monotonic non-decreasing
    adj = np.empty(m); adj[order] = np.minimum(adj_sorted, 1.0)
    return adj


def main():
    combined = []
    for name, fn in TESTS:
        df = pd.read_csv(HERE / fn).set_index("span")
        df = df.loc[SPAN_ORDER]
        df["p_perm_fdr"] = bh_fdr(df["p_perm"].values)
        df.to_csv(HERE / f"group_fdr_{fn}")
        print(f"\n=== {name} : Q4(top) vs Q1(bottom), BH-FDR over 4 spans ===")
        show = df.reset_index()[["span", "diff", "p_perm", "p_perm_fdr"]]
        print(show.to_string(index=False,
              formatters={"diff": "{:+.4f}".format, "p_perm": "{:.4f}".format,
                          "p_perm_fdr": "{:.4f}".format}))
        for sp, row in df.iterrows():
            combined.append(dict(test=name, span=sp, diff=row["diff"],
                                 p_raw=row["p_perm"], p_fdr=row["p_perm_fdr"],
                                 fdr_sig=row["p_perm_fdr"] < 0.05))
    out = pd.DataFrame(combined)
    out.to_csv(HERE / "group_fdr_combined.csv", index=False)
    nsig = int(out.fdr_sig.sum())
    print(f"\nwrote group_fdr_combined.csv  |  spans significant at FDR<0.05: {nsig}/12")


if __name__ == "__main__":
    main()
