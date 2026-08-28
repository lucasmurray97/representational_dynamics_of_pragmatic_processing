"""Test 12: does the IMPLICATURE-vs-CONTROL curvature-change contrast predict pragmatic uptake?
Single size-controlled regression on the 41-model panel:

    dprag_i = b0 + b1 * X_i + b2 * log(size) + error_i,   H0: b1 = 0
    X_i = mean_depth|c4_I - c0_I|  -  mean_depth|c4_PU - c0_PU|

X_i is the per-model excess curvature change in the implicature span \Inf{} over the literal/control
span \PU{} -- a within-model contrast that nets out each model's global curvature reactivity, isolating
the I-specific component. One pre-specified hypothesis, so the raw two-sided regression p is reported
(no multiple-comparison correction). Reports beta, p, incremental R^2 beyond size, and n.
Runnable: python test12_I_minus_PU.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, str(Path(__file__).resolve().parent))
import data as D

HERE = Path(__file__).resolve().parent


def curv_change_span(a, span):
    c0, c4 = D.curv_prof(a, 0, span), D.curv_prof(a, 4, span)
    if c0 is None or c4 is None:
        return None
    return float(np.mean(np.abs(c4 - c0)))


def main():
    prag, _ = D._scores()
    panel = [a for a in D.SIZE if (a, 0) in prag and (a, 4) in prag and a not in D.PANEL_EXCLUDE]
    rows = []
    for a in panel:
        cI, cPU = curv_change_span(a, "I"), curv_change_span(a, "PU")
        if cI is None or cPU is None:
            continue
        rows.append((a, cI - cPU, prag[(a, 4)] - prag[(a, 0)], float(np.log(D.SIZE[a]))))
    x = np.array([r[1] for r in rows]); y = np.array([r[2] for r in rows]); ls = np.array([r[3] for r in rows])
    n = len(rows)

    X = np.column_stack([np.ones(n), x, ls])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yh = X @ beta; rss = float(((y - yh) ** 2).sum()); dfres = n - 3
    se = np.sqrt(rss / dfres * np.diag(np.linalg.inv(X.T @ X)))
    tval = beta / se; pval = 2 * stats.t.sf(np.abs(tval), dfres)
    b_x, p_x = beta[1], pval[1]
    tss = float(((y - y.mean()) ** 2).sum()); r2 = 1 - rss / tss
    Xs = np.column_stack([np.ones(n), ls]); bs, *_ = np.linalg.lstsq(Xs, y, rcond=None)
    r2_size = 1 - float(((y - Xs @ bs) ** 2).sum()) / tss
    incr = r2 - r2_size

    # size-partialled correlation as a robustness read
    def resid(v):
        Z = np.column_stack([np.ones_like(ls), ls]); b, *_ = np.linalg.lstsq(Z, v, rcond=None)
        return v - Z @ b
    rp = float(np.corrcoef(resid(x), resid(y))[0, 1])
    rho = float(stats.spearmanr(resid(x), resid(y)).statistic)

    pd.DataFrame([dict(n=n, beta=b_x, p=p_x, incrR2=incr, partial_r=rp, spearman_partial=rho,
                       size_beta=beta[2], size_p=pval[2])]).to_csv(HERE / "test12_results.csv", index=False)
    print(f"dprag ~ (|Dc_I| - |Dc_PU|) + logsize   n={n}\n")
    print(f"  beta_contrast = {b_x:+.4f} / deg")
    print(f"  p             = {p_x:.4f}   (raw, two-sided; single pre-specified test)")
    print(f"  incremental R2= {incr:.4f}")
    print(f"  partial r     = {rp:+.3f}   spearman-partial = {rho:+.3f}")
    print(f"  [size term: beta={beta[2]:+.4f}, p={pval[2]:.3f}]")
    print(f"  mean X = {x.mean():+.3f} deg  (I minus PU curvature change; >0 = I changes more)")


if __name__ == "__main__":
    main()
