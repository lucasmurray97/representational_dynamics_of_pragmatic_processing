"""Consolidated per-model table for the statistical tests. One row per model with the behavioural
outcome, the representational predictors, size, and model-set flags. Every test imports `master()`
so the numbers are defined in exactly one place.

Sources (all path-configured in ../config.py, model roster in ../models.csv):
  scores_all_complete.csv (behaviour), curvature parquet r1_curv_*_{0,4}sh_fwd.parquet,
  surrogate parquet surrogate_*_{0,4}sh_fwd.parquet, CKA matrices <alias>_s{shot}.npz.
Run to dump the table:  python stats/data.py
"""
from __future__ import annotations
import sys
from pathlib import Path
from functools import lru_cache
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root: config, registry
import config
import registry

HERE = Path(__file__).resolve().parent
SPANS = ["QUD_polar", "U", "I", "PU"]
GRID = np.linspace(0, 1, 40)
POOL = "bertscore"

# roster + panel definitions come from models.csv via registry
SIZE = registry.SIZE
DECREASERS = registry.DECREASERS
NONRESPONDERS = registry.NONRESPONDERS
DISENGAGERS = registry.DISENGAGERS
PANEL_EXCLUDE = registry.PANEL_EXCLUDE
panel_models = registry.panel_models


@lru_cache(maxsize=None)
def _scores():
    sc = pd.read_csv(config.SCORES_CSV)
    prag = {(r.model, int(r.shot)): r.pragmaticity for _, r in sc.iterrows()}
    res = {(r.model, int(r.shot)): r.resolution for _, r in sc.iterrows()}
    return prag, res


def curv_prof(a, shot, span):
    """Measured per-span curvature profile (deg) interpolated onto GRID, or None."""
    fp = config.CURV_DIR / f"r1_curv_{a}_{shot}sh_fwd.parquet"
    if not fp.exists():
        return None
    r = pd.read_parquet(fp); L = r.layer.max() + 1
    p = r[r.span == span].groupby("layer").curv_deg.mean()
    if p.empty or p.isna().any():
        return None
    return np.interp(GRID, p.index.to_numpy() / (L - 1), p.to_numpy())


def surro_prof(a, shot, span):
    """Surrogate null profile (deg) on GRID, or None."""
    fp = config.SURR_DIR / f"surrogate_{a}_{shot}sh_fwd.parquet"
    if not fp.exists():
        return None
    s = pd.read_parquet(fp); s = s[s.span == span]
    if s.empty or s.curv_surrogate.isna().any():
        return None
    return np.interp(GRID, s.rel_depth.to_numpy(), s.curv_surrogate.to_numpy())


def curv_change(a):
    """mean over spans of mean-depth |curv4 - curv0| (deg). Baseline-free (a 0<->4 difference)."""
    vals = []
    for span in SPANS:
        c0, c4 = curv_prof(a, 0, span), curv_prof(a, 4, span)
        if c0 is None or c4 is None:
            return np.nan
        vals.append(np.mean(np.abs(c4 - c0)))
    return float(np.mean(vals))


def curv_dip(a, shot):
    """Span-averaged minimum of (curv - surrogate null) over depth (deg, <=0 = straightening)."""
    dips = []
    for span in SPANS:
        c = curv_prof(a, shot, span); s = surro_prof(a, shot, span)
        if c is None or s is None:
            return np.nan
        dips.append(float((c - s).min()))
    return float(np.mean(dips))


def surro_mean(a):
    vals = []
    for shot in (0, 4):
        for span in SPANS:
            s = surro_prof(a, shot, span)
            if s is not None:
                vals.append(float(s.mean()))
    return float(np.mean(vals)) if vals else np.nan


def _load_cka(a, shot):
    fp = config.CKA_DIR / f"{a}_s{shot}.npz"
    return np.load(fp, allow_pickle=True)["data"].item() if fp.exists() else None


def cka_change(a):
    """mean over spans of off-diagonal mean |CKA4 - CKA0| (within-bracket)."""
    m0, m4 = _load_cka(a, 0), _load_cka(a, 4)
    if m0 is None or m4 is None or POOL not in m0:
        return np.nan
    vals = []
    for s in SPANS:
        D = np.abs(m4[POOL]["within"][s] - m0[POOL]["within"][s])
        if np.isnan(D).any():
            return np.nan
        vals.append(float(D[~np.eye(D.shape[0], dtype=bool)].mean()))
    return float(np.mean(vals))


@lru_cache(maxsize=1)
def master():
    """One row per model with behaviour + predictors + flags. Cached."""
    prag, res = _scores()
    rows = []
    for a in SIZE:
        if (a, 0) not in prag or (a, 4) not in prag:
            continue
        p0, p4 = prag[(a, 0)], prag[(a, 4)]
        r0, r4 = res.get((a, 0), np.nan), res.get((a, 4), np.nan)
        rows.append(dict(
            model=a, size=SIZE[a], logsize=float(np.log(SIZE[a])),
            p0=p0, p4=p4, dprag=p4 - p0,
            r0=r0, r4=r4, dres=r4 - r0,
            curv_change=curv_change(a),
            curv_dip0=curv_dip(a, 0), curv_dip4=curv_dip(a, 4),
            ddip=curv_dip(a, 4) - curv_dip(a, 0),
            cka_change=cka_change(a),
            surro_mean=surro_mean(a),
            flag_decreaser=a in DECREASERS,
            flag_nonresponder=a in NONRESPONDERS,
            flag_disengager=a in DISENGAGERS,
        ))
    df = pd.DataFrame(rows).sort_values(["size", "model"]).reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = master()
    out = HERE / "master_table.csv"
    df.to_csv(out, index=False)
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(df.round(4).to_string(index=False))
    print(f"\n{len(df)} models -> {out}")
    print("curvature available:", df.curv_change.notna().sum(),
          "| cka available:", df.cka_change.notna().sum(),
          "| surrogate available:", df.surro_mean.notna().sum())
