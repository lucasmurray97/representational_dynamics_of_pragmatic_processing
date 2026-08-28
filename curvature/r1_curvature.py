"""R1 position-curvature profile for one model, from extracted span tokens.

POSITION curvature = angle between successive token difference vectors within a span, at a fixed
layer, averaged over stimuli and positions, as a function of relative depth. Reference lines at
90 deg (random-walk null) and 120 deg (iid null); below 120 means straightening. Unit = stimulus.

Writes ``r1_curv_<alias>_<shot>sh_<order>.parquet`` (columns: prompt_id, span, layer, curv_deg,
rel_depth) to config.CURV_DIR, plus a per-model diagnostic plot.
Run: python curvature/r1_curvature.py ALIAS SHOT ORDER [SETTAG]
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))          # local: metrics
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))      # repo root: config
import metrics as M
import config

HERE = Path(__file__).resolve().parent
SPANS = ["QUD_polar", "U", "I", "PU", "last5"]
RNG = np.random.default_rng(0)


def load(alias, shot, order, settag="spantokE"):
    d = config.REPR_OUTPUTS / alias / f"repr_conc_polar_{shot}sh_{settag}_{order}"
    meta = json.load(open(d / "span_tokens_meta.json"))
    mm = np.memmap(d / "span_tokens.bin", dtype=np.float16 if meta["dtype"] == "fp16" else np.float32,
                   mode="r", shape=(meta["rows"], meta["n_layers"], meta["d"]))
    idx = pd.read_parquet(d / "span_tokens_index.parquet")
    return mm, idx, meta["n_layers"]


def per_stimulus_curvature(mm, idx, L):
    """Return long df: one row per (prompt_id, span, layer) with mean position curvature (deg)."""
    rows = []
    for (pid, span), g in idx.groupby(["prompt_id", "span"], sort=False):
        if span not in SPANS:
            continue
        r = g.sort_values("pos_in_span")["row"].to_numpy()
        if r.size < 3:                       # need >= 3 tokens for >= 1 pointwise curvature
            continue
        X = np.asarray(mm[r], dtype=np.float32)      # [n_tok, L, d]
        for l in range(L):
            c = M.pointwise_curvature(X[:, l, :])
            rows.append((int(pid), span, l, float(np.degrees(np.nanmean(c))) if c.size else np.nan))
    return pd.DataFrame(rows, columns=["prompt_id", "span", "layer", "curv_deg"])


def main():
    alias = sys.argv[1] if len(sys.argv) > 1 else "Qwen2.5-7B-Instruct"
    shot = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    order = sys.argv[3] if len(sys.argv) > 3 else "fwd"
    settag = sys.argv[4] if len(sys.argv) > 4 else "spantokE"
    tag = f"{alias}_{shot}sh_{order}"
    config.CURV_DIR.mkdir(parents=True, exist_ok=True)
    mm, idx, L = load(alias, shot, order, settag)
    print(f"# {alias} shot={shot} {order}: {mm.shape[0]} tokens, {L} layers, spans "
          f"{dict(idx.span.value_counts())}")

    df = per_stimulus_curvature(mm, idx, L)
    df["rel_depth"] = df["layer"] / (L - 1)
    df.to_parquet(config.CURV_DIR / f"r1_curv_{tag}.parquet")

    print(f"\n{'span':10s} {'n_stim':>7} {'min_deg':>8} {'argmin_reldepth':>16} {'in[0.3,0.7]':>12} "
          f"{'end_deg(L0/L1)':>16}")
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    tab = plt.get_cmap("tab10").colors
    for k, span in enumerate(SPANS):
        s = df[df.span == span]
        if not len(s):
            continue
        prof = s.groupby("layer")["curv_deg"].mean()
        rel = prof.index.to_numpy() / (L - 1)
        vals = prof.to_numpy()
        amin_layer = int(np.nanargmin(vals)); amin_rel = amin_layer / (L - 1)
        pivot = s.pivot_table(index="prompt_id", columns="layer", values="curv_deg")
        boots = []
        pv = pivot.to_numpy(); nS = pv.shape[0]
        for _ in range(500):
            bs = pv[RNG.integers(0, nS, nS)]
            m = np.nanmean(bs, axis=0)
            boots.append(np.nanargmin(m) / (L - 1))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        ok = 0.3 <= amin_rel <= 0.7 and lo > 0.3 and hi < 0.7
        print(f"{span:10s} {pivot.shape[0]:>7d} {np.nanmin(vals):>8.1f} "
              f"{amin_rel:>10.2f} [{lo:.2f},{hi:.2f}] {str(ok):>12} "
              f"{vals[0]:>7.1f}/{vals[-1]:>7.1f}")
        ax.plot(rel, vals, marker="o", ms=3, color=tab[k % 10], label=f"{span}")
    ax.axhline(120, ls="--", c="#c44", lw=1, label="120 deg (iid null)")
    ax.axhline(90, ls="--", c="#888", lw=1, label="90 deg (random walk)")
    ax.set_xlabel("relative depth"); ax.set_ylabel("mean position curvature (deg)")
    ax.set_title(f"{alias} [{order}] R1 position curvature by depth (span tokens)")
    ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)
    pdir = config.CURV_DIR / "plots"; pdir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(pdir / f"r1_position_curvature_{tag}.png", dpi=150, facecolor="white")
    print(f"\nwrote {config.CURV_DIR}/r1_curv_{tag}.parquet + plots/r1_position_curvature_{tag}.png")


if __name__ == "__main__":
    main()
