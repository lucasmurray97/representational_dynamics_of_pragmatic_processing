"""Gaussian anisotropy-matched surrogate NULL for the position-curvature analysis, for ALL models
and shots, computed directly from the stored spantokE activations (no GPU / no re-extraction).

Per (model, shot, order) and per layer we fit the per-PC variance spectrum of the real activations
from a token subsample, then draw iid N(0, diag(lambda)) surrogate trajectories of MATCHED length
(one per real span trajectory >= 3 tokens) and take their mean pointwise curvature. This is the
correct null for the ~120 deg baseline under anisotropy (curvature is rotation invariant). The
measured curvature itself is already stored in r1_curv_*.parquet, so we only produce the surrogate
here. Output: surrogates/surrogate_{alias}_{shot}sh_{order}.parquet [span, layer, rel_depth,
curv_surrogate, n_traj]. CPU only. Runnable (background):
    python batch_surrogate.py [ORDER] [SETTAG] [MAXSUB]
    python batch_surrogate.py ONE ALIAS SHOT ORDER   # single (validation)
"""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits          # runtime cap (env vars don't cap openblas)
sys.path.insert(0, str(Path(__file__).resolve().parent))          # local: metrics
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))      # repo root: config
import metrics as M
import config

OUT = config.REPR_OUTPUTS
HERE = Path(__file__).resolve().parent
SURR = config.SURR_DIR; SURR.mkdir(parents=True, exist_ok=True)
SPANS = ["QUD_polar", "U", "I", "PU", "last5"]
MAXSUB = 4000
NTHREADS = 8


def surrogate_one(alias, shot, order, settag="spantokE", seed=0):
    outp = SURR / f"surrogate_{alias}_{shot}sh_{order}.parquet"
    if outp.exists():                                    # resumable: skip finished jobs
        print(f"{alias:32s} {shot}sh {order}: exists, skip", flush=True)
        return outp
    d = OUT / alias / f"repr_conc_polar_{shot}sh_{settag}_{order}"
    if not (d / "span_tokens_meta.json").exists():
        return None
    meta = json.load(open(d / "span_tokens_meta.json"))
    L, D, R = meta["n_layers"], meta["d"], meta["rows"]
    mm = np.memmap(d / "span_tokens.bin",
                   dtype=np.float16 if meta["dtype"] == "fp16" else np.float32,
                   mode="r", shape=(R, L, D))
    idx = pd.read_parquet(d / "span_tokens_index.parquet")
    # span trajectory lengths (>= 3 tokens), grouped per span
    lens = {s: [] for s in SPANS}
    for (pid, span), g in idx.groupby(["prompt_id", "span"], sort=False):
        if span in SPANS and len(g) >= 3:
            lens[span].append(len(g))

    rng = np.random.default_rng(seed)
    # contiguous row block read ONCE (sequential I/O; avoids a random-seek storm over NFS). A
    # contiguous block is a fine covariance-spectrum estimate for a null. Held as [nsub, L, D].
    nsub = min(MAXSUB, R)
    sub = np.asarray(mm[:nsub], dtype=np.float32)             # [nsub, L, D], one sequential read

    rows = []
    with threadpool_limits(limits=NTHREADS):
        for l in range(L):
            A = sub[:, l, :]
            sc = A - A.mean(0, keepdims=True)
            sv = np.linalg.svd(sc, compute_uv=False)
            std_pc = np.sqrt(np.maximum(sv ** 2 / (nsub - 1), 0.0))   # per-PC std, len <= D
            npc = std_pc.size
            for span in SPANS:
                vals = []
                for n in lens[span]:
                    Z = rng.standard_normal((n, npc)) * std_pc
                    c = M.pointwise_curvature(Z)
                    if c.size:
                        vals.append(float(np.degrees(np.nanmean(c))))
                rows.append({"span": span, "layer": l, "rel_depth": l / (L - 1),
                             "curv_surrogate": float(np.nanmean(vals)) if vals else np.nan,
                             "n_traj": len(lens[span])})
    df = pd.DataFrame(rows)
    outp = SURR / f"surrogate_{alias}_{shot}sh_{order}.parquet"
    df.to_parquet(outp)
    # quick summary over content spans
    cs = df[df.span.isin(["QUD_polar", "U", "I", "PU"])]
    print(f"{alias:32s} {shot}sh {order}: L={L} surro mean={cs.curv_surrogate.mean():.2f} "
          f"min={cs.curv_surrogate.min():.2f} max={cs.curv_surrogate.max():.2f} -> {outp.name}",
          flush=True)
    return outp


def main():
    if sys.argv[1:2] == ["ONE"]:
        _, _, alias, shot, order = sys.argv[:5]
        surrogate_one(alias, int(shot), order)
        return
    order = sys.argv[1] if len(sys.argv) > 1 else "fwd"
    settag = sys.argv[2] if len(sys.argv) > 2 else "spantokE"
    nworkers = int(sys.argv[3]) if len(sys.argv) > 3 else 1   # partition jobs across processes
    worker = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    # discover all (alias, shot) with a spantokE_{order} dir
    jobs = []
    for al in sorted(OUT.glob("*")):
        for dd in al.glob(f"repr_conc_polar_*sh_{settag}_{order}"):
            name = dd.name  # repr_conc_polar_{shot}sh_{settag}_{order}
            if "smoke" in name or "test_" in name:
                continue
            try:
                shot = int(name.split("repr_conc_polar_")[1].split("sh_")[0])
            except (IndexError, ValueError):
                continue
            jobs.append((al.name, shot))
    jobs = sorted(set(jobs))
    shot_filter = os.environ.get("SURR_SHOTS")            # e.g. "0,4" to restrict shots
    if shot_filter:
        keep = {int(x) for x in shot_filter.split(",")}
        jobs = [(a, s) for (a, s) in jobs if s in keep]

    # smallest-model-first so quick results stream in early (cost ~ layers * d). Read the cheap meta.
    def dim(alias, shot):
        mp = OUT / alias / f"repr_conc_polar_{shot}sh_{settag}_{order}" / "span_tokens_meta.json"
        try:
            m = json.load(open(mp)); return m["n_layers"] * m["d"]
        except Exception:
            return 1 << 30
    jobs.sort(key=lambda j: dim(*j))
    # partition by round-robin across workers (each worker gets an interleaved mix of sizes)
    jobs = [j for i, j in enumerate(jobs) if i % nworkers == worker]
    print(f"# worker {worker}/{nworkers}: {len(jobs)} (model, shot) jobs, order={order}, settag={settag}", flush=True)
    done = 0
    for alias, shot in jobs:
        try:
            if surrogate_one(alias, shot, order, settag) is not None:
                done += 1
        except Exception as e:
            print(f"!! {alias} {shot}sh {order}: {type(e).__name__}: {e}", flush=True)
    print(f"# done {done}/{len(jobs)}", flush=True)


if __name__ == "__main__":
    main()
