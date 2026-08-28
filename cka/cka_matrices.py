"""Within-bracket and cross-pair cross-layer (L x L) linear-CKA matrices for one (model, shot) cell.

For each pooling (last / mean / bertscore) and each content bracket b, builds CKA(b@i, b@j) over all
layer pairs (i, j) -- the "within-bracket" matrix -- plus the cross-bracket-pair matrices. The 'last'
pooling reads the precomputed pooled_bracket_*_last.safetensors; 'mean'/'bertscore' are derived from
span_tokens.bin in a single row-major pass. Cached to config.CKA_DIR/<alias>_s<shot>.npz under key
"data" as {pool: {"within": {bracket: LxL}, "pair": {(a,b): LxL}}}.
Run: python cka/cka_matrices.py ALIAS SHOT
"""
from __future__ import annotations
import sys, json, math
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))       # local: cka_helpers
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root: config
import cka_helpers as H
import config

OUT = H.OUT
BR = H.BRACKET_ORDER
PAIRS = H.ORD_PAIRS
POOLS = ["last", "mean", "bertscore"]
MATS = config.CKA_DIR


def _last_stacks(alias, shot):
    """{bracket: [L,N,d]} from the precomputed *_last pooled brackets, or None."""
    for d in (OUT / alias / f"repr_conc_polar_{shot}sh_spantokE_fwd",
              OUT / alias / f"repr_conc_polar_{shot}sh_fwd"):
        if (d / "pooled_bracket_U_last.safetensors").exists():
            return {b: H._load_cka_set(d, b, "_last") for b in BR}
    return None


def _derive_fast(spe, pools):
    """{pool: {bracket: [L,N,d]}} for pools subset of {mean, bertscore}, from span_tokens.bin.
    Row-major: the file is [rows, layers, d], so mm[rows] reads each token's full [L,d] contiguously.
    bertscore IDF = log((N+1)/(df+1)), df = #stimuli whose bracket span contains the token."""
    meta = json.load(open(spe / "span_tokens_meta.json"))
    L, D, Rn = meta["n_layers"], meta["d"], meta["rows"]
    mm = np.memmap(spe / "span_tokens.bin", dtype=np.float16 if meta["dtype"] == "fp16" else np.float32,
                   mode="r", shape=(Rn, L, D))
    idx = pd.read_parquet(spe / "span_tokens_index.parquet")
    Nitems = idx.prompt_id.nunique()
    idf, groups, pset = {}, {b: {} for b in BR}, {}
    for b in BR:
        sub = idx[idx.span == b].sort_values(["prompt_id", "pos_in_span"])
        dfc = sub.groupby("token_id").prompt_id.nunique()
        idf[b] = {int(t): math.log((Nitems + 1) / (c + 1)) for t, c in dfc.items()}
        for pid, g in sub.groupby("prompt_id", sort=False):
            groups[b][pid] = (g.row.to_numpy(), g.token_id.to_numpy())
        pset[b] = set(groups[b])
    common = sorted(set.intersection(*pset.values()))
    N = len(common)
    out = {p: {b: np.zeros((L, N, D), np.float32) for b in BR} for p in pools}
    for b in BR:
        for i, pid in enumerate(common):
            rows, tids = groups[b][pid]
            V = np.asarray(mm[rows], dtype=np.float32)          # [k, L, d] (contiguous per row)
            if "mean" in pools:
                out["mean"][b][:, i, :] = V.mean(0)
            if "bertscore" in pools:
                w = np.array([idf[b][int(t)] for t in tids], np.float32)
                ws = w.sum()
                # IDF degeneracy (df=N -> IDF=0 for every token): fall back to the unweighted mean.
                out["bertscore"][b][:, i, :] = (V.mean(0) if ws == 0
                                                else (w[:, None, None] * V).sum(0) / ws)
    return out


def _lxl_matrices(brk):
    """within/pair L x L CKA from {bracket:[L,N,d]}: flatten each layer's centered Gram to a row and
    CKA[i,j] = <F_a[i], F_b[j]> / (||F_a[i]|| ||F_b[j]||) is one matmul, not an L^2 python loop."""
    F, nrm = {}, {}
    for b in BR:
        K = H._layer_kernels(brk[b])
        F[b] = np.stack([k.ravel() for k in K]).astype(np.float32)
        nrm[b] = np.linalg.norm(F[b], axis=1)
    lxl = lambda a, c: (F[a] @ F[c].T) / np.maximum(np.outer(nrm[a], nrm[c]), 1e-12)
    return {"within": {b: lxl(b, b) for b in BR}, "pair": {p: lxl(p[0], p[1]) for p in PAIRS}}


def cell_matrices(alias, shot):
    """{pool: {'within': {b: LxL}, 'pair': {(a,b): LxL}}} for a cell; cache to npz. None if no source.
    Merges pools into any existing cache, so adding mean/bertscore reuses the already-cached 'last'."""
    MATS.mkdir(parents=True, exist_ok=True)
    cache = MATS / f"{alias}_s{shot}.npz"
    out = np.load(cache, allow_pickle=True)["data"].item() if cache.exists() else {}
    if all(p in out for p in POOLS):
        return out
    need = [p for p in POOLS if p not in out]
    if "last" in need and (bl := _last_stacks(alias, shot)) is not None:
        out["last"] = _lxl_matrices(bl); need.remove("last")
    derive = [p for p in need if p in ("mean", "bertscore")]
    if derive:
        spe = OUT / alias / f"repr_conc_polar_{shot}sh_spantokE_fwd"
        if (spe / "span_tokens_meta.json").exists():
            for pool, brk in _derive_fast(spe, derive).items():   # single file pass for both
                out[pool] = _lxl_matrices(brk)
    if not out:
        return None
    np.savez_compressed(cache, data=np.array(out, dtype=object))
    print(f"  cached {cache}  poolings={list(out)}", flush=True)
    return out


if __name__ == "__main__":
    alias = sys.argv[1] if len(sys.argv) > 1 else "Qwen2.5-7B-Instruct"
    shot = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    m = cell_matrices(alias, shot)
    if m is None:
        raise SystemExit(f"no source activations for {alias} s{shot} under {OUT}")
    print(f"{alias} s{shot}: poolings={list(m)}, within-brackets={list(m['bertscore']['within'])}")
