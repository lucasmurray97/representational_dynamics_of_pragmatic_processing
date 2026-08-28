"""Regenerate scores_all_complete.csv: pragmaticity / resolution / literality / netS per model x shot,
for every model fully extracted (5 shots, both counterbalance orders).

Behaviour is scored from the spantokE metadata with the answer-token pair auto-detected per model:
a response is pragmatic if the model picks the implicature option in the fwd order and the literal
option after the two candidates are swapped (rev order). gemma-4-12B is a {0,1}-token model pulled
from polar2_summary.csv instead (if present). The DeepSeek reasoning family is dropped (its 1/2 answer
is unreadable).
  pragmaticity = prag/N   resolution = (prag+lit)/N   literality = lit/N   netS = (prag-und)/N
  (N = prag+lit+und)
Run: python scoring/regen_scores.py
"""
import sys
from pathlib import Path
import pandas as pd, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root: config
import config

OUT = config.REPR_OUTPUTS
SUMMARY = config.RESPONSES / "polar2_summary.csv"


def token_map(model):
    toks = set()
    for s in range(5):
        p = OUT / model / f"repr_conc_polar_{s}sh_spantokE_fwd" / "metadata.parquet"
        if p.exists():
            toks |= set(pd.read_parquet(p)["answer_text"].unique())
    valid = sorted(t for t in toks if t in {"0", "1", "2"})
    if len(valid) < 2:
        valid = ["0", "1"] if "0" in toks else ["1", "2"]
    return {valid[0]: 0, valid[1]: 1}


def score_spantok(model, shot, vmap):
    fp = OUT / model / f"repr_conc_polar_{shot}sh_spantokE_fwd" / "metadata.parquet"
    rp = OUT / model / f"repr_conc_polar_{shot}sh_spantokE_rev" / "metadata.parquet"
    if not (fp.exists() and rp.exists()):
        return None
    f = pd.read_parquet(fp); r = pd.read_parquet(rp)
    ap = float(pd.concat([f["answer_prob"], r["answer_prob"]]).mean())
    f = f[["prompt_id", "answer_text"]].rename(columns={"answer_text": "a"})
    r = r[["prompt_id", "answer_text"]].rename(columns={"answer_text": "b"})
    m = f.merge(r, on="prompt_id")
    cl = m["a"].isin(vmap) & m["b"].isin(vmap); mm = m[cl]
    a = mm["a"].map(vmap); b = mm["b"].map(vmap)
    prag = int(((a == 0) & (b == 1)).sum()); lit = int(((a == 1) & (b == 0)).sum())
    und = int((a == b).sum()); N = len(mm)
    return prag, lit, und, N, ap, "/".join(sorted(vmap, key=vmap.get)), "spantokE"


def score_polar2(shot):
    if not SUMMARY.exists():
        return None
    d = pd.read_csv(SUMMARY)
    d = d[(d["dir"] == "gemma-4-12B-it") & (d["prompt"] == "conceptual") & (d["shot"] == shot)]
    if not len(d):
        return None
    prag, lit, und = int(d["prag"].iloc[0]), int(d["lit"].iloc[0]), int(d["und"].iloc[0])
    return prag, lit, und, prag + lit + und, np.nan, "polar2", "polar2_summary"


def main():
    rows = []
    for m in sorted({p.name for p in OUT.iterdir() if p.is_dir()}):
        if m.startswith("DeepSeek"):
            continue
        c = sum(all((OUT / m / f"repr_conc_polar_{s}sh_spantokE_{o}" / "span_tokens_meta.json").exists()
                    for o in ("fwd", "rev")) for s in range(5))
        if c < 5:
            continue
        is_g412 = (m == "gemma-4-12B-it")
        vmap = token_map(m)
        for s in range(5):
            r = score_polar2(s) if is_g412 else score_spantok(m, s, vmap)
            if not r:
                continue
            prag, lit, und, N, ap, tok, src = r
            rows.append(dict(model=m, shot=s, source=src, tokens=tok, answer_prob=ap,
                             prag=prag, lit=lit, und=und, N=N,
                             pragmaticity=prag / N if N else np.nan,
                             resolution=(prag + lit) / N if N else np.nan,
                             literality=lit / N if N else np.nan,
                             netS=(prag - und) / N if N else np.nan))
    df = pd.DataFrame(rows)
    config.RESPONSES.mkdir(parents=True, exist_ok=True)
    outp = config.SCORES_CSV
    df.to_csv(outp, index=False)
    print(f"saved {outp}: {df.model.nunique()} models x 5 shots = {len(df)} rows (DeepSeek dropped)")
    lp = df.groupby("model")["answer_prob"].mean()
    low = lp[lp < 0.1]
    print("low answer_prob (<0.1, potential unreadable):", list(low.index) or "none")


if __name__ == "__main__":
    main()
