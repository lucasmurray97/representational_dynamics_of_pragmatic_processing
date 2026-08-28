# Few-shot pragmatic uptake and representational change in LLMs

Code and stimuli for the paper studying whether few-shot prompting changes how instruction-tuned
language models represent an indirect exchange, and whether that change tracks their behavioural
pragmatic uptake. Representations are analysed two ways across model depth: **position curvature** of
the per-token trajectory within each content span, and **within-bracket linear CKA** between layers.
The task is a Spanish conversational-implicature judgement over four content elements — the question
under discussion (**QUD**), the indirect response (**U**), the implicature reading (**I**), and the
literal reading (**PU**).

This repository is anonymized for double-blind review.

## What's here

```
core/          shared primitives: stimulus filling, activation I/O, curvature/CKA math
extraction/    extract.py — hidden-state + response extraction from any HF causal LM
curvature/     r1_curvature.py (position curvature) + batch_surrogate.py (Gaussian null)
cka/           cka_matrices.py — within-bracket cross-layer linear-CKA matrices
scoring/       regen_scores.py — pragmaticity / resolution from model responses
stats/         data.py (per-model table) + test1..test12 + group_fdr.py (all reported tests)
data/          prompts/ (0–4 shot templates), stimuli/ (648 items), responses/ (scores)
models.csv     the 44-model roster: alias, HF checkpoint, size, panel flag
config.py      all input/output paths (override via environment variables)
```

## Install

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
# For gated checkpoints (e.g. Llama), set HF_TOKEN (copy .env.example -> .env).
```
Python 3.11. The extraction step needs a CUDA GPU; everything downstream is CPU-only.

## Reproducibility boundary

Shipped: all code, the stimuli, the prompt templates, and the **behavioural responses/scores**
(`data/responses/`). **Not** shipped: the multi-GB per-token activations and the derived curvature /
surrogate / CKA artifacts — these are regenerated locally (they require downloading the model
checkpoints from Hugging Face and a GPU). Paths for all regenerated data are configured in
`config.py` and default under `outputs/` and `artifacts/`.

## Pipeline

Per model (GPU for step 1), driven end-to-end by the helper script:

```bash
scripts/run_pipeline.sh <alias>        # alias = a row in models.csv; default shots 0 and 4
```

which runs, for each shot:

1. **Extraction** — `extraction/extract.py` fills `data/prompts/repr_conc_polar_<shot>shot.txt`
   against `data/stimuli/items_648.xlsx`, counterbalancing the two answer options
   (`--counterbalance I PU`), and saves per-token span activations + per-item responses to
   `REPR_OUTPUTS/<alias>/repr_conc_polar_<shot>sh_spantokE_{fwd,rev}/`.
2. **Curvature** — `curvature/r1_curvature.py` → `artifacts/curvature/r1_curv_<alias>_<shot>sh_fwd.parquet`.
3. **Surrogate null** — `curvature/batch_surrogate.py` → `artifacts/surrogates/…parquet` (anisotropy-
   matched Gaussian null; ≈120° by construction, the curvature baseline).
4. **CKA** — `cka/cka_matrices.py` → `artifacts/cka_matrices/<alias>_s<shot>.npz`.

Then, once every model is processed:

```bash
python scoring/regen_scores.py        # -> data/responses/scores_all_complete.csv
python stats/data.py                  # -> stats/master_table.csv (one row per model)
python stats/test6_regression_perspan.py   # per-span curvature regression (and test1..test12)
python stats/group_fdr.py             # FDR correction across the group tests
```

The reported panels use the 41-model set (the full roster minus the three pragmaticity-`decreaser`
models flagged in `models.csv`); `registry.panel_models()` returns it.

## Configuration

Override any path with an environment variable (see `config.py`): `REPR_OUTPUTS` (activation root),
`ARTIFACTS` (derived data), `DATA_ROOT`. Model checkpoints resolve through `models.csv`; an alias not
listed there is treated as a raw Hugging Face repo id.

## License

Code released under the MIT License (see `LICENSE`). Stimuli: see `DATASHEET.md`.
