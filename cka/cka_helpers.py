"""Pure linear-CKA helpers: per-layer centered-Gram kernels and pooled-bracket loaders.

Extracted from the analysis pipeline with no model tables and no plotting. Linear CKA on each of
three poolings (mean / last / bertscore) of the four content brackets (QUD/U/I/PU).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from safetensors.numpy import load_file
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root: core, config
from core.metrics import kernel, linear_cka
import config

OUT = config.REPR_OUTPUTS
BRACKET_ORDER = ["QUD_polar", "U", "I", "PU"]
DISPLAY = {"QUD_polar": "QUD"}
ORD_PAIRS = [(BRACKET_ORDER[n], BRACKET_ORDER[m]) for n in range(4) for m in range(n + 1, 4)]
POOL_SUFFIX = {"mean": "", "last": "_last", "bertscore": "_bertscore"}
disp = lambda b: DISPLAY.get(b, b)


def _load_cka_set(run: Path, field: str, suffix: str):
    """Load a GPU-extracted pooled_bracket_<field><suffix>.safetensors as [L, N, d], or None."""
    fp = run / f"pooled_bracket_{field}{suffix}.safetensors"
    if not fp.exists():
        return None
    d = load_file(str(fp))
    order = sorted(d, key=lambda k: int(k.split("_")[1]))
    return np.stack([d[k].astype(np.float32) for k in order], 0)


def _layer_kernels(stk):
    """Per-layer centered Gram kernels for a [L, N, d] stack."""
    return [kernel(stk[l]) for l in range(stk.shape[0])]


def _cka_diag(Ka, Kb):
    """Same-layer CKA vector."""
    return np.array([linear_cka(Ka[l], Kb[l]) for l in range(len(Ka))])


def _cka_LxL(Ka, Kb):
    """Cross-layer L x L CKA matrix."""
    return np.array([[linear_cka(Ka[i], Kb[j]) for j in range(len(Kb))] for i in range(len(Ka))])
