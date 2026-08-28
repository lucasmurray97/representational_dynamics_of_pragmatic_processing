"""Central path configuration for the release.

Every script imports paths from here instead of hardcoding them. All locations can be overridden
with environment variables so the pipeline runs anywhere. Add the repo root to sys.path (the scripts
do this automatically) and ``import config``.
"""
from __future__ import annotations
import os
from pathlib import Path

# Repo root = directory containing this file.
ROOT = Path(__file__).resolve().parent

# --- shipped data (small; in the repo) ------------------------------------------------------------
DATA_ROOT = Path(os.environ.get("DATA_ROOT", ROOT / "data"))
PROMPTS = DATA_ROOT / "prompts"
STIMULI = DATA_ROOT / "stimuli"
RESPONSES = DATA_ROOT / "responses"
SCORES_CSV = RESPONSES / "scores_all_complete.csv"
MODELS_CSV = ROOT / "models.csv"

# --- extraction outputs (large per-token activations; regenerated locally, not shipped) -----------
REPR_OUTPUTS = Path(os.environ.get("REPR_OUTPUTS", ROOT / "outputs" / "repr_outputs"))

# --- derived artifacts (curvature parquet, surrogate null, CKA npz; regenerated locally) ----------
ARTIFACTS = Path(os.environ.get("ARTIFACTS", ROOT / "artifacts"))
CURV_DIR = ARTIFACTS / "curvature"        # r1_curv_<alias>_<shot>sh_<order>.parquet
SURR_DIR = ARTIFACTS / "surrogates"       # surrogate_<alias>_<shot>sh_<order>.parquet
CKA_DIR = ARTIFACTS / "cka_matrices"      # <alias>_s<shot>.npz


def ensure_dirs() -> None:
    """Create the writable output directories if missing (producers call this)."""
    for d in (REPR_OUTPUTS, CURV_DIR, SURR_DIR, CKA_DIR):
        d.mkdir(parents=True, exist_ok=True)
