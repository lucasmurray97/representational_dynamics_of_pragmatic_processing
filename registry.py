"""Model registry: single source of truth for the roster, sizes, HF checkpoints, and panel flags.

Reads ``models.csv``. Replaces the per-script ``SIZE``/exclusion dicts. The reported panel (v7) is
the full 44-model roster minus the three pragmaticity ``decreaser`` models, giving 41 models.
"""
from __future__ import annotations
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
_ROWS = []
with open(ROOT / "models.csv", newline="") as fh:
    for r in csv.DictReader(fh):
        r["params_b"] = int(r["params_b"])
        r["in_panel"] = r["in_panel"].strip().lower() == "true"
        _ROWS.append(r)

ALIASES = [r["alias"] for r in _ROWS]
SIZE = {r["alias"]: r["params_b"] for r in _ROWS}
HF_ID = {r["alias"]: r["hf_checkpoint"] for r in _ROWS}
FLAG = {r["alias"]: r["flag"] for r in _ROWS}

DECREASERS = {r["alias"] for r in _ROWS if r["flag"] == "decreaser"}
NONRESPONDERS = {r["alias"] for r in _ROWS if r["flag"] == "nonresponder"}
DISENGAGERS = {r["alias"] for r in _ROWS if r["flag"] == "disengager"}

# The reported panels (v7) exclude only the pragmaticity-decreasers -> 41 models.
PANEL_EXCLUDE = set(DECREASERS)


def panel_models():
    """The 41 aliases of the reported (v7) panels: full roster minus the decreasers."""
    return [a for a in ALIASES if a not in PANEL_EXCLUDE]
