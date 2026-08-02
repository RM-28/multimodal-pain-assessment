"""Paths, constants, and the one thing that makes notebooks portable.

The whole point of this module: a notebook cell should never contain a
hardcoded path. Raj runs on Kaggle, Jonathan runs locally, Meng might do
either -- and the same cell has to work for all three. So everything goes
through DATA_ROOT, which figures out where it is at import time.
"""
from __future__ import annotations

import os
from pathlib import Path

# --- where are we running? ---------------------------------------------------
# Kaggle mounts attached datasets read-only under /kaggle/input.
IS_KAGGLE = Path("/kaggle/input").exists()

# Repo root = two levels up from this file (src/painnet/config.py -> repo/)
REPO_ROOT = Path(__file__).resolve().parents[2]


def _find_data_root() -> Path:
    """Locate the 'Multimodal Pain Dataset' folder, wherever it happens to be.

    Kaggle slugs the dataset directory, and whoever unzips locally may or may
    not keep the outer 'PhysioPain Dataset' wrapper -- so rather than guess,
    just go looking for the folder we actually care about.
    """
    # explicit override always wins (useful in CI, or if you moved things)
    if env := os.environ.get("PAINNET_DATA_ROOT"):
        return Path(env)

    search_bases = (
        [Path("/kaggle/input")] if IS_KAGGLE else [REPO_ROOT / "data" / "raw"]
    )
    for base in search_bases:
        if not base.exists():
            continue
        # the folder is named the same in both worlds; find it at any depth
        for candidate in base.rglob("Multimodal Pain Dataset"):
            if candidate.is_dir():
                return candidate

    # Not found. Don't raise at import time -- that would make `import painnet`
    # explode in a notebook before the user has had a chance to download
    # anything. Return the expected local path and let the loaders complain.
    return REPO_ROOT / "data" / "raw" / "PhysioPain Dataset" / "Multimodal Pain Dataset"


DATA_ROOT = _find_data_root()

# --- the tiers we use --------------------------------------------------------
# RAW EEG (1Hz)/All is the clean one: 83 files, 83 unique subjects, one label
# each. The PROCESSED EEG tier looks bigger but duplicates comorbid subjects
# across pain-type folders with byte-identical signals -- see docs/dataset-notes.md.
EEG_RAW_DIR = DATA_ROOT / "RAW EEG DATA (1Hz)" / "All"
WATCH_PROCESSED_DIR = DATA_ROOT / "PROCESSED WATCH DATA"
SURVEY_XLSX = DATA_ROOT / "SURVEY DATA" / "survey_answers_en.xlsx"

# writable output locations (on Kaggle only /kaggle/working is writable)
OUTPUT_ROOT = Path("/kaggle/working") if IS_KAGGLE else REPO_ROOT
PROCESSED_DIR = OUTPUT_ROOT / "data" / "processed"
MODELS_DIR = OUTPUT_ROOT / "models"
RESULTS_DIR = OUTPUT_ROOT / "results"

# --- labels ------------------------------------------------------------------
# Order is fixed so confusion matrices are comparable across runs/people.
PAIN_CLASSES = ["no_pain", "headache", "back_pain", "menstrual_pain"]
CLASS_TO_IDX = {c: i for i, c in enumerate(PAIN_CLASSES)}

# subject counts as measured from the raw tier (see docs/dataset-notes.md)
SUBJECT_COUNTS = {"headache": 30, "back_pain": 28, "no_pain": 15, "menstrual_pain": 10}
N_SUBJECTS = 83
MAJORITY_BASELINE = 30 / 83  # 0.361 -- the number to beat, subject-level

# --- EEG feature columns -----------------------------------------------------
# The NeuroSky MindWave gives 8 band powers at 1 Hz plus two proprietary
# "eSense" scores. That's it -- single channel, no spatial info to work with.
BAND_COLS = ["Delta", "Theta", "Alpha1", "Alpha2", "Beta1", "Beta2", "Gamma1", "Gamma2"]
ESENSE_COLS = ["Attention", "Meditation"]
EEG_FEATURE_COLS = BAND_COLS + ESENSE_COLS

# Columns we deliberately throw away, and why:
#   Derived  -> 100% NA in every file
#   totPwr   -> exactly sum(BAND_COLS); corr 1.0, max abs diff 0. Collinear.
#   class    -> undocumented X/X*/1/1* marker, never '1' for menstrual subjects
#   obs,time -> row index and clock, not features
DROP_COLS = ["Derived", "totPwr", "class", "obs", "time"]

SUBJECT_COL = "id"
LABEL_COL = "pain_type"

# --- windowing defaults ------------------------------------------------------
# 60s window / 15s stride at 1 Hz => 60 timesteps, 75% overlap. Matches the
# reference implementation so our numbers are at least comparable to theirs.
WINDOW_SECONDS = 60
STEP_SECONDS = 15

# --- reproducibility ---------------------------------------------------------
SEED = 42
N_FOLDS = 5


def ensure_dirs() -> None:
    """Make the writable output dirs. Safe to call repeatedly."""
    for d in (PROCESSED_DIR, MODELS_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def describe() -> str:
    """One-liner for the top of a notebook so you know where you are."""
    where = "Kaggle" if IS_KAGGLE else "local"
    ok = "found" if EEG_RAW_DIR.exists() else "MISSING (run scripts/get_data.py)"
    return f"painnet | env={where} | data={ok} | {DATA_ROOT}"
