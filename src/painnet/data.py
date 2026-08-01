"""Loading PhysioPain into tidy frames, with the known warts handled.

Warts this module deals with so you don't have to (all verified against the
actual archive -- see docs/dataset-notes.md):
  * CSV headers carry leading spaces: ' time', ' Delta', ..., ' class '
  * 'Derived' is 100% NA in every file
  * 'totPwr' is exactly sum(band columns) -- collinear, drop it
  * pain intensity is NOT in the EEG files at all; it lives in the survey xlsx
  * the processed watch CSVs have the Empatica sample-rate row leaked in as row 0
"""
from __future__ import annotations

import pandas as pd

from . import config


def _read_eeg_csv(path) -> pd.DataFrame:
    """One subject's raw 1 Hz EEG file, columns de-spaced."""
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]  # ' class ' -> 'class'
    return df


def load_raw_eeg(drop_junk: bool = True) -> pd.DataFrame:
    """All 83 subjects of the raw 1 Hz EEG tier, concatenated.

    Returns ~101k rows. Columns: id, pain_type, the 8 band powers, plus
    Attention/Meditation. One row per second of recording.
    """
    d = config.EEG_RAW_DIR
    if not d.exists():
        raise FileNotFoundError(
            f"EEG data not found at {d}\n"
            "Local: run `python scripts/get_data.py`\n"
            "Kaggle: attach the 'orvile/physiopain-dataset' dataset to the notebook."
        )

    files = sorted(d.glob("S*.csv"))
    if not files:
        raise FileNotFoundError(f"No S*.csv files under {d}")

    df = pd.concat([_read_eeg_csv(f) for f in files], ignore_index=True)

    if drop_junk:
        df = df.drop(columns=[c for c in config.DROP_COLS if c in df.columns])

    # Attention/Meditation are ~0.3% NA -- forward-fill within subject rather
    # than dropping whole rows (we'd lose ~300 otherwise for no good reason).
    for col in config.ESENSE_COLS:
        if col in df.columns:
            df[col] = df.groupby(config.SUBJECT_COL)[col].ffill().bfill()

    return df


def load_survey() -> pd.DataFrame:
    """The participant survey (99 rows x 108 cols), indexed by subject id.

    This is the only place pain INTENSITY lives -- the EEG files don't carry it.
    Also has Age and Gender, which we need for the confound check.
    """
    if not config.SURVEY_XLSX.exists():
        raise FileNotFoundError(f"Survey not found at {config.SURVEY_XLSX}")
    sv = pd.read_excel(config.SURVEY_XLSX)
    return sv.set_index("id")


# The severity question we treat as the intensity label. There are five of
# these "Rate the severity..." columns; this is the most direct one.
SEVERITY_COL = (
    "Rate the severity of your pain (Likert scale) [How severe is your pain now?]"
)
SEVERITY_ORDER = ["Not severe at all", "Mild", "Moderate", "Severe", "Very severe"]


def subject_table() -> pd.DataFrame:
    """One row per subject: label, n_rows, and the survey fields we care about.

    This is the frame to reason about, because with per-subject constant labels
    the subject IS the unit of analysis. 83 rows, not 101k.
    """
    eeg = load_raw_eeg(drop_junk=False)
    per = (
        eeg.groupby(config.SUBJECT_COL)
        .agg(pain_type=(config.LABEL_COL, "first"), n_seconds=(config.LABEL_COL, "size"))
    )

    sv = load_survey()
    keep = {
        "Age": "age",
        "Gender (Biological)": "gender",
        "What is your pain type?": "self_reported_type",
        SEVERITY_COL: "severity",
    }
    per = per.join(sv[list(keep)].rename(columns=keep), how="left")

    # ordered categorical so sorting/plotting behaves
    per["severity"] = pd.Categorical(
        per["severity"], categories=SEVERITY_ORDER, ordered=True
    )
    per["severity_ord"] = per["severity"].cat.codes.replace(-1, pd.NA)

    # flag the 2 subjects whose EEG label says menstrual_pain but who
    # self-reported "Abdominal pain" -- decide per-analysis whether to drop them
    per["label_disagrees"] = (
        (per["pain_type"] == "menstrual_pain")
        & (per["self_reported_type"] == "Abdominal pain")
    )
    return per


def load_watch(pain_type: str, tier_hz: int = 4) -> pd.DataFrame:
    """Processed Empatica wristband signals for one pain-type folder.

    NOTE: drops row 0 of every file. The Empatica E4 puts the sample rate on
    row 2 of its raw CSVs and the dataset authors' script only stripped row 1,
    so the rate values leaked in as the first data row (e.g. eda=4.0,
    temperature=4.0 in a 4 Hz file). Verified; see docs/dataset-notes.md.
    """
    folder = config.WATCH_PROCESSED_DIR / pain_type / f"signal_{tier_hz}"
    if not folder.exists():
        raise FileNotFoundError(
            f"{folder} not found. The watch tier is ~4.9 GB -- "
            "run `python scripts/get_data.py --watch` to extract it."
        )

    frames = []
    for f in sorted(folder.glob("*.csv")):
        d = pd.read_csv(f)
        d.columns = [c.strip() for c in d.columns]
        d = d.iloc[1:].reset_index(drop=True)  # <- the leaked rate row
        frames.append(d)
    return pd.concat(frames, ignore_index=True)
