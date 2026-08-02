"""Turning per-second rows into fixed-length windows a Conv1D can eat.

Two things happen here and the order matters:
  1. per-subject z-scoring  (removes each person's baseline offset)
  2. sliding-window segmentation

Per-subject normalisation is safe to do before splitting *because* the stats
are computed within a single subject and GroupKFold keeps subjects whole -- no
training-set statistic ever touches a test subject. It IS transductive though:
normalising a test subject needs that subject's whole recording. Fine for an
offline benchmark, worth a sentence in the report.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def zscore_per_subject(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Centre and scale each subject's channels using only that subject's data.

    The band powers span ~4 orders of magnitude (Delta max 4.0e6, Gamma2 min 24)
    and the per-person baseline varies enormously with electrode contact. Without
    this the network mostly learns "how well was the headset seated".
    """
    out = df.copy()
    g = out.groupby(config.SUBJECT_COL)[feature_cols]
    mu, sd = g.transform("mean"), g.transform("std")
    out[feature_cols] = (out[feature_cols] - mu) / sd.replace(0, np.nan)
    # a flat channel (sd == 0) becomes NaN above -> it carries no information,
    # so zero is the honest fill
    out[feature_cols] = out[feature_cols].fillna(0.0)
    return out


def make_windows(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    window: int = config.WINDOW_SECONDS,
    step: int = config.STEP_SECONDS,
):
    """Slide a window over each subject independently.

    Returns (X, y, groups):
        X       (n_windows, window, n_features) float32
        y       (n_windows,) str   -- the subject's pain_type
        groups  (n_windows,) str   -- subject id, for GroupKFold

    Windows never straddle a subject boundary. Labels are constant within a
    subject so there's no majority-vote needed -- every window inherits the one
    label its subject has.
    """
    feature_cols = feature_cols or config.EEG_FEATURE_COLS
    X, y, groups = [], [], []

    for sid, g in df.groupby(config.SUBJECT_COL, sort=True):
        arr = g[feature_cols].to_numpy(dtype=np.float32)
        label = g[config.LABEL_COL].iloc[0]
        n = len(arr)
        if n < window:
            continue  # recording shorter than one window -- skip (none are, but still)
        for start in range(0, n - window + 1, step):
            X.append(arr[start:start + window])
            y.append(label)
            groups.append(sid)

    if not X:
        raise ValueError("No windows produced -- check window/step vs recording length.")

    return np.stack(X), np.asarray(y), np.asarray(groups)


def build_dataset(window: int = config.WINDOW_SECONDS, step: int = config.STEP_SECONDS):
    """The one-liner: raw files -> normalised windows ready for Keras.

    >>> X, y, groups = build_dataset()
    >>> X.shape
    (6465, 60, 10)
    """
    from .data import load_raw_eeg  # local import keeps module load cheap

    df = load_raw_eeg()
    df = zscore_per_subject(df, config.EEG_FEATURE_COLS)
    return make_windows(df, config.EEG_FEATURE_COLS, window, step)


def encode_labels(y) -> np.ndarray:
    """pain_type strings -> ints, using the fixed order in config.PAIN_CLASSES."""
    return np.asarray([config.CLASS_TO_IDX[v] for v in y], dtype=np.int32)


def to_tabular(X) -> np.ndarray:
    """Flatten (n, timesteps, channels) -> (n, channels*4) summary stats.

    LightGBM can't eat a 3D tensor, so each window collapses to mean/std/min/max
    per channel. Crude, but at 1 Hz over 60s that's most of what's there, and it
    makes the tree baseline a fair comparison rather than a straw man -- if it
    matches the conv nets, that tells us the sequence structure isn't buying much.
    """
    return np.concatenate(
        [X.mean(1), X.std(1), X.min(1), X.max(1)], axis=1
    ).astype(np.float32)


def class_weights(y_int) -> dict[int, float]:
    """Median-frequency weights, to stop the model just predicting 'headache'.

    menstrual_pain is 12% of subjects vs headache's 36%, so unweighted training
    happily ignores it.
    """
    vals, counts = np.unique(y_int, return_counts=True)
    median = float(np.median(counts))
    return {int(v): median / float(c) for v, c in zip(vals, counts)}
