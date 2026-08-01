"""The leakage guard. Run this constantly; it takes about a second.

If any of these fail, every model number in the project is meaningless. That's
not hyperbole -- with per-subject constant labels, a subject appearing on both
sides of a split lets the model score by recognising the person.

    pytest -q                 # from the repo root
"""
from __future__ import annotations

import numpy as np
import pytest

from painnet import config, splits


# --- synthetic data, so this runs with or without the dataset downloaded ------
def fake_windows(n_subjects=20, per_subject=8, n_classes=4, seed=0):
    """Subjects with ONE label each -- mimics the real PhysioPain structure."""
    rng = np.random.default_rng(seed)
    subject_ids = [f"S{i:03d}" for i in range(n_subjects)]
    subject_label = {s: i % n_classes for i, s in enumerate(subject_ids)}
    groups = np.repeat(subject_ids, per_subject)
    y = np.array([subject_label[s] for s in groups])
    return y, groups


def test_guard_catches_deliberate_leak():
    """The whole point: a subject on both sides must raise."""
    with pytest.raises(splits.LeakageError):
        splits.assert_no_subject_leakage(["S001", "S002"], ["S002", "S003"])


def test_guard_passes_when_disjoint():
    splits.assert_no_subject_leakage(["S001", "S002"], ["S003", "S004"])


def test_subject_folds_never_leak():
    y, groups = fake_windows()
    n = 0
    for train_idx, test_idx in splits.subject_folds(groups, y, n_splits=5):
        tr, te = set(groups[train_idx]), set(groups[test_idx])
        assert not (tr & te), "subject in both train and test"
        n += 1
    assert n == 5


def test_every_window_used_exactly_once_as_test():
    """Folds should partition the data -- no window tested twice or never."""
    y, groups = fake_windows()
    seen = np.zeros(len(y), dtype=int)
    for _, test_idx in splits.subject_folds(groups, y, n_splits=5):
        seen[test_idx] += 1
    assert (seen == 1).all()


def test_stratified_spreads_the_rare_class():
    """The 10-subject menstrual class must not all land in one fold.

    Simulates the real imbalance: 30 / 28 / 15 / 10 subjects.
    """
    counts = [30, 28, 15, 10]
    groups, y = [], []
    sid = 0
    for cls, n in enumerate(counts):
        for _ in range(n):
            groups += [f"S{sid:03d}"] * 8
            y += [cls] * 8
            sid += 1
    groups, y = np.array(groups), np.array(y)

    rare_per_fold = []
    for _, test_idx in splits.subject_folds(groups, y, n_splits=5, stratified=True):
        rare_per_fold.append(len(set(groups[test_idx][y[test_idx] == 3])))

    # every fold should see at least one of the rare class, or macro-F1 for
    # that fold is undefined
    assert min(rare_per_fold) >= 1, f"a fold has no rare-class subjects: {rare_per_fold}"


def test_class_order_is_pinned():
    """Confusion matrices are only comparable if class order never moves."""
    assert config.PAIN_CLASSES == ["no_pain", "headache", "back_pain", "menstrual_pain"]
    assert config.CLASS_TO_IDX["no_pain"] == 0


# --- these only run if the dataset is actually present ------------------------
needs_data = pytest.mark.skipif(
    not config.EEG_RAW_DIR.exists(),
    reason="dataset not downloaded (run scripts/get_data.py)",
)


@needs_data
def test_real_labels_are_constant_per_subject():
    """The claim the whole design rests on. Verify it against the real files."""
    from painnet import data

    df = data.load_raw_eeg()
    varying = df.groupby(config.SUBJECT_COL)[config.LABEL_COL].nunique()
    assert (varying == 1).all(), (
        f"{(varying > 1).sum()} subject(s) have more than one pain_type -- "
        "the per-subject-constant assumption is broken, revisit the design"
    )
    assert varying.size == config.N_SUBJECTS


@needs_data
def test_real_windows_split_cleanly():
    from painnet import windows

    X, y, groups = windows.build_dataset()
    assert X.ndim == 3 and X.shape[1] == config.WINDOW_SECONDS
    y_int = windows.encode_labels(y)
    for train_idx, test_idx in splits.subject_folds(groups, y_int):
        assert not (set(groups[train_idx]) & set(groups[test_idx]))
