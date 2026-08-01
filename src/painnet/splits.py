"""Subject-wise cross-validation, and the assertions that keep us honest.

This is the most important file in the repo.

Every participant in PhysioPain has exactly ONE pain_type for their entire
recording (verified: 0 of 83 subjects vary). So a window's label is determined
entirely by WHICH PERSON it came from. Split randomly at the window level and
the model just memorises people -- you'd see 95%+ and it would mean nothing.

Effective sample size is 83, not ~6,500 windows. Group on subject id, always.
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

from . import config


class LeakageError(AssertionError):
    """Raised when a split would put one subject on both sides. Do not catch."""


def assert_no_subject_leakage(groups_train, groups_test) -> None:
    """The guard. Call it on every fold, every time. It costs microseconds.

    Raises LeakageError if any subject appears in both train and test.
    """
    tr, te = set(np.asarray(groups_train).ravel()), set(np.asarray(groups_test).ravel())
    overlap = tr & te
    if overlap:
        raise LeakageError(
            f"{len(overlap)} subject(s) in BOTH train and test: "
            f"{sorted(overlap)[:10]}{'...' if len(overlap) > 10 else ''}. "
            "Refusing to continue -- any score from this split is meaningless."
        )


def subject_folds(groups, y=None, n_splits: int = config.N_FOLDS, stratified: bool = True):
    """Yield (train_idx, test_idx) with subjects held out whole.

    stratified=True uses StratifiedGroupKFold so the 10 menstrual_pain subjects
    get spread across folds rather than all landing in one. With classes this
    small that matters a lot -- plain GroupKFold can hand you a test fold with
    zero menstrual subjects, and then macro-F1 is undefined for that class.
    """
    groups = np.asarray(groups)

    if stratified:
        if y is None:
            raise ValueError("stratified=True needs y (the per-window labels)")
        splitter = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=config.SEED
        )
        split_iter = splitter.split(np.zeros(len(groups)), np.asarray(y), groups)
    else:
        # GroupKFold has no random_state -- it's deterministic by design
        splitter = GroupKFold(n_splits=n_splits)
        split_iter = splitter.split(np.zeros(len(groups)), None, groups)

    for train_idx, test_idx in split_iter:
        # belt and braces: verify what sklearn just handed us
        assert_no_subject_leakage(groups[train_idx], groups[test_idx])
        yield train_idx, test_idx


def fold_summary(groups, y, n_splits: int = config.N_FOLDS, stratified: bool = True):
    """Print how many subjects/windows of each class land in each test fold.

    Worth eyeballing once before you trust any result -- if a fold has 1
    menstrual subject, that fold's macro-F1 is basically a coin flip.
    """
    groups, y = np.asarray(groups), np.asarray(y)
    rows = []
    for k, (tr, te) in enumerate(subject_folds(groups, y, n_splits, stratified)):
        n_subj_te = len(set(groups[te]))
        per_class = {
            c: len(set(groups[te][y[te] == c])) for c in np.unique(y)
        }
        rows.append(
            {"fold": k, "train_windows": len(tr), "test_windows": len(te),
             "test_subjects": n_subj_te, **per_class}
        )
    return rows
