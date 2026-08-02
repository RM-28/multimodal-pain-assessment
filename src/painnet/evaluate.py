"""Metrics that survive contact with 83 subjects and a 12% minority class.

Accuracy is nearly useless here -- always predicting 'headache' scores 0.36.
Macro-F1 is the headline number because it weights the 10-subject
menstrual_pain class the same as the 30-subject headache class.

Everything is reported mean +- std ACROSS FOLDS, never from a single split.
With ~17 test subjects per fold, one or two subjects flipping moves the number
several points -- a single-split figure would be noise dressed as a result.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
)

from . import config


def fold_metrics(y_true, y_pred, labels=None) -> dict:
    """The per-fold numbers we collect."""
    labels = labels if labels is not None else list(range(len(config.PAIN_CLASSES)))
    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
        "accuracy": float(np.mean(np.asarray(y_true) == np.asarray(y_pred))),
    }


def per_class_f1(y_true, y_pred) -> dict:
    """F1 broken out by class -- the table that actually tells the story."""
    scores = f1_score(
        y_true, y_pred, average=None,
        labels=list(range(len(config.PAIN_CLASSES))), zero_division=0,
    )
    return {f"f1_{c}": float(s) for c, s in zip(config.PAIN_CLASSES, scores)}


def aggregate(fold_rows: list[dict]) -> pd.DataFrame:
    """mean +- std across folds. This is what goes in the report."""
    df = pd.DataFrame(fold_rows)
    summary = df.agg(["mean", "std"]).T
    summary.columns = ["mean", "std"]
    summary["report"] = summary.apply(lambda r: f"{r['mean']:.3f} ± {r['std']:.3f}", axis=1)
    return summary


def confusion(y_true, y_pred, normalize: str | None = None) -> pd.DataFrame:
    cm = confusion_matrix(
        y_true, y_pred,
        labels=list(range(len(config.PAIN_CLASSES))),
        normalize=normalize,
    )
    return pd.DataFrame(cm, index=config.PAIN_CLASSES, columns=config.PAIN_CLASSES)


def text_report(y_true, y_pred) -> str:
    return classification_report(
        y_true, y_pred,
        labels=list(range(len(config.PAIN_CLASSES))),
        target_names=config.PAIN_CLASSES,
        zero_division=0,
    )


# --- ordinal intensity ------------------------------------------------------
# Only meaningful on the 68 subjects who reported pain, and even then the top
# level ("Very severe") has 4 subjects total. Report with heavy caveats.

def ordinal_metrics(y_true, y_pred) -> dict:
    """MAE plus quadratic weighted kappa.

    QWK is the right call for an ordinal target: predicting 'Mild' when the
    truth is 'Very severe' should cost more than predicting 'Moderate'. Plain
    accuracy treats both as equally wrong.
    """
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "qwk": cohen_kappa_score(y_true, y_pred, weights="quadratic"),
    }


# --- the confound probe -----------------------------------------------------

def sex_confound_check(y_sex_true, y_sex_pred) -> dict:
    """How well do these same features predict biological sex?

    All 10 menstrual_pain subjects are female and none are male, so a model can
    score on that class by detecting sex rather than pain. Train an identical
    model on a sex target and compare: if sex is easier than pain, you've
    measured the size of the shortcut. Report it, don't hide it.
    """
    return {
        "sex_macro_f1": f1_score(y_sex_true, y_sex_pred, average="macro", zero_division=0),
        "sex_accuracy": float(np.mean(np.asarray(y_sex_true) == np.asarray(y_sex_pred))),
    }
