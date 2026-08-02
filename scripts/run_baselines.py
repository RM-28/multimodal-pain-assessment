#!/usr/bin/env python
"""EEG-only baselines: 5-fold subject-wise CV over the three nets + LightGBM.

    python scripts/run_baselines.py                 # everything
    python scripts/run_baselines.py --models cnn1d  # just one
    python scripts/run_baselines.py --epochs 3      # quick smoke test

Writes results/baselines_folds.csv (one row per model per fold) and
results/baselines_summary.csv (mean +- std). Best fold of each model gets
saved to models/ so we have something to point at.

Everything is grouped by subject. If that ever breaks, splits.subject_folds
raises rather than quietly handing back an inflated number.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import warnings

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from painnet import config, evaluate, models, splits, windows


def run_keras(name, build, X, y_int, groups, epochs, batch_size, log):
    """5 folds of one Keras architecture. Returns a list of per-fold dicts."""
    rows, best = [], (-1.0, None)

    for k, (tr, te) in enumerate(splits.subject_folds(groups, y_int)):
        t0 = time.time()
        model = models.compile_model(build(X.shape[1:]))
        hist = model.fit(
            X[tr], y_int[tr],
            validation_split=0.15,
            epochs=epochs, batch_size=batch_size, verbose=2,
            class_weight=windows.class_weights(y_int[tr]),
            callbacks=models.default_callbacks(),
        )
        pred = model.predict(X[te], verbose=0).argmax(1)

        row = {
            "model": name, "fold": k,
            **evaluate.fold_metrics(y_int[te], pred),
            **evaluate.per_class_f1(y_int[te], pred),
            "epochs_ran": len(hist.history["loss"]),
            "test_subjects": len(set(groups[te])),
            "seconds": round(time.time() - t0, 1),
        }
        rows.append(row)
        log(f"  {name} fold {k}: macro_f1={row['macro_f1']:.3f} "
            f"acc={row['accuracy']:.3f} ({row['epochs_ran']} epochs, {row['seconds']}s)")

        # hang on to the best fold so there's a saved model for the deliverable
        if row["macro_f1"] > best[0]:
            best = (row["macro_f1"], model)

    if best[1] is not None:
        config.ensure_dirs()
        path = config.MODELS_DIR / f"{name}_bestfold.keras"
        best[1].save(path)
        log(f"  saved {path.name} (macro_f1={best[0]:.3f})")

    return rows


def run_majority(y_int, groups, log):
    """Always predict the training set's most common class. The real floor.

    Worth being careful here: 0.361 is the majority class's share of subjects,
    so it's the ACCURACY of this model, not its macro-F1. Macro-F1 for a
    single-class predictor is roughly 0.53/4 ~= 0.13, because three of the four
    classes score a flat zero. Since macro-F1 is our headline number, quoting
    0.361 as "the bar" would set it about 3x too high. Measure it instead of
    doing the algebra in a docstring.
    """
    rows = []
    for k, (tr, te) in enumerate(splits.subject_folds(groups, y_int)):
        maj = np.bincount(y_int[tr]).argmax()
        pred = np.full(len(te), maj)
        rows.append({
            "model": "majority", "fold": k,
            **evaluate.fold_metrics(y_int[te], pred),
            **evaluate.per_class_f1(y_int[te], pred),
            "epochs_ran": None,
            "test_subjects": len(set(groups[te])),
            "seconds": 0.0,
        })
        log(f"  majority fold {k}: macro_f1={rows[-1]['macro_f1']:.3f} "
            f"acc={rows[-1]['accuracy']:.3f}")
    return rows


def run_lightgbm(X, y_int, groups, log):
    """Tabular baseline on window summary stats. Not deep learning, on purpose.

    Worth having: with 83 subjects, if a gradient-boosted tree on 40 summary
    numbers matches the conv nets, the honest read is that the extra capacity
    isn't earning its keep at this sample size.
    """
    import lightgbm as lgb

    Xt = windows.to_tabular(X)
    rows = []

    for k, (tr, te) in enumerate(splits.subject_folds(groups, y_int)):
        t0 = time.time()
        w = windows.class_weights(y_int[tr])
        clf = lgb.LGBMClassifier(
            objective="multiclass", num_class=len(config.PAIN_CLASSES),
            n_estimators=300, learning_rate=0.05, num_leaves=15,
            min_child_samples=40,          # small data, keep the leaves honest
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            random_state=config.SEED, verbose=-1,
        )
        clf.fit(Xt[tr], y_int[tr], sample_weight=[w[c] for c in y_int[tr]])
        pred = clf.predict(Xt[te])

        row = {
            "model": "lightgbm", "fold": k,
            **evaluate.fold_metrics(y_int[te], pred),
            **evaluate.per_class_f1(y_int[te], pred),
            "epochs_ran": None,
            "test_subjects": len(set(groups[te])),
            "seconds": round(time.time() - t0, 1),
        }
        rows.append(row)
        log(f"  lightgbm fold {k}: macro_f1={row['macro_f1']:.3f} "
            f"acc={row['accuracy']:.3f} ({row['seconds']}s)")

    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*",
                    default=["majority", "lightgbm", "cnn1d", "cnn_lstm", "tcn"])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--tag", default="baselines")
    args = ap.parse_args()

    def log(msg):
        print(msg, flush=True)   # flush so tailing the log actually shows progress

    log(f"=== {config.describe()} ===")
    t_start = time.time()

    X, y, groups = windows.build_dataset()
    y_int = windows.encode_labels(y)
    log(f"X {X.shape} | {len(set(groups))} subjects | "
        f"majority baseline {config.MAJORITY_BASELINE:.3f}")
    log(f"class counts: {dict(zip(*np.unique(y, return_counts=True)))}\n")

    all_rows = []
    for name in args.models:
        log(f"--- {name} ---")
        if name == "majority":
            all_rows += run_majority(y_int, groups, log)
        elif name == "lightgbm":
            all_rows += run_lightgbm(X, y_int, groups, log)
        else:
            all_rows += run_keras(name, models.BUILDERS[name], X, y_int, groups,
                                  args.epochs, args.batch_size, log)
        log("")

    config.ensure_dirs()
    folds = pd.DataFrame(all_rows)
    folds.to_csv(config.RESULTS_DIR / f"{args.tag}_folds.csv", index=False)

    # mean +- std across folds -- the only honest way to report this
    metric_cols = [c for c in folds.columns
                   if c not in ("model", "fold", "epochs_ran", "test_subjects", "seconds")]
    summary = folds.groupby("model")[metric_cols].agg(["mean", "std"]).round(4)
    summary.to_csv(config.RESULTS_DIR / f"{args.tag}_summary.csv")

    log("=" * 66)
    log(f"RESULTS  (mean +- std over {config.N_FOLDS} subject-wise folds)")
    if "majority" in folds.model.values:
        mj = folds[folds.model == "majority"]
        log(f"trivial floor: macro_f1={mj.macro_f1.mean():.3f}  "
            f"accuracy={mj.accuracy.mean():.3f}  <- beat BOTH of these")
    log("=" * 66)
    for m in folds.model.unique():
        s = folds[folds.model == m]
        log(f"\n{m}")
        log(f"  macro_f1     {s.macro_f1.mean():.3f} +- {s.macro_f1.std():.3f}")
        log(f"  balanced_acc {s.balanced_acc.mean():.3f} +- {s.balanced_acc.std():.3f}")
        log(f"  accuracy     {s.accuracy.mean():.3f} +- {s.accuracy.std():.3f}")
        per_class = "  ".join(
            f"{c}={s[f'f1_{c}'].mean():.2f}" for c in config.PAIN_CLASSES
        )
        log(f"  per-class F1  {per_class}")

    log(f"\ntotal wall clock: {(time.time()-t_start)/60:.1f} min")
    log(f"wrote {config.RESULTS_DIR}/{args.tag}_folds.csv and _summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
