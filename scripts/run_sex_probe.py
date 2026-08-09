#!/usr/bin/env python
"""The sex-confound probe: how much of our signal is just 'is this a woman?'

All 10 menstrual_pain subjects are female and no male carries that label, so a
model can score on that class by detecting sex rather than pain. This trains
the SAME architectures on a sex target with the SAME subject-wise folds and
puts the two tasks side by side. If sex is easier to read from the signals than
pain, the shortcut is real and we can say how big it is.

    python scripts/run_sex_probe.py

Writes results/sex_probe_folds.csv and prints the comparison.
"""
from __future__ import annotations

import os
import time
import warnings

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import keras
import lightgbm as lgb
from sklearn.metrics import balanced_accuracy_score, f1_score

from painnet import config, data, models, splits, watch, windows

SEX_CLASSES = ["Female", "Male"]


def sex_map():
    """subject id -> 0 (Female) / 1 (Male), from the survey."""
    sv = data.load_survey()
    col = sv["Gender (Biological)"]
    return {sid: SEX_CLASSES.index(v) for sid, v in col.items() if v in SEX_CLASSES}


def binary_metrics(y_true, y_pred):
    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
        "accuracy": float(np.mean(np.asarray(y_true) == np.asarray(y_pred))),
    }


def lgbm(n_classes):
    return lgb.LGBMClassifier(
        objective="multiclass", num_class=n_classes,
        n_estimators=300, learning_rate=0.05, num_leaves=15,
        min_child_samples=40, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, random_state=config.SEED, verbose=-1)


def run_task(tag, X, X_tab, y_int, groups, n_classes, tcn_input, log):
    """One target (pain or sex) through TCN + LightGBM, same folds."""
    rows = []
    for k, (tr, te) in enumerate(splits.subject_folds(groups, y_int)):
        # majority floor for this fold
        maj = np.bincount(y_int[tr]).argmax()
        rows.append({"task": tag, "model": "majority", "fold": k,
                     **binary_metrics(y_int[te], np.full(len(te), maj))})

        # TCN, sized to the target
        t = time.time()
        keras.backend.clear_session()
        m = models.compile_model(models.build_tcn(tcn_input, n_classes=n_classes))
        m.fit(X[tr], y_int[tr], validation_split=0.15,
              epochs=60, batch_size=64, verbose=0,
              class_weight=windows.class_weights(y_int[tr]),
              callbacks=models.default_callbacks(patience=8))
        pred = m.predict(X[te], verbose=0).argmax(1)
        rows.append({"task": tag, "model": "tcn", "fold": k,
                     **binary_metrics(y_int[te], pred)})
        log(f"  {tag} tcn fold {k}: macro_f1={rows[-1]['macro_f1']:.3f} ({time.time()-t:.0f}s)")

        # LightGBM on the tabular stats
        w = windows.class_weights(y_int[tr])
        clf = lgbm(n_classes)
        clf.fit(X_tab[tr], y_int[tr], sample_weight=[w[c] for c in y_int[tr]])
        rows.append({"task": tag, "model": "lightgbm", "fold": k,
                     **binary_metrics(y_int[te], clf.predict(X_tab[te]))})
    return rows


def main() -> int:
    t0 = time.time()
    log = lambda s: print(s, flush=True)
    log(f"=== {config.describe()} ===")

    smap = sex_map()

    # --- wristband, paired tier (where the confound should live: EDA/temp/BVP
    # all carry sex differences) -------------------------------------------
    X_eeg, X_watch, y, groups = watch.build_fusion_dataset()
    y_pain = windows.encode_labels(y)
    y_sex = np.array([smap[s] for s in groups])
    Xw_tab = windows.to_tabular(X_watch)
    Xe_tab = windows.to_tabular(X_eeg)

    f, m = (y_sex == 0).sum(), (y_sex == 1).sum()
    subj_f = len({s for s in set(groups) if smap[s] == 0})
    log(f"paired subjects: {len(set(groups))} ({subj_f} F / "
        f"{len(set(groups)) - subj_f} M) | windows {len(y_sex)} ({f} F / {m} M)\n")

    rows = []
    log("--- WRISTBAND: pain (4-class) vs sex (2-class) ---")
    rows += run_task("watch_pain", X_watch, Xw_tab, y_pain, groups,
                     len(config.PAIN_CLASSES), X_watch.shape[1:], log)
    rows += run_task("watch_sex", X_watch, Xw_tab, y_sex, groups,
                     2, X_watch.shape[1:], log)

    log("--- EEG: pain vs sex ---")
    rows += run_task("eeg_pain", X_eeg, Xe_tab, y_pain, groups,
                     len(config.PAIN_CLASSES), X_eeg.shape[1:], log)
    rows += run_task("eeg_sex", X_eeg, Xe_tab, y_sex, groups,
                     2, X_eeg.shape[1:], log)

    config.ensure_dirs()
    df = pd.DataFrame(rows)
    df.to_csv(config.RESULTS_DIR / "sex_probe_folds.csv", index=False)

    log("\n" + "=" * 66)
    log("SEX PROBE  (macro-F1 mean +- std over 5 subject-wise folds)")
    log("nb: 4-class pain and 2-class sex have different chance floors --")
    log("    that's what the per-task majority row is for.")
    log("=" * 66)
    for task in ["watch_pain", "watch_sex", "eeg_pain", "eeg_sex"]:
        s = df[df.task == task]
        log(f"\n{task}")
        for mod in ["majority", "lightgbm", "tcn"]:
            r = s[s.model == mod]
            log(f"  {mod:9s} macro_f1 {r.macro_f1.mean():.3f} +- {r.macro_f1.std():.3f}"
                f"   bal_acc {r.balanced_acc.mean():.3f}")
        # lift over its own floor is the comparable number across tasks
        for mod in ["lightgbm", "tcn"]:
            lift = (s[s.model == mod].macro_f1.values
                    - s[s.model == "majority"].macro_f1.values).mean()
            log(f"  {mod} lift over floor: {lift:+.3f}")

    log(f"\ntotal: {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
