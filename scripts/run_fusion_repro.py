#!/usr/bin/env python
"""Independent rerun of Meng's fusion comparison (notebooks/fusion.ipynb, Test 2).

Same protocol, different machine: paired dataset from watch.build_fusion_dataset,
nested subject-disjoint train/val/test folds, then EEG TCN vs wristband LightGBM
vs dual-branch fusion CNN. Her numbers came off Colab; if ours land within the
fold-to-fold std of hers, the table is trustworthy for the report.

    python scripts/run_fusion_repro.py

Writes results/fusion_repro_folds.csv + _summary.csv.
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

from painnet import config, evaluate, models, splits, watch, windows


def nested_folds(groups, y_int, n_outer=5, n_inner=4):
    """Meng's split: outer test fold + subject-disjoint val carved from train."""
    folds = []
    for outer_train_idx, test_idx in splits.subject_folds(groups, y_int, n_splits=n_outer):
        inner = splits.subject_folds(
            groups[outer_train_idx], y_int[outer_train_idx], n_splits=n_inner
        )
        inner_train_rel, val_rel = next(inner)
        train_idx = outer_train_idx[inner_train_rel]
        val_idx = outer_train_idx[val_rel]
        # same triple-disjoint assertion her notebook makes
        assert set(groups[train_idx]).isdisjoint(groups[val_idx])
        assert set(groups[train_idx]).isdisjoint(groups[test_idx])
        assert set(groups[val_idx]).isdisjoint(groups[test_idx])
        folds.append((train_idx, val_idx, test_idx))
    return folds


def main() -> int:
    t0 = time.time()
    print(f"=== {config.describe()} ===", flush=True)

    X_eeg, X_watch, y, groups = watch.build_fusion_dataset()
    y_int = windows.encode_labels(y)
    print(f"paired: X_eeg {X_eeg.shape} X_watch {X_watch.shape} "
          f"subjects {len(set(groups))}", flush=True)

    folds = nested_folds(groups, y_int)
    Xw_tab = windows.to_tabular(X_watch)

    rows = []

    def record(name, k, test_idx, pred, secs, epochs=None):
        rows.append({
            "model": name, "fold": k,
            **evaluate.fold_metrics(y_int[test_idx], pred),
            **evaluate.per_class_f1(y_int[test_idx], pred),
            "epochs_ran": epochs, "seconds": round(secs, 1),
        })
        print(f"  {name} fold {k}: macro_f1={rows[-1]['macro_f1']:.3f} "
              f"acc={rows[-1]['accuracy']:.3f}", flush=True)

    for k, (tr, va, te) in enumerate(folds):
        # --- EEG TCN ---
        t = time.time()
        keras.backend.clear_session()
        m = models.compile_model(models.build_tcn(X_eeg.shape[1:]))
        h = m.fit(X_eeg[tr], y_int[tr], validation_data=(X_eeg[va], y_int[va]),
                  epochs=60, batch_size=64, verbose=2,
                  class_weight=windows.class_weights(y_int[tr]),
                  callbacks=models.default_callbacks(patience=8))
        record("eeg_tcn", k, te, m.predict(X_eeg[te], verbose=0).argmax(1),
               time.time() - t, len(h.history["loss"]))

        # --- wristband LightGBM (her exact hyperparams == run_baselines') ---
        t = time.time()
        w = windows.class_weights(y_int[tr])
        clf = lgb.LGBMClassifier(
            objective="multiclass", num_class=len(config.PAIN_CLASSES),
            n_estimators=300, learning_rate=0.05, num_leaves=15,
            min_child_samples=40, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, random_state=config.SEED, verbose=-1)
        clf.fit(Xw_tab[tr], y_int[tr], sample_weight=[w[c] for c in y_int[tr]])
        record("watch_lightgbm", k, te, clf.predict(Xw_tab[te]), time.time() - t)

        # --- dual-branch fusion CNN ---
        t = time.time()
        keras.backend.clear_session()
        m = models.compile_model(models.build_fusion(X_eeg.shape[1:], X_watch.shape[1:]))
        h = m.fit([X_eeg[tr], X_watch[tr]], y_int[tr],
                  validation_data=([X_eeg[va], X_watch[va]], y_int[va]),
                  epochs=60, batch_size=64, verbose=2,
                  class_weight=windows.class_weights(y_int[tr]),
                  callbacks=models.default_callbacks(patience=8))
        record("fusion_cnn", k, te,
               m.predict([X_eeg[te], X_watch[te]], verbose=0).argmax(1),
               time.time() - t, len(h.history["loss"]))

    config.ensure_dirs()
    df = pd.DataFrame(rows)
    df.to_csv(config.RESULTS_DIR / "fusion_repro_folds.csv", index=False)

    print("\n" + "=" * 64)
    print("REPRO RESULTS vs Meng's notebook (macro-F1 mean +- std)")
    print("=" * 64)
    MENG = {"eeg_tcn": "0.176 ± 0.031", "watch_lightgbm": "0.218 ± 0.079",
            "fusion_cnn": "0.202 ± 0.062"}
    summary = []
    for mname in ["eeg_tcn", "watch_lightgbm", "fusion_cnn"]:
        s = df[df.model == mname]
        ours = f"{s.macro_f1.mean():.3f} ± {s.macro_f1.std():.3f}"
        print(f"  {mname:16s} ours={ours}   meng={MENG[mname]}")
        summary.append({"model": mname, "ours_macro_f1": ours, "meng_macro_f1": MENG[mname],
                        "ours_acc": f"{s.accuracy.mean():.3f} ± {s.accuracy.std():.3f}"})
    pd.DataFrame(summary).to_csv(config.RESULTS_DIR / "fusion_repro_summary.csv", index=False)

    fc = df.pivot_table(index="fold", columns="model", values="macro_f1")
    print("\nper-fold comparison:")
    print((fc["fusion_cnn"] - fc["eeg_tcn"]).rename("fusion-tcn").to_string())
    print(f"fusion beat tcn in {(fc.fusion_cnn > fc.eeg_tcn).sum()}/5 folds")
    print(f"fusion beat lgbm in {(fc.fusion_cnn > fc.watch_lightgbm).sum()}/5 folds")
    print(f"\ntotal: {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
