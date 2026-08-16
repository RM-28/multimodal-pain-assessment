#!/usr/bin/env python
"""Measure what a leaked split actually scores, so the demo quotes a real number.

Same data, same model, same 80/20 ratio. The only difference is whether the
split respects participant boundaries. Run it rather than assert it.

    python scripts/run_leakage_demo.py
"""
from __future__ import annotations

import os
import warnings

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
warnings.filterwarnings("ignore")

import numpy as np
import keras

from painnet import config, evaluate, models, splits, windows


def train_eval(Xtr, ytr, Xte, yte):
    keras.backend.clear_session()
    keras.utils.set_random_seed(config.SEED)
    m = models.compile_model(models.build_tcn(Xtr.shape[1:]))
    m.fit(Xtr, ytr, validation_split=0.15, epochs=60, batch_size=64, verbose=0,
          class_weight=windows.class_weights(ytr),
          callbacks=models.default_callbacks(patience=8))
    pred = m.predict(Xte, verbose=0).argmax(1)
    return evaluate.fold_metrics(yte, pred)


def main() -> int:
    X, y, groups = windows.build_dataset()
    y_int = windows.encode_labels(y)
    rng = np.random.default_rng(config.SEED)
    print(f"{X.shape[0]} windows from {len(set(groups))} participants\n", flush=True)

    # --- the leak: shuffle windows, ignore who they came from ---------------
    perm = rng.permutation(len(y_int))
    cut = int(0.8 * len(perm))
    tr, te = perm[:cut], perm[cut:]
    shared = len(set(groups[tr]) & set(groups[te]))
    print(f"random window split: {shared} of {len(set(groups))} participants "
          f"appear on BOTH sides", flush=True)
    leaked = train_eval(X[tr], y_int[tr], X[te], y_int[te])
    print(f"  accuracy {leaked['accuracy']:.3f}   macro-F1 {leaked['macro_f1']:.3f}\n",
          flush=True)

    # --- the honest one: same ratio, participants kept whole ---------------
    tr2, te2 = next(iter(splits.subject_folds(groups, y_int, n_splits=5)))
    shared2 = len(set(groups[tr2]) & set(groups[te2]))
    print(f"subject-wise split: {shared2} participants on both sides", flush=True)
    honest = train_eval(X[tr2], y_int[tr2], X[te2], y_int[te2])
    print(f"  accuracy {honest['accuracy']:.3f}   macro-F1 {honest['macro_f1']:.3f}\n",
          flush=True)

    print("=" * 58)
    print(f"  leaked  accuracy {leaked['accuracy']:.3f}  macro-F1 {leaked['macro_f1']:.3f}")
    print(f"  honest  accuracy {honest['accuracy']:.3f}  macro-F1 {honest['macro_f1']:.3f}")
    print(f"  inflation factor on macro-F1: "
          f"{leaked['macro_f1']/max(honest['macro_f1'],1e-9):.1f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
