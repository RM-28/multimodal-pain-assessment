# Verification notes: fusion reproduction and the sex-confound probe

Run 2026-08-08 on Jonathan's machine (`ai570` env, CPU). Raw fold tables:
`results/fusion_repro_folds.csv`, `results/sex_probe_folds.csv`.

## 1. Independent rerun of the fusion comparison

`scripts/run_fusion_repro.py` re-executes the Test 2 protocol from
`notebooks/fusion.ipynb` outside Colab: same paired dataset
(77 subjects, 5,936 windows — matches the notebook build exactly), same nested
subject-disjoint splits, same builders, callbacks, and LightGBM hyperparameters.

macro-F1, mean ± std over 5 folds:

| model | this rerun | fusion.ipynb (Colab) |
|---|---|---|
| EEG TCN | 0.196 ± 0.042 | 0.176 ± 0.031 |
| Wristband LightGBM | **0.291** ± 0.056 | **0.218** ± 0.079 |
| Dual-branch fusion | 0.241 ± 0.051 | 0.202 ± 0.062 |

**What reproduces:** the ranking (wristband > fusion > EEG) and both headline
comparisons. Fusion beat the EEG branch in 4/5 folds here (3/5 in the
notebook) and beat the wristband in 0/5 (1/5 in the notebook). The proposal's
prediction that fusion wins outright fails on both machines.

**What does not reproduce:** the absolute numbers, which run 0.02-0.07 higher
here. Every mean is inside the other run's fold-to-fold std, and the shift
moves all three models together. Likely cause: different sklearn / LightGBM
builds assign different subjects to folds even at the same seed, and with ~15
test subjects per fold the fold lottery moves means by exactly this much.

**Reporting guidance:** quote the notebook's table as the primary result and
cite this rerun as confirmation of the ranking. Do not mix numbers from the two
runs in one table. The cross-machine shift is itself worth one sentence in the
report: exact scores on 77 subjects are not portable, which is one more reason
every number carries a std.

## 2. The sex-confound probe, now run

All 10 menstrual-pain subjects are female. `scripts/run_sex_probe.py` trains
the same TCN and LightGBM, on the same paired windows and the same subject-wise
folds, against two targets: pain type (4-class) and biological sex (2-class,
46 F / 31 M among the 77 paired subjects). The floors differ by construction,
so the comparable number is each model's lift over its own majority floor.

macro-F1, mean ± std over 5 folds:

| target | floor | LightGBM | lift | TCN | lift |
|---|---|---|---|---|---|
| Wristband, pain | 0.131 | 0.278 ± 0.044 | +0.148 | 0.239 ± 0.055 | +0.108 |
| Wristband, sex | 0.363 | 0.579 ± 0.056 | **+0.216** | 0.515 ± 0.038 | +0.152 |
| EEG, pain | 0.131 | 0.225 ± 0.081 | +0.095 | 0.270 ± 0.038 | +0.140 |
| EEG, sex | 0.363 | **0.712** ± 0.129 | **+0.348** | 0.599 ± 0.098 | +0.236 |

**Finding: sex is easier to decode than pain from both modalities, by a wide
margin.** The surprise is which modality leaks more. We expected the wristband
(EDA, temperature, and heart-rate physiology all differ by sex). Instead the
EEG band powers are the stronger sex signal: 0.72 balanced accuracy against a
0.50 floor, versus 0.31 balanced accuracy on the 4-class pain task from the
same windows. Sex differences in EEG spectral power are documented in the
literature, so this is plausible rather than anomalous, but we did not predict
it.

**What this means for the pain results:** any per-class score on
menstrual_pain must be read knowing the model can reach that class through a
signal (sex) it decodes far more reliably than pain. With 10 subjects we cannot
remove the confound; we have now measured it, which is the honest alternative.
This upgrades the slide-14 caveat from "designed, not run" to a quantified
result.

## 3. Checks that came along free

- `watch.build_fusion_dataset()` on this machine yields byte-for-byte the same
  counts as the notebook: 86 watch subjects, 77 common, 5,936 windows,
  per-class subject counts 15/27/25/10.
- The row-0 rate leak is present in the freshly extracted files and absent
  after `watch.py`'s loader, confirmed by hand on `S006_4Hz.csv`.
- All nested folds passed the triple-disjointness assertions in both scripts.
