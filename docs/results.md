# Results

Every number here is a mean and standard deviation across 5 subject-wise folds.
Raw per-fold tables live in `results/*.csv`. Figures in `docs/figures/`.

Two subject pools appear below and they must never share a table:

* **83 subjects** have usable EEG. Used for the EEG-only experiments.
* **77 subjects** wore both devices. Used for anything comparing modalities.

## 1. Trivial floor

A model that always predicts the majority class scores:

| metric | value |
|---|---|
| accuracy | 0.360 |
| macro-F1 | **0.132** |

Macro-F1 is our headline metric because the classes are unbalanced, so 0.132 is
the number to beat. The 0.361 figure that appears in our proposal is the
majority class's share of subjects, which is this model's accuracy, not its
macro-F1. Quoting it as the bar sets it roughly three times too high.

## 2. EEG only, 83 subjects

`scripts/run_baselines.py`, figure `eeg_baselines_folds.png`.

| model | macro-F1 | balanced acc | accuracy |
|---|---|---|---|
| **TCN** | **0.240 ± 0.046** | 0.279 ± 0.020 | 0.257 ± 0.047 |
| LightGBM | 0.235 ± 0.098 | 0.255 ± 0.100 | 0.288 ± 0.104 |
| 1D CNN | 0.221 ± 0.064 | 0.294 ± 0.043 | 0.250 ± 0.056 |
| CNN-LSTM | 0.199 ± 0.040 | 0.272 ± 0.034 | 0.224 ± 0.027 |
| majority floor | 0.132 ± 0.003 | 0.250 | 0.360 ± 0.010 |

All three networks clear the macro-F1 floor with significance (paired across
folds: TCN p=0.007, CNN-LSTM p=0.024, 1D CNN p=0.036). LightGBM does not
(p=0.084), because its spread is enormous.

Only the TCN is significantly above chance on balanced accuracy (0.279 vs 0.250,
p=0.029). Nothing beats the majority model on raw accuracy, which is the class
weighting trading accuracy for coverage of the rare classes. Worth stating
plainly rather than leading with macro-F1 alone.

**The fold spread is the finding.** LightGBM ranges 0.114 to 0.355 across folds.
On fold 4 the 1D CNN records its worst score of the project while LightGBM
records its best, on identical test subjects. Evaluate on one split and you
could conclude either that trees beat deep learning threefold or the reverse.
See `fold4_reversal.png`.

## 3. Modality comparison, 77 paired subjects

Meng's `notebooks/fusion.ipynb`, Test 2. Nested splits: train, validation and
test are three disjoint sets of people, roughly 47/15/15 per fold.

| model | macro-F1 |
|---|---|
| Wristband LightGBM | **0.218 ± 0.079** |
| Dual-branch fusion CNN | 0.202 ± 0.062 |
| EEG TCN | 0.176 ± 0.031 |

Fusion beat the EEG branch in 3 of 5 folds but beat the wristband in only 1 of
5. **Our proposal predicted fusion would outperform both single modalities. It
does not.** The wristband alone is the strongest single signal, and the weak EEG
branch appears to dilute the fusion rather than complement it. Per-class the
picture is more interesting: fusion is best on headache, wristband on back pain,
EEG on menstrual pain.

Matched-architecture ablation (all three as CNNs) ranks fusion 0.202 > wristband
0.183 > EEG 0.178, so the fusion advantage over single modalities is real when
architecture is held constant. It is the strong non-neural wristband baseline
that beats it.

### Reproduction

`scripts/run_fusion_repro.py` reran this protocol on a second machine. The
ranking reproduces (wristband > fusion > EEG) and so do both comparisons
(4/5 and 0/5 folds). Absolute means come out 0.02 to 0.07 higher, every one
inside the other run's fold-to-fold standard deviation.

The cause is worth a sentence in the report: different scikit-learn builds
assign different subjects to folds at the same seed, and with 15 test subjects
per fold that shifts the mean by this much. Exact scores on 77 people do not
survive a library upgrade. Quote the notebook's table as primary and this run as
confirmation of the ranking.

## 4. The sex confound, measured

`scripts/run_sex_probe.py`. All 10 menstrual-pain subjects are female and no
male carries that label, so a model can reach that class by detecting sex rather
than pain. We trained the same models on the same folds against a sex target
(46 F / 31 M among the 77 paired subjects). Floors differ between a 2-class and
a 4-class problem, so the comparable quantity is each model's lift over its own
floor.

| target | floor | LightGBM | lift | TCN | lift |
|---|---|---|---|---|---|
| Wristband, pain | 0.131 | 0.278 ± 0.044 | +0.148 | 0.239 ± 0.055 | +0.108 |
| Wristband, sex | 0.363 | 0.579 ± 0.056 | +0.216 | 0.515 ± 0.038 | +0.152 |
| EEG, pain | 0.131 | 0.225 ± 0.081 | +0.095 | 0.270 ± 0.038 | +0.140 |
| **EEG, sex** | 0.363 | **0.712 ± 0.129** | **+0.348** | 0.599 ± 0.098 | +0.236 |

**Sex is easier to decode than pain from both modalities, and the EEG leaks it
worst.** From the same windows where 4-class pain reaches 0.31 balanced
accuracy, sex reaches 0.72 against a 0.50 floor. We expected the wristband to be
the leaky modality, since skin conductance, temperature and heart rate all
differ by sex. Sex differences in EEG spectral power are documented, so the
result is plausible, but we did not predict it.

Any per-class score on menstrual pain has to be read knowing the model has a far
more reliable route into that class than pain itself. With 10 subjects the
confound cannot be removed. It can be measured, which is what this is.

## 5. Interpreting the scale of these numbers

Published work using multi-channel research EEG, on a within-subject binary
task, reports AUC around 0.83. We use one dry frontal electrode, four classes,
and subject-independent evaluation with strangers in the test set. Modest
numbers are the expected outcome, and a near-perfect score on this dataset would
indicate leakage rather than success.

The contribution is the evaluation discipline: subject-wise splitting enforced
in code, a measured rather than assumed floor, mean and standard deviation over
folds, and a quantified confound.
