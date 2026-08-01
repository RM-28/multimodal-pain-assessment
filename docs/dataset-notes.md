# PhysioPain — dataset notes

Everything here was measured against the actual archive on 2026-07-31, not taken
from the paper or the Kaggle page. Where our numbers disagree with a published
figure, both are given.

**Read this before writing modelling code.** Several of these findings will
silently inflate your accuracy if you don't design around them.

---

## 1. What you get

| | |
|---|---|
| Source | Kaggle `orvile/physiopain-dataset` · Mendeley DOI `10.17632/mf2cgph9cy.4` |
| Licence | CC BY 4.0 (per the dataset description) |
| Download | **1.26 GB** zipped, **5.36 GB** extracted, 2,215 files |
| Auth | **None needed.** Plain `curl -L` on the API download endpoint works. |
| Devices | NeuroSky MindWave Mobile 2 (EEG) + Empatica E4 (wristband) |
| Institution | Istanbul Kültür Üniversitesi |

Kaggle's `totalBytes` field reports **5,362,178,803** — that's the *uncompressed*
size. The actual download is **1,264,163,874** bytes. Don't budget disk off the
first number.

### Directory tree, with real sizes

```
PhysioPain Dataset/Multimodal Pain Dataset/
├── RAW EEG DATA (1Hz)/        166 files     19.1 MB   ← we use this
│   ├── All/                    83 files                 one CSV per subject
│   └── Categorical/            83 files                 same data, foldered by class
├── PROCESSED EEG DATA/        199 files    305.7 MB   ← avoid, see §4
├── RAW WATCH DATA/           1376 files    171.8 MB
├── PROCESSED WATCH DATA/      472 files   4865.6 MB   ← the 4.9 GB, for fusion
└── SURVEY DATA/                 2 files      0.1 MB   ← intensity labels live here
```

The EEG tier we model on is **19 MB**. You do not need the other 5.3 GB unless
you're building the wristband branch.

---

## 2. The finding that shapes the whole project

**Every participant has exactly one `pain_type` for their entire recording.**

Verified directly: of 83 subjects in `RAW EEG DATA (1Hz)/All/`, **zero** have
more than one label. S004 is `headache` for all 1,197 of its rows; S031 is
`no_pain` throughout.

### Why this matters more than anything else

A window's label is determined entirely by *which person it came from*. So:

- **Effective sample size is 83, not 101,356 rows or 6,465 windows.**
- A random window-level split trains and tests on overlapping windows from the
  same person — the model scores by recognising individuals, and you get a
  meaningless 95%+.
- Grouped CV on `id` is not a nicety. It is the only thing making the numbers mean
  anything.
- The reference implementation's `--min_majority_frac` window-purity filter is a
  no-op here: every window is already 100% pure by construction.

`painnet.splits` enforces this and raises `LeakageError` otherwise.

---

## 3. Class balance — count subjects, not rows

| class | **subjects** | rows | subj % | ≈ per test fold (5-fold) |
|---|---|---|---|---|
| headache | 30 | 36,492 | 36.1% | ~6 |
| back_pain | 28 | 33,766 | 33.7% | ~5.6 |
| no_pain | 15 | 18,016 | 18.1% | ~3 |
| **menstrual_pain** | **10** | 13,082 | 12.0% | **~2** |
| total | **83** | 101,356 | | ~17 |

**Majority-class baseline: 0.361.**

Row counts track subject counts closely, so they add no information — quoting
"101,356 samples" would misrepresent the statistical power by three orders of
magnitude.

Use `StratifiedGroupKFold`, not plain `GroupKFold`: with only 10 menstrual
subjects, an unstratified split can hand you a test fold containing none, and
macro-F1 for that fold becomes undefined. Our stratified folds give every fold
2 menstrual / 3 no_pain / 6 headache / 5–6 back_pain subjects.

### Recording length

Ragged: **740 – 1,826 seconds**, median 1,228, mean 1,221 ± 119. Sequence models
need windowing (what we do), padding, or truncation. At 60 s windows / 15 s
stride this yields **6,465 windows** (46–118 per subject).

---

## 4. Two data-integrity defects

### (a) The processed EEG tier duplicates comorbid subjects — **avoid it**

11 subjects appear in 2–3 pain-type folders with **byte-identical signal
matrices**. `S057` appears under `back_pain`, `headache` *and* `menstrual_pain`,
and all three 1,251×8 matrices are `np.allclose`-identical. `S031` appears as
both `headache` and `no_pain` with identical 1,231×8 matrices.

Roughly **26.5%** of rows in the merged processed file carry more than one label.
Consequences:

- A hard accuracy ceiling well under 100%, since identical inputs have conflicting targets.
- Grouping on `(person_id, pain_version)` gives **95** groups and splits identical
  rows across folds — silent leakage. **Group on `person_id` alone (83 groups).**

**Mitigation: use `RAW EEG DATA (1Hz)/All/`**, which is disjoint — 83 files, 83
unique subjects, one label each. This is what `painnet.data.load_raw_eeg()` reads.
The wristband tier is also label-disjoint and unaffected.

### (b) Empatica sample-rate rows leaked into the processed watch CSVs

The E4 writes the unix timestamp on row 1 and the **sample rate on row 2**. The
dataset authors' script stripped row 1 but not row 2, so the rate values survive
as the first data row. In `PROCESSED WATCH DATA/back_pain/signal_4/S006_4Hz.csv`,
row 0 reads `eda=4.0, temperature=4.0` — literally the 4 Hz rate.

Confirmed for the natively-4 Hz channels (`eda`, `temperature`); the resampled
channels (`bvp`, `x/y/z`) appear unaffected in the file checked. **Drop row 0 of
every processed watch file regardless** — `painnet.data.load_watch()` does.

---

## 5. Columns

```
RAW EEG   id, obs, time, Delta, Theta, Alpha1, Alpha2, Beta1, Beta2,
          Gamma1, Gamma2, Attention, Meditation, Derived, totPwr, class, pain_type
```

Headers carry **leading spaces** (`' time'`, `' Delta'` … and `' class '` has a
trailing one too). `id`, `obs`, `pain_type` are clean. Read with
`skipinitialspace=True` and strip the column names.

| column | verdict | why |
|---|---|---|
| `Delta … Gamma2` (8 bands) | **keep** | the actual signal |
| `Attention`, `Meditation` | **keep** | NeuroSky eSense scores, only 0.28% NA |
| `Derived` | **drop** | 100.00% NA in every file |
| `totPwr` | **drop** | *exactly* `sum(bands)` — correlation 1.0, max abs diff 0 |
| `class` | **drop for now** | undocumented; values `X` (82,020) / `X*` (13,167) / `1` (5,294) / `1*` (875). Never takes `1` for menstrual subjects. Meaning unknown — investigate before using. |
| `obs`, `time` | drop | row index and clock |

### Scale

Band powers span ~4 orders of magnitude (Delta mean 4.8e5, max 4.0e6; Gamma2
min 24). Per-subject z-scoring is doing real work — without it the network mostly
learns how well each participant's headset was seated.

Note this is a **single-channel** consumer EEG (one frontal dry electrode), so
there is no spatial information and no re-referencing or ICA to be done. That's a
genuine limitation worth stating in the report.

---

## 6. Pain intensity is in the survey, not the EEG files

`SURVEY DATA/survey_answers_en.xlsx` — 99 rows × 108 columns, keyed by `id`.
All 83 EEG subjects are present (16 survey subjects have no EEG).

The intensity label is:

> `Rate the severity of your pain (Likert scale) [How severe is your pain now?]`

on a **5**-point scale — `Not severe at all` / `Mild` / `Moderate` / `Severe` /
`Very severe`. (Our proposal assumed 4 levels; it's 5.)

### It is not viable as a 5-class subject-level target

| severity | back_pain | headache | menstrual | no_pain | **total** |
|---|---|---|---|---|---|
| (missing) | 0 | 0 | 0 | 12 | 12 |
| Not severe at all | 0 | 1 | 0 | 3 | 4 |
| Mild | 12 | 11 | 1 | 0 | 24 |
| Moderate | 10 | 12 | 4 | 0 | 26 |
| Severe | 4 | 6 | 3 | 0 | 13 |
| **Very severe** | 2 | 0 | 2 | 0 | **4** |

**"Very severe" has 4 subjects in the entire dataset** — under 1 per test fold.
All `no_pain` subjects are missing or "Not severe at all", so intensity is
partly collinear with type.

Options: collapse to binary (mild vs moderate-and-above), restrict to the 68 pain
subjects, or drop the secondary target. Do **not** report a 5-class ordinal result
as if it were estimable.

---

## 7. Confounds and label-quality issues

### `menstrual_pain` is perfectly confounded with sex

| | back_pain | headache | menstrual_pain | no_pain |
|---|---|---|---|---|
| Female | 16 | 14 | **10** | 6 |
| Male | 12 | 16 | **0** | 9 |

All 10 menstrual subjects are female; no male subject carries the label. Since
sex has real effects on EDA, heart rate and skin temperature, a model can score
on this class by detecting sex rather than pain — especially the wristband branch.

**Recommended treatment:** train an identical model on a *sex* target and report
its performance alongside the pain model. If sex is easier to predict, you have
quantified the size of the shortcut. `painnet.evaluate.sex_confound_check()`
exists for this. This is a reportable result, not a flaw to hide.

### Two subjects disagree with their own self-report

Survey pain type vs. EEG `pain_type`, cross-tabulated on the 83 merged subjects:
everything matches **except two subjects — `S059` and `S071` — labelled
`menstrual_pain` in the EEG files who self-reported "Abdominal pain"**. So that
10-subject class is really 8 menstrual + 2 abdominal.
`painnet.data.subject_table()` flags these as `label_disagrees`; decide
per-analysis whether to drop them.

The survey also offers a fifth type ("Abdominal pain") that the EEG tier's
`Categorical/` folders don't have.

### Age

| class | n | mean age | sd |
|---|---|---|---|
| back_pain | 27 | 24.2 | 10.1 |
| headache | 30 | 22.9 | 1.6 |
| menstrual_pain | 10 | 22.0 | 2.2 |
| no_pain | 15 | 25.1 | 7.2 |

Headache subjects are notably homogeneous in age; back_pain and no_pain are much
more spread. Worth a sentence, probably not worth modelling around.

---

## 8. Participant-count discrepancies

Three numbers circulate; they don't reconcile:

- **99** — Kaggle/Mendeley description, and the survey row count
- **93** — "usable" figure in our proposal, from the source paper
- **83** — subject CSVs actually present in `RAW EEG DATA (1Hz)/All/` (**what we use**)
- **86** — unique subjects in `PROCESSED WATCH DATA/`

The 83/86 mismatch means the fusion model can only use the intersection. Quote
**83** for the EEG-only results and state the intersection size for fusion.

---

## 9. Prior work on this dataset

[`Nafiz2310/EEG-PainCategorization-PhysioPain`](https://github.com/Nafiz2310/EEG-PainCategorization-PhysioPain)
— the only public implementation we found. Relevant to the "justification for
reusing existing code" section, and it needs care:

- **No licence file.** Default copyright is all-rights-reserved. Cite it, read
  it, do **not** copy code into a graded deliverable (Turnitin).
- **Two files only** — `README.md` and `obj1_pipeline.py`. The `src/`, `data/`,
  `results/`, `notebooks/`, `docs/` folders its README advertises don't exist,
  and neither does the `requirements.txt` its install instructions reference.
- **No reported metrics.** No results directory, no tables. Its description
  claims SHAP interpretability; there is no `shap` import anywhere in the source.
- Its example input path (`.../All/compiled_dataset.csv`) **does not exist** in
  the archive — you'd have to build it by concatenating the 83 per-subject files.
- **Its "GroupKFold" evaluates a single fold**: the code is
  `train_idx, test_idx = list(gkf.split(...))[0]`. It does not cross-validate.

It remains a useful *architectural* reference (window/stride choices,
hierarchical gating, four baseline families, and it's TensorFlow/Keras like us).

**Our defensible differentiation:** real 5-fold `StratifiedGroupKFold` instead of
one fold, the §4(a) de-duplication they don't perform, the sex-confound
quantification, and extension from EEG-only to multimodal fusion.

---

## 10. Quick reference

```python
from painnet import config, data, windows, splits

config.describe()          # where am I, is the data present
df    = data.load_raw_eeg()      # 101,356 rows, warts handled
subj  = data.subject_table()     # 83 rows — the real unit of analysis
X, y, groups = windows.build_dataset()   # (6465, 60, 10)

for tr, te in splits.subject_folds(groups, windows.encode_labels(y)):
    ...   # raises LeakageError if a subject ever lands on both sides
```

| number | value |
|---|---|
| subjects | 83 |
| windows @ 60 s / 15 s | 6,465 |
| features per timestep | 10 (8 bands + Attention + Meditation) |
| majority baseline | 0.361 |
| test subjects per fold | ~17 |
