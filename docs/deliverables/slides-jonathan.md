# Slide drafts: Jonathan's sections (slides 7-12)

Drafts for the stub slides covering the dataset, pipeline, and EEG baselines.
Each slide has on-slide bullets plus speaker notes sized for 75-90 seconds,
which keeps my six slides near 9 minutes of the 25.

Figures referenced here live in `docs/deliverables/figures/`. Numbers come from
`results/baselines_folds.csv` (EEG-only, 83 subjects) and, where marked, from
Meng's `notebooks/fusion.ipynb` (paired data, 77 subjects). Keep the two
datasets out of the same table; the subject pools differ.

---

## Slide 7 — Data Collecting

**On slide:**

- PhysioPain (Istanbul Kultur University, 2025). Public on Kaggle, CC BY 4.0
- Two consumer wearables, one 20-minute session per participant:
  - NeuroSky MindWave Mobile 2: single-channel EEG, 8 band powers at 1 Hz
  - Empatica E4 wristband: BVP, EDA, skin temperature, 3-axis accelerometer
- Labels: pain type reported by the participant. One label per person
- 99 recruited. Usable: 83 (EEG), 86 (wristband), 77 (both)
- Figure: `subjects_per_class.png` (30 headache / 28 back / 15 none / 10 menstrual)

**Speaker notes:**

We used PhysioPain, a public multimodal dataset collected at Istanbul Kultur
University. Each participant wore a consumer EEG headband and an Empatica
wristband for about twenty minutes while experiencing whatever pain state they
reported that day. So the label is the participant's own report of pain type:
headache, back pain, menstrual pain, or no pain.

Two facts about this dataset drive every design decision you'll see. First, the
EEG is a single dry electrode reporting band powers once per second. There is no
spatial information; this is the cheapest possible brain signal. Second, each
participant has exactly one label for their entire recording. We verified that
across all 83 usable EEG files: zero exceptions. Keep that in mind, because it
means the participant, not the row, is our unit of analysis.

The paper reports 99 participants, but the tiers do not agree: 83 have usable
EEG, 86 have usable wristband data, and 77 have both. We quote whichever count
matches the experiment being shown.

---

## Slide 8 — Data Preprocessing

**On slide:**

- Audit before code: we read the archive and found three defects
  - Processed EEG tier lists comorbid subjects under 2-3 labels with identical
    signals (26.5% of rows). We use the raw disjoint tier instead
  - Empatica sample-rate row leaked into row 0 of every processed watch file
  - `Derived` column 100% empty; `totPwr` exactly equals the sum of the bands
- Per-subject z-scoring (band powers span 4 orders of magnitude)
- Sliding windows: 60 s, 15 s stride. 6,465 EEG windows; wristband windows at
  4 Hz paired on the 77 common subjects

**Speaker notes:**

Before writing any model code we audited the archive itself, and that audit paid
for itself twice over. The dataset ships a processed EEG tier that looks
convenient, but participants with more than one pain condition appear in two or
three label folders with byte-identical signals. About a quarter of the rows
carry more than one label. Train on that naively and identical inputs sit on
both sides of your split with conflicting targets. We use the raw one-hertz
tier, which is disjoint: 83 files, 83 people, one label each.

Second find: the Empatica's file format puts the sample rate on line two, and
the dataset authors' conversion script kept it, so the first data row of every
processed wristband file says the EDA was 4.0 and the temperature was 4.0
degrees. Our loader drops that row.

The real preprocessing is deliberately simple. We z-score every channel within
each participant, because raw band powers vary by four orders of magnitude and
mostly encode how well the headset was seated. Then we cut sliding windows,
sixty seconds long every fifteen seconds. For fusion, windows from the two
devices are paired on the 77 participants that both tiers share.

---

## Slide 9 — Methodology

**On slide:**

- One label per participant, so a random window split is leakage by construction
- Effective sample size: 83 subjects, not 101k rows or 6.5k windows
- StratifiedGroupKFold, 5 folds, grouped on subject; every fold keeps 2
  menstrual-pain subjects
- The split guard is code, not policy: any subject on both sides raises an error
- Measured floor (trivial always-headache model): accuracy 0.360 but
  macro-F1 0.132. Macro-F1 is our headline metric

**Speaker notes:**

Here is the methodological core of the project. Because every participant
carries one label, any window-level random split lets the model recognize the
person instead of the pain, and near-duplicate overlapping windows from the same
recording land in both train and test. Scores come out looking wonderful and
mean nothing. So all cross-validation is grouped by subject, stratified so the
ten menstrual-pain participants spread across folds instead of landing in one.

We did not leave this as a convention. The split function asserts that train and
test subjects are disjoint on every fold and throws an exception otherwise, and
that guard has a test suite. On a dataset this small, the split is the result.

One correction we caught in our own work: the number to beat. The trivial model
that always predicts headache gets 36 percent accuracy, and we quoted that as
the bar for a while. But its macro-F1 is 0.132, because it scores zero on three
of four classes. Macro-F1 is the honest headline given the class imbalance, so
that is the floor everything must clear.

---

## Slide 10 — Model selection and parameters

**On slide:**

- Four families, capacity kept small on purpose (83 subjects):
  - 1D CNN, 14.5k params. Local waveform shape
  - CNN-LSTM, 28.7k. Conv features into recurrence
  - TCN, 71k. Dilated causal convs (1/2/4/8) cover the whole 60 s window
  - Dual-branch fusion, 26k. Separate EEG (1 Hz) and wristband (4 Hz) encoders,
    concatenated. Keras Functional API
- Non-deep control: LightGBM on per-window mean/std/min/max
- Adam 1e-3, batch norm, dropout 0.2-0.4, class-weighted loss
- No data augmentation in the final runs; class weighting handles imbalance

**Speaker notes:**

We evaluated four architectures plus one deliberate non-deep-learning control.
The baseline is a small one-dimensional CNN. The CNN-LSTM hands convolutional
features to a recurrent layer. The temporal convolutional network uses dilated
causal convolutions; with dilations one through eight the receptive field spans
essentially the entire sixty-second window without recurrence. The fusion model
is the reason this is a Functional API project: the two devices sample at
different rates, so each modality gets its own encoder branch and the learned
embeddings concatenate into a shared classifier head. No resampling, no
pretending the signals are one tensor.

Every model is small, tens of thousands of parameters, because with 83 people a
large network just memorizes individuals. Same reasoning behind the LightGBM
control: it sees only forty summary statistics per window, and if it keeps up
with the deep models, that tells us something real about how much the temporal
structure is worth at this sample size. Spoiler: it keeps up.

Our proposal mentioned augmentation; in the final runs we relied on class
weighting instead, and we state that plainly as a difference from the plan.

---

## Slide 11 — Training, validation, testing

**On slide:**

- Identical protocol per fold and per model: seed reset, class weights,
  early stopping on val loss (patience 8), ReduceLROnPlateau
- Models converge in 9-17 epochs. Full EEG suite: 3.6 min on a laptop CPU
- Fusion experiments (Meng): nested split, train/val/test all subject-disjoint,
  roughly 47/15/15 subjects per fold, asserted in the notebook
- Reproducible: fixed seeds, pinned class order, results checked into git

**Speaker notes:**

Training is deliberately boring. Every model gets the same protocol: the seed is
reset before each build so comparisons are fair, the loss is class-weighted,
and early stopping watches validation loss with patience of eight. Everything
converges in nine to seventeen epochs, and the entire EEG suite, five folds by
four models, runs in under four minutes on a laptop. Small data has few
consolations; this is one.

For the fusion experiments Meng tightened the protocol further with a nested
split: within each fold's training subjects she carved out a subject-disjoint
validation set, so early stopping never peeks at test subjects even indirectly.
Train, validation, and test are three disjoint sets of people, roughly 47, 15,
and 15 per fold, and the notebook asserts that disjointness at runtime.

Reproducibility is handled the boring way too: fixed seeds, a pinned class
order so confusion matrices always line up, and the per-fold results tables are
committed to the repository next to the code that made them.

---

## Slide 12 — Evaluation: baseline performance

**On slide:**

- Figure: `eeg_baselines_folds.png` (bars = mean, dots = the 5 folds)
- EEG-only, 83 subjects, macro-F1: TCN 0.240 +/- 0.046 best; LightGBM
  0.235 +/- 0.098 ties it with twice the spread; floor 0.132
- Wristband-only (Raj, 86 subjects): LightGBM 0.231
- Only the TCN is significantly above chance on balanced accuracy
  (0.279 vs 0.250, p = 0.029)
- Figure: `fold4_reversal.png`. Same test subjects, opposite conclusions:
  fold 4 is the CNN's worst fold and LightGBM's best

**Speaker notes:**

Here are the honest numbers. On EEG alone, the TCN leads at 0.240 macro-F1
against a floor of 0.132. All three neural networks clear the floor with
statistical significance. But look at the dots, not the bars. LightGBM's five
folds span 0.11 to 0.35; the same model, on the same data, looks either useless
or clearly best depending on which seventeen people you test it on. The TCN's
real advantage is not its mean, it is that its folds cluster twice as tightly.

The second figure is the single most instructive result in the project. On fold
four the CNN records its worst score while LightGBM records its best, on
identical test subjects. A paper that evaluated one split could have concluded
trees beat deep learning by a factor of three, or the exact opposite, and both
papers would have felt rigorous. This is why every number we report is a mean
and a standard deviation over subject-wise folds, and it is why we distrust the
much higher single-split accuracies you sometimes see in this literature.

For calibration: published work with multi-channel research EEG on an easier
within-subject binary task reports AUC around 0.83. One dry electrode, four
classes, and strangers in the test set is a much harder problem, and the numbers
reflect that. The contribution here is the evaluation discipline, not the score.

---

## Handoff notes for slides 13-14 (Meng and shared)

Not drafting these, they are Meng's and the group's, but the numbers that
belong there from our side:

- **Slide 13 (outcomes):** on the 77-subject paired data the ranking is
  wristband LightGBM 0.218, fusion CNN 0.202, EEG TCN 0.176. Fusion beat the
  EEG branch in 3 of 5 folds but beat the wristband in only 1 of 5. The
  proposal predicted fusion wins outright; it did not, and the wristband alone
  is the strongest single signal. Per-class strengths differ (fusion best on
  headache, wristband best on back pain, EEG best on menstrual pain), which is
  the interesting nuance. The ranking is confirmed by an independent rerun on
  a second machine (`docs/deliverables/verification-notes.md`): wristband >
  fusion > EEG holds there too, with means shifted within one std.
- **Slide 14 (lessons):** count subjects, not rows; measure your floor instead
  of deriving it (our 0.361 vs 0.132 mix-up); a single split on 83 people is
  noise (fold 4); audit the archive before modeling (row-0 leak, duplicated
  subjects); and the sex confound, now quantified: all 10 menstrual-pain
  participants are female, and the same pipeline that reads pain at 0.31
  balanced accuracy reads SEX at 0.72 from the same EEG windows. The model has
  a far more reliable path to the menstrual class than pain itself. Numbers in
  `verification-notes.md`; this is a result, not a caveat.
