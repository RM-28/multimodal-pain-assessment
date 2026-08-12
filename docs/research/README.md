# Research reading, annotated

The eight PDFs in this folder, with what each one actually gives us. Ordered by
how directly it bears on our design rather than by date.

Three of these (Thiam, Lu, Ding) were also the basis of Jonathan's individual
Scientific Reading & Review.

---

## Directly shapes our architecture

### Thiam et al. (2021), *Multi-Modal Pain Intensity Assessment Based on Physiological Signals: A Deep Learning Perspective*
`Thiam2021_MultiModal_Pain_DeepLearning.pdf` · Front. Physiol. 12:720464 · [doi:10.3389/fphys.2021.720464](https://doi.org/10.3389/fphys.2021.720464)

The closest published analogue to what we're building: deep networks over
multiple physiological channels for pain, with an explicit treatment of fusion
strategy. Argues the case for letting networks learn features rather than
hand-designing them, and is candid about the drawbacks (data hunger, opacity).

**Why it matters to us:** our justification for late fusion over hand-engineered
features, and a source for the early-vs-late fusion discussion in the report.

### Ding, Ma & Li (2025), *Multimodal physiological signal emotion recognition based on multi-head cross attention with representation learning*
`Ding2025_CrossAttention_Multimodal.pdf` · Front. Psychiatry 16:1713559 · [doi:10.3389/fpsyt.2025.1713559](https://doi.org/10.3389/fpsyt.2025.1713559) · CC BY

Emotion rather than pain, but structurally it is *exactly* our problem: a
**dual-branch architecture processing EEG and peripheral signals separately**,
then fusing. Their contribution is replacing plain concatenation with multi-head
cross-attention, on the argument that simple fusion "fails to capture the
complex, dynamic interactions between modalities."

**Why it matters to us:** this is the blueprint for our stretch goal. Our
`build_fusion` does concatenation-style late fusion; cross-attention is the
upgrade path if the baseline fusion works and time allows. Also the cleanest
citation for *why* two branches beats one resampled tensor.

⚠️ Their setup has far more subjects than our 83, attention modules overfit fast
at our scale. Treat as aspiration, not a default.

### Lu, Ozek & Kamarthi (2023), *PainAttnNet: Transformer Encoder with Multiscale Deep Learning for Pain Classification Using Physiological Signals*
`Lu2023_PainAttnNet.pdf` · arXiv:2303.06845v2 · Northeastern University

Multiscale convolutional feature extraction feeding a transformer encoder, for
pain *intensity* from physiological signals.

**Why it matters to us:** the multiscale-CNN idea is cheap to borrow and doesn't
require attention, parallel conv branches at different kernel widths, then
concatenate. A plausible improvement to `build_cnn1d` that stays within our data
budget. Also a reference point for how people frame pain intensity as a
classification rather than regression problem.

---

## EEG-specific evidence

### Chen et al. (2022), *Scalp EEG-Based Pain Detection Using Convolutional Neural Network*
`Scalp_EEG-Based_Pain_Detection_Using_Convolutional_Neural_Network.pdf` · IEEE TNSRE 30:274

CNN on EEG to separate induced pain from rest in **10 chronic back pain
patients**, under two induction protocols (movement, video). Reports
**AUC 0.83 ± 0.09** and **0.81 ± 0.15**.

**Why it matters to us:** the single most useful calibration point in the folder.
Multi-channel *scalp* EEG on a within-subject binary task, far easier than ours,
and it lands at ~0.83 AUC with a ±0.09 spread. We have one dry frontal electrode,
four classes, and subject-independent evaluation. **If we report near-perfect
numbers, this paper is the evidence that something is wrong.** Cite it when
justifying modest expectations.

### Al-Nafjan, Alshehri & Aldayel (2025), *Objective Pain Assessment Using Deep Learning Through EEG-Based Brain–Computer Interfaces*
`biology-14-00210.pdf` · Biology 14(2):210

Two-stage EEG system: pain/no-pain detection, then severity into low/moderate/
high. Reports **91.84%** detection and **87.94%** severity accuracy.

**Why it matters to us:** the two-stage structure (gate first, then subtype) is
worth considering, the reference implementation's `--hierarchical` flag does the
same thing, and it may suit our imbalance, since pain-vs-no-pain is 68 vs 15.
⚠️ Read the reported accuracies critically: check whether the evaluation is
subject-independent before quoting them as a target. Accuracy on an imbalanced
set is also not comparable to our macro-F1.

---

## Peripheral-signal evidence (the wristband branch)

### Pinzon-Arenas et al. (2023), *Design and Evaluation of Deep Learning Models for Continuous Acute Pain Detection Based on Phasic Electrodermal Activity*
`Design_and_Evaluation_of_Deep_Learning_Models_for_Continuous_Acute_Pain_Detection_Based_on_Phasic_Electrodermal_Activity.pdf` · IEEE JBHI 27(9):4250

Systematically compares **1D-CNN, LSTM, and three CNN–LSTM hybrids** on phasic
EDA from 36 volunteers under thermal-grill pain. The best model, a parallel TCN
plus stacked bi/uni-directional LSTM, reaches **F1 77.8%** and generalises to 37
independent subjects from BioVid at 91.5% accuracy.

**Why it matters to us:** the best single justification for our model ladder,
since it evaluates almost exactly the architectures we're building on a signal we
have. Also establishes EDA as the strongest single peripheral channel, which
should inform what the wristband branch prioritises. The phasic/tonic
decomposition is a preprocessing idea we could adopt for the E4's EDA.

### Phan et al. (2023), *Pain Recognition With Physiological Signals Using Multi-Level Context Information*
`Pain_Recognition_With_Physiological_Signals_Using_Multi-Level_Context_Information.pdf` · IEEE Access · [doi:10.1109/ACCESS.2023.3248654](https://doi.org/10.1109/ACCESS.2023.3248654)

Multi-level context vectors plus an attention module on BioVid Part A and
Emopain. Evaluated **leave-one-subject-out**, reporting 84.8 ± 13.3% (87 subjects)
and 87.8 ± 11.4% (67 subjects) on the hardest binary contrast.

**Why it matters to us:** methodologically the closest to our discipline, subject-independent evaluation, results as mean ± std. **Look at those standard
deviations: ±13.3 and ±11.4 percentage points.** That is what honest
subject-independent pain classification looks like, and it's the best argument in
the folder for why we report mean ± std across folds rather than a single number.

---

## Framing and background

### Cascella et al. (2023), *Artificial Intelligence for Automatic Pain Assessment: Research Methods and Perspectives*
`Pain Research and Management - 2023 - Cascella - ...pdf` · Pain Research and Management, Art. ID 6018736 · [doi:10.1155/2023/6018736](https://doi.org/10.1155/2023/6018736)

Review article covering the whole automatic-pain-assessment field, modalities
(facial, physiological, behavioural), methods, and open problems.

**Why it matters to us:** the natural backbone for the report's Introduction and
Related Works sections, and the right citation for the clinical motivation
(objective assessment for patients who cannot self-report). Use it to position
our contribution rather than for technical detail.

---

## Gaps we should be honest about

None of these papers uses PhysioPain. The only public work on this dataset is
[`Nafiz2310/EEG-PainCategorization-PhysioPain`](https://github.com/Nafiz2310/EEG-PainCategorization-PhysioPain),
which is unlicensed, reports no metrics, and evaluates a single fold: see
[`../dataset-notes.md` §9](../dataset-notes.md). So we have **no published
benchmark to compare against on this data**. That cuts both ways: nothing to beat,
but also nothing to sanity-check us. Hence the emphasis on the majority-class
baseline (0.361) and on the Chen and Phan papers as external calibration.

Also worth stating: every paper here classifies pain from *induced or clinically
observed* pain states. PhysioPain's labels are self-reported pain *type*, which
tracks the region a participant named as much as any distinct physiological
signature. That's a weaker target than these papers work with, and the report
should say so.

---

## Citing

All eight are journal or arXiv papers with DOIs above. The dataset itself is
CC BY 4.0 and must be cited: see the README's *Data source & credits*.
