Multimodal Pain Assessment

Deep learning for multimodal pain assessment from EEG and wristband physiological
signals.
PSU AI 570 S002 — Deep Learning, Summer 2026
Team: Jonathan Miller · Meng Li · Raj Mamidala

---

What this is

We classify **pain type** (no pain / headache / back pain / menstrual pain) from
the [PhysioPain dataset](https://www.kaggle.com/datasets/orvile/physiopain-dataset)
— single-channel EEG band powers at 1 Hz plus Empatica E4 wristband signals
(BVP, EDA, temperature, accelerometer).

The centrepiece is a **two-branch late-fusion network** built with the Keras
Functional API: one encoder per modality, concatenated embeddings, shared
classifier head. Single-modality variants act as ablations so we can measure
what each sensor actually contributes.

**The most important thing about this dataset:** every participant has exactly
one pain label for their entire recording. So the effective sample size is
**83 subjects**, not the ~101,000 rows or ~6,500 windows it superficially looks
like. All cross-validation is grouped by subject. Read
[`docs/dataset-notes.md`](docs/dataset-notes.md) before writing any modelling
code — it documents several traps that will silently inflate your accuracy.

Majority-class baseline is 0.132 macro-F1. Anything near-perfect is a bug, not a result.

---
How to run:

    Python 3.10+ with TensorFlow 2.16+. From the project root:

    1. conda create -n painnet python=3.12 -y && conda activate painnet
    2. pip install -e ".[local,dev]"
    3. python scripts/get_data.py --watch
    4. pytest
    5. jupyter notebook notebooks/99_final_deliverable.ipynb

---
Standalone experiment scripts, results are written to results/*.csv:
    1. python scripts/run_baselines.py --models majority lightgbm cnn1d cnn_lstm tcn --epochs 60 --batch-size 64 --tag baselines
    2. python scripts/run_leakage_demo.py
    3. python scripts/run_fusion_repro.py
    4. python scripts/run_sex_probe.py

Trained model: models/final/tcn_final.keras, loads with keras.models.load_model

---
Dataset: PhysioPain, Istanbul Kültür Üniversitesi. CC BY 4.0.
  Kaggle: `orvile/physiopain-dataset` · Mendeley DOI `10.17632/mf2cgph9cy.4`

Prior art: [`Nafiz2310/EEG-PainCategorization-PhysioPain`](https://github.com/Nafiz2310/EEG-PainCategorization-PhysioPain)
  — EEG-only pipeline on the same data. No licence file, so we treat it as
  reference reading only: architectures here are reimplemented from the
  described approach, not copied. See `docs/dataset-notes.md` for how our
  methodology differs (real 5-fold CV vs. their single fold, plus the
  de-duplication they don't do).
