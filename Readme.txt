Deep Learning-Based Pain Assessment Using Multimodal Physiological Data
AI 570, Deep Learning, Summer 2026
Team: Jonathan Miller, Meng Li, Raj Mamidala

ABSTRACT

We classify self-reported pain type (none, headache, back pain, menstrual
pain) from consumer wearables: a single-electrode NeuroSky EEG headband and
an Empatica E4 wristband, using the public PhysioPain dataset (83 usable
participants). We compare a 1D CNN, a CNN-LSTM, a temporal convolutional
network, a dual-branch fusion model built with the Keras Functional API, and
a LightGBM control. Because every participant carries one label, all
evaluation is grouped by participant; a controlled experiment shows a random
window split inflates macro-F1 by 3.5x (0.899 vs 0.258). Under subject-wise
5-fold cross-validation the TCN performs best on EEG alone (macro-F1
0.240 +/- 0.046 against a 0.132 floor), the wristband alone beats the fusion
model, and the same pipeline decodes participant sex far more reliably than
pain, which matters because all menstrual-pain participants are female. The
contribution is a leakage-controlled benchmark on this dataset rather than a
headline accuracy.

HOW TO RUN

Requires Python 3.10+ with TensorFlow 2.16+. From the project root:

    pip install -e ".[local]"
    python scripts/get_data.py          downloads the dataset (1.26 GB, no
                                        Kaggle account needed), keeps ~19 MB
    pytest                              the split-leakage guard, ~2 s
    jupyter notebook notebooks/99_final_deliverable.ipynb

The notebook runs end to end on a laptop CPU in about 10 minutes. The other
notebooks (01_eda, data, watch, fusion) are the exploratory and per-modality
work they are stitched from. Wristband and fusion experiments need the full
extract: python scripts/get_data.py --watch (4.9 GB).

Scripts, runnable standalone:
    scripts/run_baselines.py            EEG models, 5-fold CV, ~4 min
    scripts/run_leakage_demo.py         the leaked vs grouped split experiment
    scripts/run_fusion_repro.py         the modality comparison
    scripts/run_sex_probe.py            the confound measurement

The trained model is models/final/tcn_final.keras and loads with
keras.models.load_model. Per-fold results tables are in results/*.csv.

DATASET

PhysioPain, Istanbul Kultur Universitesi, CC BY 4.0. The dataset zip in this
submission contains the EEG tier and the survey (the ~19 MB actually used by
the main notebook), so the EEG experiments re-run with no download. The full
archive (1.26 GB, needed only for the wristband and fusion experiments) is
fetched by the script above, or directly:
https://www.kaggle.com/datasets/orvile/physiopain-dataset
Mendeley DOI: 10.17632/mf2cgph9cy.4
