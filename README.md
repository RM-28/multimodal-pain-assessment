# Multimodal Pain Assessment

Deep learning for multimodal pain assessment from EEG and wristband physiological
signals.

**PSU AI 570 §002 — Deep Learning, Summer 2026**
Team: Jonathan Miller · Meng Li · Raj Mamidala

---

## What this is

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

Majority-class baseline is **0.361**. Anything near-perfect is a bug, not a result.

---

## Quick start

### On Kaggle (recommended — no setup, free GPU, dataset pre-mounted)

1. New Notebook → **Add Input** → search `physiopain` → add `orvile/physiopain-dataset`
2. Settings → Accelerator → **GPU T4 x2**
3. First cell:

```python
!pip install -q git+https://$GH_TOKEN@github.com/RM-28/multimodal-pain-assesment.git
from painnet import config, data, windows, splits, models, evaluate, plots
print(config.describe())
```

For the private-repo token, see [`CONTRIBUTING.md`](CONTRIBUTING.md#kaggle-setup).

### Locally

```bash
git clone https://github.com/RM-28/multimodal-pain-assesment.git
cd multimodal-pain-assesment

conda create -n painnet python=3.12 -y && conda activate painnet
pip install -e ".[local,dev]"

python scripts/get_data.py     # 1.26 GB download, ~19 MB kept
pytest                          # leakage guard — should be all green
```

`scripts/get_data.py --watch` additionally extracts the 4.9 GB wristband tier,
needed only for the fusion model.

---

## Layout

```
src/painnet/        the actual code — notebooks import this
  config.py         paths & constants; auto-detects Kaggle vs local
  data.py           loaders, with the dataset's known warts handled
  windows.py        per-subject z-scoring + sliding windows
  splits.py         subject-wise CV and the leakage guard  ← read this one
  models.py         cnn1d / cnn_lstm / tcn / fusion builders
  evaluate.py       macro-F1, per-class, confusion, QWK
  plots.py          report figures, styled consistently

notebooks/          thin — one owner each, to avoid merge conflicts
scripts/get_data.py local dataset fetch
tests/              the leakage guard. run it often.
docs/               dataset notes, research bibliography, deliverables
```

---

## How to run the pipeline

```python
from painnet import windows, splits, models, evaluate

X, y, groups = windows.build_dataset()        # (6465, 60, 10)
y_int = windows.encode_labels(y)

rows = []
for train_idx, test_idx in splits.subject_folds(groups, y_int):
    model = models.compile_model(models.build_cnn1d(X.shape[1:]))
    model.fit(X[train_idx], y_int[train_idx],
              validation_split=0.15, epochs=60, batch_size=64, verbose=2,
              class_weight=windows.class_weights(y_int[train_idx]),
              callbacks=models.default_callbacks())
    pred = model.predict(X[test_idx], verbose=0).argmax(1)
    rows.append(evaluate.fold_metrics(y_int[test_idx], pred))

print(evaluate.aggregate(rows))               # mean ± std across folds
```

Always report mean ± std across folds. With ~17 test subjects per fold, a single
split is noise.

---

## Data source & credits

- **Dataset:** PhysioPain, Istanbul Kültür Üniversitesi. CC BY 4.0.
  Kaggle: `orvile/physiopain-dataset` · Mendeley DOI `10.17632/mf2cgph9cy.4`
- **Prior art:** [`Nafiz2310/EEG-PainCategorization-PhysioPain`](https://github.com/Nafiz2310/EEG-PainCategorization-PhysioPain)
  — EEG-only pipeline on the same data. **No licence file**, so we treat it as
  reference reading only: architectures here are reimplemented from the
  described approach, not copied. See `docs/dataset-notes.md` for how our
  methodology differs (real 5-fold CV vs. their single fold, plus the
  de-duplication they don't do).

Full reading list with notes: [`docs/research/README.md`](docs/research/README.md).
