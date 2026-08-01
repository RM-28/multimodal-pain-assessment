# data/

**Nothing in `raw/` or `processed/` is tracked in git** — see `.gitignore`. This
file is the tracked part.

## Getting the data

### Kaggle (nothing to do)

Attach `orvile/physiopain-dataset` to the notebook. It mounts read-only at
`/kaggle/input/physiopain-dataset/`. `painnet.config` finds it automatically.

### Locally

```bash
python scripts/get_data.py            # 1.26 GB download, ~19 MB kept
python scripts/get_data.py --watch    # + the 4.9 GB wristband tier
```

No Kaggle credentials required — the dataset is served unauthenticated.

If you'd rather do it by hand:

```bash
curl -L -o data/raw/physiopain.zip \
  "https://www.kaggle.com/api/v1/datasets/download/orvile/physiopain-dataset"
cd data/raw && unzip -q physiopain.zip \
  "PhysioPain Dataset/Multimodal Pain Dataset/RAW EEG DATA (1Hz)/*" \
  "PhysioPain Dataset/Multimodal Pain Dataset/SURVEY DATA/*"
```

## What lands where

```
data/
├── raw/
│   └── PhysioPain Dataset/Multimodal Pain Dataset/
│       ├── RAW EEG DATA (1Hz)/All/     83 CSVs, 19 MB   ← the modelling data
│       ├── SURVEY DATA/                2 xlsx           ← intensity + demographics
│       └── PROCESSED WATCH DATA/       4.9 GB           ← only with --watch
└── processed/                          cached .npz windows (regenerable)
```

## Verify you got it

```bash
pytest -q            # the two @needs_data tests should run, not skip
python -c "from painnet import config; print(config.describe())"
```

Expect **83** subject CSVs. If you see a different number, something extracted
wrong.

## Sizes, so you can plan

| | |
|---|---|
| zip download | 1.26 GB |
| full extract | 5.36 GB |
| EEG + survey only | **19 MB** |
| wristband tier | 4.87 GB |

## Licence

CC BY 4.0 — Istanbul Kültür Üniversitesi. Mendeley DOI `10.17632/mf2cgph9cy.4`.
Attribution is required; the citation is in the root README.

⚠️ Read [`../docs/dataset-notes.md`](../docs/dataset-notes.md) before modelling.
The processed EEG tier has duplicate subjects under conflicting labels, and the
processed watch files have sample-rate values leaked into row 0.
