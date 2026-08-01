# Working on this project

Three of us, two weeks, one shared repo. These conventions exist mostly to stop
us stepping on each other.

> **Branching is provisional** — Raj is leading this phase, so if he prefers
> something else, his call wins and we'll edit this file.

---

## The one rule that isn't negotiable

**Never split the data randomly. Always group by subject.**

Every participant has one pain label for their whole recording, so a random
window-level split lets the model score by recognising the person rather than
the pain. Use `painnet.splits.subject_folds()`, which asserts disjointness on
every fold and raises `LeakageError` if not.

If you find yourself typing `train_test_split`, stop.

```bash
pytest    # runs in ~2s. do this before every push.
```

---

## Branching

```bash
git checkout -b feat/jonathan-data-loader
# ...work...
git push -u origin feat/jonathan-data-loader
# open a PR, tag the other two
```

- One branch per person per piece of work: `feat/<name>-<thing>`
- `main` should always run. If you break it, fix it or revert.
- Notebooks: **one owner each.** Don't edit someone else's `.ipynb` — the JSON
  doesn't merge (a single matplotlib PNG is a 50 KB base64 blob on one line).
  If you need something from another notebook, factor it into `src/painnet/`.

## Where code goes

| Kind of thing | Where |
|---|---|
| Anything reusable, or anything with a correctness risk | `src/painnet/` |
| Exploration, narrative, figures | `notebooks/` |
| One-off utilities | `scripts/` |

Rule of thumb: if getting it wrong would silently corrupt results, it belongs in
`src/` with a test.

---

## Kaggle setup

The dataset is already hosted on Kaggle, so this is the low-friction path — no
download, free GPU, nothing to install locally.

1. Sign in, then **phone-verify** your account (Settings → Phone Verification).
   GPU access needs it and it sometimes takes a while — do it early.
2. New Notebook → **Add Input** → `orvile/physiopain-dataset`
3. Settings → Accelerator → **GPU T4 x2** (not P100 — Kaggle's own docs warn the
   P100 image has issues)
4. Add the repo token so `pip install` can reach our private repo:
   **Add-ons → Secrets → New secret**, name it `GH_TOKEN`, paste a GitHub
   personal access token with `repo` scope
   ([create one](https://github.com/settings/tokens)).

Then your first cell:

```python
from kaggle_secrets import UserSecretsClient
tok = UserSecretsClient().get_secret("GH_TOKEN")
!pip install -q git+https://{tok}@github.com/RM-28/multimodal-pain-assesment.git

from painnet import config
print(config.describe())
```

**Don't paste the token into a cell directly** — Kaggle notebooks can be shared,
and the token would go with them.

### Kaggle gotchas

- Sessions die after 12 h, and idle sessions after 20–60 min. Save often.
- **No real-time co-editing.** Two people in one notebook will clobber each
  other. One notebook, one owner.
- GPU quota is ~30 h/week per account. Use *Quick Save* for routine saves;
  reserve *Save & Run All* for producing the final deliverable.

---

## Producing the deliverable

The course wants a notebook **with output cells**, a saved model, slides, a
report, and a `readme.txt`, all in one zip.

1. `notebooks/99_final_deliverable.ipynb` is the graded notebook. Keep it clean
   and top-to-bottom runnable.
2. On Kaggle: **Save Version → Save & Run All**, then from the version viewer
   use **(…) → Download code**. Open the downloaded `.ipynb` and confirm
   `"outputs"` is non-empty before submitting.
3. Locally, the equivalent is:
   ```bash
   jupyter nbconvert --to notebook --inplace --execute --allow-errors \
       notebooks/99_final_deliverable.ipynb
   ```
4. The trained model goes in `models/final/` — that path is deliberately
   un-ignored in `.gitignore`, so `git add models/final/` works.

---

## Style

We're all learning this as we go, and the code should read that way — clear over
clever, comments where something is non-obvious rather than on every line. Two
things we do consistently:

- `keras.utils.set_random_seed(config.SEED)` before building **each** model, so
  comparisons are fair and reruns match what's written in the markdown. The
  builders in `models.py` already do this.
- `verbose=2` in `model.fit` — one tidy line per epoch instead of progress bars
  that render as garbage in a PDF.

TensorFlow/Keras throughout. No PyTorch — it's not what the course uses.
