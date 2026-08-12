# Slide 10 "EEG model" — content and voiceover

The deck was restructured, so the earlier six-slide script no longer matches it.
Slides 6, 7, 8, 11, 12 and 13 are already narrated by Meng and Raj. The slides
still missing audio are **9, 10, 14, 17, 18**, and slide 10 is empty.

This file covers **slide 10** (mine), plus short optional scripts for **9** and
**14**. Total about 3 minutes.

---

## Slide 10 — content to put on the slide

**Title:** EEG model

**Left column — Model selection and architecture**

- Input (60, 10): 60 s windows at 1 Hz — 8 band powers + attention + meditation
- Four candidates, all deliberately small (83 people overfits anything larger)
  - 1D CNN — 14.5k params
  - CNN-LSTM — 28.7k
  - **TCN** — 71k, dilated convolutions span the full window
  - LightGBM control — 40 summary stats per window, no time series
- Per-subject normalization, class-weighted loss, early stopping

**Right column — Results (EEG only, 83 subjects, 5 subject-wise folds)**

| model | macro-F1 |
|---|---|
| TCN | **0.240 ± 0.046** |
| LightGBM | 0.235 ± 0.098 |
| 1D CNN | 0.221 ± 0.064 |
| CNN-LSTM | 0.199 ± 0.040 |
| majority floor | 0.132 ± 0.003 |

- Only the TCN clears chance on balanced accuracy (0.279 vs 0.250, p = 0.029)
- **Figure:** `figures/eeg_baselines_folds.png`

> Note the 83 here vs the 77 on slide 7: slide 7 shows the paired subjects used
> for fusion. EEG-only modelling uses all 83 with usable EEG. The script says
> this out loud so the deck doesn't look inconsistent.

---

## Slide 10 — voiceover  (~95 s)

This is the EEG branch.

[PAUSE]

The input is a sixty-second window at one hertz. Ten channels: eight frequency
band powers, plus the headset's own attention and meditation scores.

One note on numbers. The table you saw earlier had seventy-seven people, because
that's how many wore both devices. For EEG on its own we can use everyone with
a usable recording, which is eighty-three.

[PAUSE]

We tried four models. A small one-dimensional CNN. A CNN-LSTM. A temporal
convolutional network, which sees the whole window at once through dilated
convolutions. And a gradient-boosted tree as a control, which gets only forty
summary numbers per window and no time series at all.

Everything here is deliberately tiny, tens of thousands of parameters. With
eighty-three people, a larger network just memorises individuals.

[FIG: eeg_baselines_folds.png]

[PAUSE]

The TCN came out best, at point two four zero macro F one, against a floor of
point one three two. That floor is what you get by always guessing "headache".

But look at the dots rather than the bars. Each dot is one fold. The tree's
folds run from point one one to point three five. Same model, same data. It
looks either useless or like our best model, depending which seventeen people
you tested on.

So the TCN's real advantage isn't a higher average. Its folds cluster about
twice as tightly.

[PAUSE]

And only the TCN beats chance on balanced accuracy, at point two seven nine
against a quarter.

For context, published work with proper multi-channel research EEG, on an easier
task within a single person, reports an AUC around point eight three. We have
one dry electrode, four categories, and strangers in the test set.

---

## Slide 9 — divider, optional  (~20 s)

Only if nobody else is covering it. It's an agenda slide.

> We approached this in three stages. First, each modality on its own: what can
> the EEG do, and what can the wristband do. Then we combined them in a
> dual-branch model, to see whether the two together beat either one alone.
> Same data and the same subject-wise splits throughout, so the three are
> directly comparable.

---

## Slide 14 — lessons, optional  (~60 s)

The slide currently says "label can leak from other labels" and "more windows is
not more data". Both are ours. The sex-confound result is missing and belongs
here — it's the strongest finding we have. Check with Meng before recording, in
case she's already covering this slide.

> Two lessons stand out.
>
> First, more windows is not more data. We had a hundred thousand rows and
> sixty-five hundred windows, but only eighty-three people, and each person has
> a single label. So the number that governs everything is eighty-three. Split
> it any other way and the model just recognises individuals.
>
> Second, and this is the one I'd flag hardest. All ten of our menstrual-pain
> participants are women. So we asked whether these same models could predict
> sex instead of pain. From the same EEG windows they read sex at point seven
> two balanced accuracy, and pain at point three one. The model has a far more
> reliable route into that category than pain itself. We can't fix that with ten
> people. We can measure it, and say so.

---

## Corrections to flag to the team

Three things in the current deck worth fixing before submission.

1. **Slide 6 says "93 with usable recordings after quality control."** That
   figure is from the source paper and matches none of the tiers we used. The
   real counts are 83 with usable EEG, 86 with usable wristband, 77 with both.
   Slide 7's table already uses 77. Suggested fix: "99 participants, 83 with
   usable EEG and 86 with wristband; 77 have both."
2. **Slide 6: "had a 20-minute survey and then took a survey."** The first one
   should be the recording session.
3. **Slide 13's fusion numbers are correct** and match an independent rerun on a
   second machine (`verification-notes.md`). No change needed; worth knowing
   they're confirmed if anyone asks.
