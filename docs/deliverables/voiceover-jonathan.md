# Voiceover script: Jonathan, slides 7-12

Read near-verbatim. Written for the ear: short sentences, one idea per breath.
About **6 and a half minutes** across six slides.

**Before you record**

- Pace: roughly 140 words a minute. Slow down on numbers, nothing else.
- `[PAUSE]` is a real beat, about half a second. `[FIG]` means that figure
  should be on screen.
- Say numbers aloud the natural way: "point two four zero", "F one".
- Two to keep clean: **Empatica** (em-PAT-ih-ka), **PhysioPain** (FIZZ-ee-oh-pain).
- Fluff a line? Stop, breathe, retake that sentence. Don't restart the slide.

---

## Slide 7 — Data Collecting  (~65 s)

Our data is PhysioPain, collected at Istanbul Kultur University and released
publicly under a Creative Commons licence.

[PAUSE]

Ninety-nine people, one twenty-minute session each, wearing two consumer
devices. A NeuroSky headband, which is a single dry electrode on the forehead
reporting eight EEG band powers per second. And an Empatica E four wristband:
pulse, skin conductance, temperature, motion.

The label is the pain each person reported that day. Headache, back pain,
menstrual pain, or none.

[PAUSE]

Two things here shaped everything that follows.

First, this is about the cheapest brain signal money can buy. One electrode. No
spatial information.

Second, each participant has one label for their whole recording. We checked all
eighty-three usable files. No exceptions.

[FIG: subjects_per_class.png]

And the class balance, counted in people rather than rows. Thirty headache,
twenty-eight back pain, fifteen no pain, and ten with menstrual pain. That ten
matters later.

---

## Slide 8 — Data Preprocessing  (~70 s)

Before writing any model code, we audited the archive itself. Best time we spent
on this project.

[PAUSE]

The serious find first. The dataset ships a processed EEG folder that looks
convenient. But anyone with more than one pain condition appears in two or three
label folders, with identical signals. One person appears three times. About a
quarter of the rows carry more than one label.

Train on that, and identical inputs sit on both sides of your split with
conflicting answers.

We don't use it. We use the raw one-hertz files: eighty-three files,
eighty-three people, one label each.

[PAUSE]

Second find. The Empatica writes its sample rate on line two of every file, and
the conversion script kept it. The first data row of every wristband file claims
the skin temperature was four degrees. Our loader drops it.

The real preprocessing is dull on purpose. We standardise each channel within
each person. Raw band powers vary by a factor of ten thousand, and mostly they
tell you how well the headset was sitting. Then sixty-second windows, every
fifteen seconds.

---

## Slide 9 — Methodology  (~75 s)

This is the part that matters most.

[PAUSE]

Every participant has one label. Split your data randomly and the model doesn't
learn pain. It learns to recognise people. Our windows overlap, so near-identical
slices of one recording land in training and testing at once. You get a
beautiful number that means nothing.

[PAUSE]

Every split we do is grouped by person instead. All of someone's windows go to
training, or all go to testing. Never both. We stratify too, so the ten
menstrual-pain participants spread across folds rather than landing in one.

That isn't left to good intentions. The splitting function checks it on every
fold and throws an error if anyone appears on both sides. It has its own tests.
On data this small, the split is the result.

[PAUSE]

One correction to our own work. What's the number to beat? A model that always
guesses "headache" gets thirty-six percent accuracy, and we quoted that as our
bar for about a week.

But its macro F one is point one three two. It scores zero on three of the four
classes.

Macro F one is the honest metric when classes are this unbalanced. Point one
three two is the real floor. Measure your baseline, don't derive it in your head.

---

## Slide 10 — Model selection and parameters  (~65 s)

Four architectures, plus one deliberate control.

[PAUSE]

A small one-dimensional convolutional network as the baseline. A CNN-LSTM, which
hands those features to a recurrent layer. A temporal convolutional network,
which sees the whole sixty-second window at once without recurrence.

Then the fusion model, which is really why this is a deep learning project. Our
two devices sample at different rates. One hertz, four hertz. Rather than
resample one and pretend they match, each modality gets its own encoder branch,
and we join the learned representations before the classifier.

[PAUSE]

Every model here is deliberately tiny. Tens of thousands of parameters, not
millions. With eighty-three people, a large network just memorises individuals.

Same thinking behind the control. We ran a gradient-boosted tree that sees only
forty summary numbers per window. No time series at all. If that keeps up with
the neural networks, it's telling us something.

It keeps up.

---

## Slide 11 — Training, validation, testing  (~55 s)

Training is boring on purpose.

[PAUSE]

Same treatment for every model. We reset the random seed before building each
one, so the comparison is fair. The loss is weighted by class frequency, so rare
categories can't be ignored. We stop early when validation loss stops improving.

Everything converges in nine to seventeen epochs. The whole EEG suite, five folds
across four models, runs in under four minutes on a laptop.

[PAUSE]

For fusion, Meng went further with a nested split. Inside each fold's training
set she carved out a separate validation group of people. Training, validation
and testing are three separate sets of participants. Early stopping never sees a
test subject, even indirectly.

Reproducibility is handled the boring way too. Fixed seeds, a pinned class order,
and every results table committed next to the code that made it.

---

## Slide 12 — Evaluation and performance  (~90 s)

Here are the honest numbers.

[FIG: eeg_baselines_folds.png]

On EEG alone the TCN comes out best, at point two four zero macro F one, against
that floor of point one three two. All three neural networks clear the floor
with statistical significance.

[PAUSE]

But don't look at the bars. Look at the dots. Each dot is one fold.

The tree's five folds run from point one one to point three five. Same model,
same data. It looks either useless or like our best model, depending entirely on
which seventeen people you tested it on.

The TCN's real advantage isn't a higher average. Its folds cluster about twice as
tightly.

[FIG: fold4_reversal.png]

[PAUSE]

And this is the most instructive result we produced. On fold four the CNN records
its worst score of the project, while the tree records its best. Identical test
subjects.

Evaluate on one split, and you could conclude that trees beat deep learning by a
factor of three. Or the exact opposite. Both write-ups would look rigorous.

That's why every number we report is a mean and a standard deviation across
folds.

[PAUSE]

For context: published work using proper multi-channel research EEG, on an easier
yes-or-no task within a single person, reports an AUC around point eight three.
We have one dry electrode, four categories, and strangers in the test set. Our
numbers reflect that.

The contribution isn't the accuracy. It's knowing what the accuracy is worth.

---

## Optional 20-second add-on

Only if you're covering the sex-confound result yourself rather than leaving it
to slide 13 or 14. Check with Meng so it isn't said twice.

> One last finding, and it's the one I'd flag hardest. All ten of our
> menstrual-pain participants are women. So we asked whether these same models
> can predict sex. From the same EEG windows they read sex at point seven two
> balanced accuracy, and pain at point three one. The model has a far more
> reliable route to that category than pain itself. We can't fix that with ten
> people. We can measure it, and say so.
