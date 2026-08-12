# Voiceover script: Jonathan, slides 7-12

Read this out loud, near enough verbatim. It is written for the ear, not the
eye: short sentences, one idea at a time, contractions where you'd naturally
use them. Roughly **9 minutes** of the 25.

**Before you record**

- Pace: about 140 words a minute. Don't rush the numbers, rush nothing else.
- `[PAUSE]` means a real beat, half a second. It's where the listener catches up.
- `[FIG]` means the figure should be on screen by then.
- Say numbers the way you'd say them aloud: "point two four zero", not
  "zero point twenty-four". "F one", not "F-sub-one".
- Two words to keep clean: **Empatica** (em-PAT-ih-ka) and **PhysioPain**
  (FIZZ-ee-oh-pain).
- If you fluff a line, stop, breathe, and retake that sentence only. Don't
  restart the slide.

---

## Slide 7 — Data Collecting  (~80 s)

Our data comes from PhysioPain, a public dataset collected at Istanbul Kultur
University and released under a Creative Commons licence.

[PAUSE]

Ninety-nine people took part. Each one wore two consumer devices for a single
twenty-minute session. The first is a NeuroSky headband. That's a single dry
electrode on the forehead, reporting eight EEG frequency band powers once per
second. The second is an Empatica E four wristband, which gives us blood volume
pulse, skin conductance, skin temperature, and motion.

[PAUSE]

The label is whatever pain the participant reported that day. Headache, back
pain, menstrual pain, or none.

Now, two things about this dataset shaped every decision you'll see in the next
few slides.

First, the EEG is about the cheapest brain signal you can buy. One electrode, no
spatial information at all.

Second, and this matters more: each participant has exactly one label for their
entire recording. We checked all eighty-three usable files. There are no
exceptions.

[FIG: subjects_per_class.png]

And here's the class balance, counted in people rather than rows. Thirty
headache, twenty-eight back pain, fifteen no pain, and just ten with menstrual
pain. Hold onto that ten. It comes back.

---

## Slide 8 — Data Preprocessing  (~85 s)

Before writing a single line of model code, we audited the archive itself. That
turned out to be the best time we spent on this project.

[PAUSE]

We found three problems.

The first one is serious. The dataset ships a processed EEG folder that looks
convenient. But participants who have more than one pain condition appear in two
or three different label folders, with byte-identical signals. One person shows
up three times. About a quarter of the rows carry more than one label. If you
train on that, identical inputs end up on both sides of your split with
conflicting answers. So we don't use it. We use the raw one-hertz tier, which is
clean: eighty-three files, eighty-three people, one label each.

[PAUSE]

Second problem. The Empatica writes its sample rate on line two of every file,
and the conversion script kept it. So the first data row of every wristband file
claims the skin temperature was four degrees. Our loader drops that row.

Third, smaller. One column is completely empty, and another is just the sum of
the other eight. Both go.

The actual preprocessing is deliberately dull. We standardise every channel
within each person, because the raw values vary by a factor of ten thousand and
mostly tell you how well the headset was sitting. Then we cut sixty-second
windows, every fifteen seconds.

---

## Slide 9 — Methodology  (~90 s)

This is the slide I'd ask you to remember.

[PAUSE]

Every participant has one label. So if you split your data randomly, the model
doesn't learn pain. It learns to recognise people. And because our windows
overlap, near-identical slices of the same recording end up in training and in
testing at the same time. You get a beautiful number that means absolutely
nothing.

[PAUSE]

So every split we do is grouped by person. All of one participant's windows go
to training, or all of them go to testing. Never both. We also stratify, so the
ten menstrual-pain participants spread across the folds instead of landing in
one.

And we didn't leave that as a good intention. The splitting function checks it
every single time, and raises an error if a person ever appears on both sides.
It has its own tests. On a dataset this small, the split isn't a detail. The
split is the result.

[PAUSE]

One more thing, and this is a correction to our own work. What's the number to
beat? A model that just guesses "headache" every time gets thirty-six percent
accuracy. We quoted that as our bar for about a week.

But its macro F one is zero point one three two. Because it scores a flat zero
on three of the four classes.

Macro F one is the honest metric when your classes are this unbalanced. So
nought point one three two is the real floor. Measure your baseline. Don't
derive it in your head.

---

## Slide 10 — Model selection and parameters  (~80 s)

We tested four architectures, plus one deliberate control.

[PAUSE]

The baseline is a small one-dimensional convolutional network. It picks up local
shape in the waveform. Then a CNN-LSTM, which hands those convolutional features
to a recurrent layer. Then a temporal convolutional network, or TCN, which uses
dilated convolutions to see the entire sixty-second window at once, without any
recurrence.

And then the fusion model, which is really why this is a deep learning project.
Our two devices sample at different rates. One hertz for the EEG, four hertz for
the wristband. Rather than resample one and pretend they match, each modality
gets its own encoder branch, and we join the learned representations before the
classifier. That's the Keras functional API doing exactly what it exists for.

[PAUSE]

Every one of these models is deliberately tiny. Tens of thousands of parameters,
not millions. With eighty-three people, a big network just memorises
individuals.

Same reasoning behind the control. We ran LightGBM, a gradient-boosted tree,
which sees only forty summary statistics per window. No time series at all. If
that keeps up with the neural networks, it tells us something real.

It keeps up. More on that shortly.

---

## Slide 11 — Training, validation, testing  (~75 s)

Training is intentionally boring, and I mean that as a compliment.

[PAUSE]

Every model gets the same treatment. We reset the random seed before building
each one, so the comparison is fair. The loss is weighted by class frequency, so
the model can't just ignore the rare categories. And we stop early when
validation loss stops improving.

Everything converges in nine to seventeen epochs. The entire EEG suite, five
folds across four models, runs in under four minutes on a laptop CPU. Small data
has very few advantages. That's one of them.

[PAUSE]

For the fusion experiments, Meng tightened this further with a nested split.
Inside each fold's training set, she carved out a separate validation group of
people. So training, validation, and testing are three completely separate sets
of participants. Early stopping never sees a test subject, not even indirectly.
The notebook asserts that at runtime.

Reproducibility we handled the boring way too. Fixed seeds, a pinned class order
so the confusion matrices always line up, and every results table committed to
the repository right next to the code that produced it.

---

## Slide 12 — Evaluation and performance  (~95 s)

Here are the honest numbers.

[FIG: eeg_baselines_folds.png]

On EEG alone, the TCN comes out best, at nought point two four zero macro F one,
against that floor of nought point one three two. All three neural networks
clear the floor with statistical significance.

[PAUSE]

But don't look at the bars. Look at the dots. Each dot is one fold.

LightGBM's five folds run from nought point one one to nought point three five.
Same model. Same data. It looks either useless or like our best model, depending
entirely on which seventeen people you happened to test it on.

So the TCN's real advantage isn't a higher average. It's that its folds cluster
about twice as tightly.

[FIG: fold4_reversal.png]

[PAUSE]

And this is the most instructive result we produced. On fold four, the CNN
records its worst score of the entire project, while LightGBM records its best.
Same test subjects. Same day.

A paper that evaluated on one split could have concluded that trees beat deep
learning by a factor of three. Or the exact opposite. And both papers would have
looked perfectly rigorous.

That's why every number we report is a mean and a standard deviation across
folds. Not a single split.

[PAUSE]

For context: published work using proper multi-channel research EEG, on an
easier within-subject yes-or-no task, reports an AUC around nought point eight
three. We're using one dry electrode, four categories, and complete strangers in
the test set. Our numbers reflect that, and they should.

The contribution here isn't the accuracy. It's knowing what the accuracy is
worth.

---

## Optional 20-second add-on

Use this only if you're recording the sex-confound finding yourself rather than
leaving it to slide 13 or 14. Check with Meng first so it isn't said twice.

> One last finding, and it's the one I'd flag hardest. All ten of our
> menstrual-pain participants are women. So we asked: can these same models
> predict sex? From the same EEG windows, they read sex at nought point seven
> two balanced accuracy, and pain at nought point three one. The model has a far
> more reliable route to that category than pain itself. We can't fix that with
> ten people. But we can measure it, and say so.
