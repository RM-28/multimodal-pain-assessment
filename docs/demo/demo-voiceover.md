# Demo voiceover

For `split-demo.html`. About **three minutes**

**Press space the moment it loads**, or it will run away from you in 32 seconds.
Then hit the right arrow when you reach each cue. Each press plays that scene
and holds on its last frame, so you can talk over a still

Say the numbers the way you would out loud: "point eight nine nine", "macro F
one"

---

## 1 · The 83 people

Before any of the modelling, here's the decision that mattered most.

We have 83 participants. Each wore an EEG headband and a wristband for twenty
minutes, and each told us one pain type. That's the whole dataset.

The loader's on the right, and the line to watch is the last one. Every row from
a person carries that same single label.

**→**

---

## 2 · Cutting it into windows

Now, to actually train on this, we chop each recording into overlapping
sixty-second windows.

And suddenly it looks like a much bigger dataset. Six and a half thousand
windows instead of 83 people. That feels like progress.

Watch the highlighted line though. Alongside the signal and the label, we're
holding onto the participant ID for every single window. That turns out to be
the whole ballgame.

**→**

---

## 3 · The shuffle

So here's the obvious move. Shuffle the windows, take eighty percent to train
on, keep the rest for testing. Three lines of numpy.

Everything on screen is our actual split. Each dot is fourteen real windows, and
the red lines join windows that came from the same person.

When we counted, all 83 participants had windows on both sides.

**→**

---

## 4 · What that buys the model

Which means the model never has to learn anything about pain at all.

Our windows overlap, so it can simply spot someone it already met in training
and recall what they said. That's a much easier problem.

We ran it. Same network, same settings, and it scores point eight nine nine.
That's a number that would have looked lovely in a report and told us nothing.

**→**

---

## 5 · Splitting by person instead

So we split by person instead.

Everything from one participant goes to one side. Sixty-seven people to train
on, sixteen to test on, and nothing crosses the middle. You can see them moving
across as whole clusters now.

That's all the groups argument is doing in the function on the right. It tells
scikit-learn to keep people together.

**→**

---

## 6 · The real number

And now that the model is facing people it has genuinely never seen, it scores
point two five eight.

We also didn't want to just trust that our split was correct. So the function
you're looking at runs on every fold, and it throws an exception if anybody
turns up on both sides. It's got its own tests, and the whole suite runs in
about a second.

**→**

---

## 7 · So what

Same data, same model, same eighty-twenty ratio. Point eight nine nine, or
point two five eight, and all we changed was where we drew the line.

Here's why that matters beyond our project. If a pain model scores well by
recognising the patients it trained on, it's useless in a clinic, because in a
clinic every patient is new.

So every result we report is grouped by participant, and we're sceptical of
single-split results on small datasets.

**[end]**

---

## Spare 15 seconds, if you need it

"Our headline result is actually the five-fold average, point two four zero,
give or take point zero four six. That spread is wide because each test fold
only holds seventeen people, which is its own argument for never quoting a
single split."

## Likely question, and the answer

**"Is that animation showing your real data?"**

Yes. The panel membership and the participant links come from
`scripts/run_leakage_demo.py`, exported to `split-data.json`. One drawn dot
stands for fourteen real windows, and we draw twenty of the eighty-three
participant links so you can still see the panels. The dot positions inside
each panel are arbitrary, since the permutation doesn't give windows a location
