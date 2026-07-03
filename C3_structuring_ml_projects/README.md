# C3 — Structuring Machine Learning Projects (ML Strategy)

Course 3 of the Deep Learning Specialization has **no programming assignments** —
it is a strategy course. These are concise notes on the key ideas, so the track
is complete. (The practical techniques are exercised in code elsewhere in this
repo: orthogonalization ≈ tuning one thing at a time in C2; the dev/test split
and single-number metric appear in every training script under `results/`.)

## Orthogonalization
Tune one knob for one effect. Chain of assumptions: fit training set well
(bigger net / better optimizer / longer training) → fit dev set well
(regularization / more data) → fit test set well (bigger dev set) → perform in
the real world (change dev set or cost function). Early stopping is discouraged
because it couples two effects (fit train ↓ and overfit ↓) at once.

## Single-number evaluation metric
Pick **one** number so you can compare models instantly. Combine precision and
recall into F1 rather than tracking both. Average across classes/regions when
needed. Speeds up the iterate-experiment loop.

## Satisficing vs optimizing metrics
With N criteria, pick 1 to **optimize** and treat the other N−1 as
**satisficing** thresholds (e.g. maximize accuracy subject to runtime < 100 ms
and ≤ 1 false-positive/day).

## Train / dev / test distributions
Dev and test sets **must come from the same distribution** — the one you
actually care about. Set the dev/test target where the arrow should land, then
aim the team at it. In the deep-learning era use e.g. 98/1/1 splits for
1M+ examples (a 1% dev set is already 10k examples).

## Human-level performance & Bayes error
Use human-level error as a proxy for Bayes (irreducible) error.
- **Avoidable bias** = training error − human-level error → reduce with a bigger
  model, better optimizer, longer training, better architecture.
- **Variance** = dev error − training error → reduce with more data,
  regularization, data augmentation.
Whichever gap is larger tells you what to work on next.

## Error analysis
Manually inspect ~100 mislabeled dev examples and tally the failure categories
(blurry, mislabeled, dog-as-cat, …). The category with the biggest count is the
highest-value thing to fix. Cheap, fast, and prevents months wasted on a 5%
ceiling improvement.

## Mismatched training and dev/test sets
When train and dev come from different distributions, carve out a
**training-dev** set (same distribution as train, not trained on) to separate
variance from **data-mismatch**. Address mismatch with error analysis + making
training data more similar to the target (e.g. artificial data synthesis, used
with care to avoid overfitting to the synthetic subspace).

## Transfer learning
Pre-train on a large dataset (task A), then fine-tune the last layers (or all
layers) on the smaller target task B. Makes sense when A and B share low-level
features and A has far more data. Used in this repo's ResNet / VGG-based
assignments (C4).

## Multi-task learning
Train one network to predict several labels at once (e.g. many objects per
image) with a summed loss. Helps when tasks share features and per-task data is
limited; unlike softmax, each example can carry multiple positive labels.

## End-to-end deep learning
Replace a hand-engineered pipeline with a single network mapping input→output.
Powerful with **enough data**; otherwise a pipeline of sub-tasks (each with its
own data) can beat it. Trade-off: lets the data speak vs. excludes useful
hand-designed components.

---

*Notes summarizing Course 3 of the DeepLearning.AI Deep Learning Specialization
(Andrew Ng). This is an independent educational summary.*
