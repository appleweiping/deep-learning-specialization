# Deep Learning Specialization — from-scratch implementations

> Every programming assignment of the **DeepLearning.AI Deep Learning Specialization**
> (Andrew Ng, Coursera), re-implemented independently and verified with real runs — part of a
> [csdiy.wiki](https://csdiy.wiki/) full-catalog build.

![status](https://img.shields.io/badge/status-complete-brightgreen)
![python](https://img.shields.io/badge/python-3.11-informational)
![torch](https://img.shields.io/badge/torch-2.x%20(CPU)-informational)
![license](https://img.shields.io/badge/license-MIT-blue)

## Overview

The specialization has five courses (C1–C5). The mechanics of neural networks are built
**from scratch in NumPy** (forward/backward props, gradient checking, He initialization,
L2/dropout, mini-batch/Momentum/Adam, conv & pool forward+backward, RNN/LSTM cells, BPTT with
gradient clipping, scaled-dot-product/multi-head attention, sinusoidal positional encoding). The
larger framework-based assignments (ResNet, U-Net, face recognition, neural style transfer,
NMT with attention, Transformer) are built in **PyTorch (CPU)**. Every result below was
**measured on this machine** and the logs are committed under [`results/`](results/); the numeric
building blocks are additionally pinned by 33 `pytest` tests (numerical gradient checks, attention
convexity, look-ahead masking, etc.).

C3 (*Structuring Machine Learning Projects*) is a strategy course with **no programming
assignments** — it is covered by concise notes in
[`C3_structuring_ml_projects/README.md`](C3_structuring_ml_projects/README.md).

## Results (measured on CPU, `torch.set_num_threads(3)`, `OMP_NUM_THREADS=3`)

### C1 — Neural Networks and Deep Learning *(NumPy, MNIST 0-vs-1 subset & planar dataset)*

| Assignment | What it does | Result (measured) |
|---|---|---|
| Logistic regression | single-neuron classifier, vectorized | train 96.55% / **test 94.60%** |
| Planar classification | 1-hidden-layer net, hidden-size sweep | best **91.8%** (n_h=8); n_h=4 → 90.8% |
| Deep NN application | L-layer `[n_x,20,7,5,1]` net | 2-layer test 94.90% / **4-layer test 94.90%** |

### C2 — Improving Deep Neural Networks *(NumPy)*

| Assignment | What it does | Result (measured) |
|---|---|---|
| Initialization | zeros vs random vs He | He **93.0%** vs random 51.75% vs zeros 50.0% |
| Regularization | none / L2(λ=0.3) / dropout(keep=0.7) | L2 **test 90.79%** vs none test 86.05% |
| Gradient checking | numerical vs analytic grads | correct rel-err **1.5e-09** (PASS); bug 2.25e-01 (DETECTED) |
| Optimization | GD / Momentum / Adam | Adam **96.33%** vs GD/Momentum 93.33% |
| Framework intro (PyTorch) | 8-epoch MLP | **final test 96.80%** |

### C4 — Convolutional Neural Networks

| Assignment | What it does | Result (measured) |
|---|---|---|
| Conv step-by-step (NumPy) | conv/pool forward **and** backward from scratch | backward matches numerical grads (see `test_c4_conv.py`) |
| ResNet (PyTorch) | identity + convolutional blocks, CIFAR-10 4-class subset | 222,404 params, **test acc 74.70%** (6 epochs) |
| Neural style transfer (PyTorch) | Gatys et al., VGG19 Gram-matrix style loss | style loss **0.00278 → 0.00031** (200 steps) — [`neural_style_transfer.png`](results/C4/neural_style_transfer.png) |
| U-Net segmentation (PyTorch) | encoder–decoder w/ skip connections | 483,475 params, **mIoU 0.997**, pixel-acc 99.95% (8 epochs) — [`unet_segmentation_sample.png`](results/C4/unet_segmentation_sample.png) |
| Face recognition (PyTorch) | triplet-loss embedding + verification | **verification acc 93.75%** (thr 0.779) |

### C5 — Sequence Models

| Assignment | What it does | Result (measured) |
|---|---|---|
| RNN & LSTM step-by-step (NumPy) | cell + full forward, by hand | shapes/gates verified; `rnn_cell_backward` matches numerical grad |
| Char-level RNN — dinosaur names (NumPy) | BPTT + gradient clipping, samples names | smoothed loss **23.09 → 21.91**; generates *Odrephodynhus, Eloceratops, …* |
| Word vectors + debiasing (NumPy) | analogies, neutralize, equalize (Bolukbasi et al.) | **real GloVe-6B (400,001 vectors)**: king→queen, japan→tokyo; neutralize cos **0.000000**; equalize ±0.7004 |
| NMT with attention (PyTorch) | Bi-LSTM encoder + Bahdanau attention, date translation | **char-acc 100%, exact-date 100%** on held-out test |
| Transformer from scratch (PyTorch) | MHA + positional encoding + enc/dec, seq2seq | **sequence accuracy 91.82%** on the reverse task |

**Verification tests:** `33 passed in 20.40s` (`pytest tests/`).

## Implemented assignments

- [x] **C1** — logistic regression · planar 1-hidden-layer net · deep L-layer application
- [x] **C2** — initialization · regularization (L2/dropout) · gradient checking · optimizers (GD/Momentum/Adam) · framework intro
- [x] **C3** — ML-strategy notes (course has no programming labs)
- [x] **C4** — conv/pool forward+backward · ResNet · neural style transfer · U-Net segmentation · face recognition (triplet loss)
- [x] **C5** — RNN/LSTM cells · char-level RNN (dinosaur names) · word vectors + debiasing · NMT with attention · Transformer

## Project structure

```
deep-learning-specialization/
├── C1_neural_networks_and_deep_learning/   # NumPy: logistic reg, planar, deep NN
├── C2_improving_deep_neural_networks/      # init, regularization, grad-check, optimizers
├── C3_structuring_ml_projects/             # ML-strategy notes (no code labs)
├── C4_convolutional_neural_networks/       # conv-from-scratch, ResNet, style, U-Net, face-rec
├── C5_sequence_models/                     # RNN/LSTM, char-RNN, word vectors, NMT, Transformer
├── common/                                 # nn_numpy engine, gradient_check, dataset loaders
├── tests/                                  # 33 pytest tests (grad checks, attention, masks, ...)
├── results/                                # committed run logs + figures (evidence)
├── requirements.txt
└── LICENSE
```

## How to run

```bash
# Python repos use the shared csdiy env (Python 3.11):
#   D:\Project\_csdiy\.venv-ml\Scripts\python.exe
python -m pip install -r requirements.txt   # numpy, torch(+torchvision, CPU), matplotlib, pytest

# Run the test suite (fast, no downloads):
python -m pytest tests/ -v

# Reproduce individual assignments (datasets download once, then cached under data/):
python C1_neural_networks_and_deep_learning/a1_logistic_regression.py
python C2_improving_deep_neural_networks/a4_optimization.py
python C4_convolutional_neural_networks/a2_resnet.py --epochs 6      # CIFAR-10 subset
python C4_convolutional_neural_networks/a5_unet_segmentation.py      # -> results/C4/*.png
python C4_convolutional_neural_networks/a4_neural_style_transfer.py  # VGG19, -> *.png
python C5_sequence_models/a2_char_rnn_dinos.py                       # dinosaur-name RNN
python C5_sequence_models/a3_word_vectors_debiasing.py               # downloads GloVe-6B
python C5_sequence_models/a4_nmt_attention.py                        # date translation
python C5_sequence_models/a5_transformer.py                          # reverse-sequence task
```

Datasets (MNIST, CIFAR-10, GloVe-6B) are downloaded at runtime and `.gitignore`d — never
redistributed. On CPU, ResNet ≈ 15 min for 6 epochs and U-Net ≈ 4 min for 8 epochs.

## Verification

- **Unit/numerical tests** — `pytest tests/` → `33 passed`. Includes finite-difference gradient
  checks for the NumPy backprop, conv backward, and RNN cell backward; convexity of attention
  weights; look-ahead / padding masks blocking the future; He-init variance; Adam bias correction.
- **Real runs** — every table entry above corresponds to a committed log in
  [`results/`](results/) (e.g. [`results/C4/a5_unet_segmentation.log`](results/C4/a5_unet_segmentation.log)
  for mIoU 0.997, [`results/C5/a4_nmt_attention.log`](results/C5/a4_nmt_attention.log) for 100%
  exact-date translation, [`results/C5/a3_word_vectors_debiasing.log`](results/C5/a3_word_vectors_debiasing.log)
  for the real GloVe debiasing algebra).
- **Figures** — `results/C4/unet_segmentation_sample.png` (input/GT/prediction) and
  `results/C4/neural_style_transfer.png` (content/style/generated triptych).

## Tech stack

Python 3.11 · NumPy (from-scratch NN/CNN/RNN/attention engine) · PyTorch 2.x + torchvision
(CPU) for ResNet/U-Net/face-rec/style-transfer/NMT/Transformer · matplotlib · pytest.

## Key ideas / what I learned

- **Backprop is just the chain rule, verified numerically.** The NumPy engine's gradients are
  checked against finite differences to ~1e-9 relative error; a deliberately buggy backprop is
  caught by the same check.
- **Optimization & regularization change outcomes measurably** — He-init unlocks training a deep
  ReLU net (93% vs ~50%), Adam beats plain GD, and L2 trades train accuracy for a real
  generalization gain here.
- **Conv and pooling backward from scratch** — writing `conv_backward` / `pool_backward` by hand
  (not autograd) makes the parameter-sharing gradient flow concrete.
- **Attention makes seq2seq exact** — the Bahdanau-attention date translator reaches 100%
  exact-match; the hand-built Transformer (scaled-dot-product + multi-head + positional encoding
  + look-ahead masks) solves the sequence task to 91.8%.
- **Embeddings encode — and can debias — bias** — on real GloVe vectors the gender axis is
  measurable, and the neutralize/equalize algebra provably removes/symmetrizes it.

## Credits & license

Based on the programming assignments of the **Deep Learning Specialization** by **Andrew Ng /
DeepLearning.AI** (Coursera). This repository is an independent educational reimplementation; all
course materials, datasets, and specifications belong to their original authors. Original code in
this repo is released under the [MIT License](LICENSE).
