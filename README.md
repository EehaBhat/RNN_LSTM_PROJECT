# RNN & LSTM from Scratch: Investigating Vanishing Gradients

A from-scratch (NumPy only, no autograd) implementation of a vanilla RNN cell
and an LSTM cell, including manually-derived backpropagation through time,
used to investigate *why* standard RNNs struggle with long-term dependencies
and how the LSTM's architecture addresses it.

See `WRITEUP.md` for the full write-up, derivations, and discussion.

## Project layout

```
src/
  rnn.py                Vanilla RNN cell: forward pass + manual BPTT
  lstm.py                LSTM cell: forward pass + manual backward pass
  losses.py              Numerically stable BCE-with-logits loss
  data.py                The "signal-in-noise" long-term dependency task
  gradient_analysis.py   Deliverable 2.2 / 2.5: vanishing-gradient plot + sweeps
  train.py               Deliverable 2.5 + stretch goal: training comparison,
                          gradient clipping
tests/
  test_gradcheck.py           Numerical gradient check (finite differences)
                               for both the RNN and LSTM backward passes
  test_lstm_vs_pytorch.py     Deliverable 2.4: forward-pass equivalence check
                               against torch.nn.LSTMCell
run_tests.py              Runs both correctness checks and prints a summary
results/                   Generated plots and CSVs (see below)
WRITEUP.md                 Full write-up: derivations, results, discussion
```

## Setup

```bash
pip install -r requirements.txt
```

Everything except `tests/test_lstm_vs_pytorch.py` runs with just `numpy` and
`matplotlib` -- PyTorch is used *only* as an external reference implementation
for that one comparison test, never inside the actual RNN/LSTM implementation.

## Running things

**Correctness harness** (gradient check + PyTorch comparison):
```bash
python run_tests.py
```
This prints a PASS/FAIL summary. If PyTorch isn't installed, the PyTorch
comparison prints a note and is skipped (not counted as a failure); everything
else runs regardless.

**Vanishing gradient demonstration** (deliverable 2.2, produces
`results/vanishing_gradients.png` and the sweep CSVs referenced in
`WRITEUP.md`):
```bash
cd src && python gradient_analysis.py
```

**Training comparison** (deliverable 2.5 + gradient clipping stretch goal,
produces `results/training_comparison.png`):
```bash
cd src && python train.py
```
Optional flags: `--seq_len`, `--steps`, `--hidden_size`, `--lr`, `--clip`.

## The task

Both models are trained on a synthetic **signal-in-noise recall task**: at
`t=0` the input carries one bit of information (`+1` or `-1`); every
subsequent timestep (`t=1 .. T-1`, with `T >= 50`) is pure Gaussian noise.
The network must output the sign of the very first input, read out at the
final timestep. Solving this requires gradient information from the loss to
propagate all the way back through `T-1` nonlinearities -- exactly the
regime where vanilla RNNs are known to struggle. See `src/data.py` and
`WRITEUP.md` for details and references.

## Headline result

With recurrent weights scaled into the vanishing-gradient regime, the
gradient reaching `t=0` in the vanilla RNN is about **17 orders of
magnitude** smaller than the gradient at the final timestep (`T=60`), and
the RNN never learns the task (stuck at chance accuracy). The LSTM's
gradient stays roughly flat across the same 60 steps and it converges to
~100% accuracy. Full numbers, plots, and the mechanism behind this (the
forget gate and the additive cell-state pathway) are in `WRITEUP.md`.
