"""
Deliverable 2.5 (+ stretch: gradient clipping).

Train the vanilla RNN and the LSTM on the same signal-in-noise long-term
dependency task (T=60) and compare:
  - final loss / accuracy
  - the size of ||dL/dh_0|| over the course of training (does the RNN ever
    get usable gradient back to t=0? does clipping change what the RNN
    learns to do?)

Run with: python train.py
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rnn import VanillaRNN
from lstm import LSTM
from data import make_batch
from losses import bce_with_logits, accuracy


def train_model(model, seq_len, steps, batch_size, lr, clip_norm=None, seed=0):
    history = {"loss": [], "acc": [], "grad_t0": [], "clip_triggered": []}
    for step in range(steps):
        X, y = make_batch(batch_size, seq_len, noise_std=0.2, seed=seed * 100000 + step)
        logits, cache = model.forward(X)
        loss, dlogits = bce_with_logits(logits, y)
        grads, grad_norms = model.backward(cache, dlogits)

        raw_norm = np.sqrt(sum(np.sum(g ** 2) for g in grads.values()))
        if clip_norm is not None:
            model.clip_grads(grads, clip_norm)

        model.sgd_step(grads, lr)

        history["loss"].append(loss)
        history["acc"].append(accuracy(logits, y))
        history["grad_t0"].append(grad_norms[0])
        history["clip_triggered"].append(raw_norm > clip_norm if clip_norm else False)
    return history


def eval_model(model, seq_len, n=2000, seed=999):
    X, y = make_batch(n, seq_len, noise_std=0.2, seed=seed)
    logits, _ = model.forward(X)
    loss, _ = bce_with_logits(logits, y)
    return loss, accuracy(logits, y)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seq_len", type=int, default=60)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--hidden_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--clip", type=float, default=1.0, help="gradient clipping max norm")
    args = p.parse_args()

    print(f"Task: signal-in-noise recall, seq_len={args.seq_len}\n")

    # NOTE: the RNNs are initialized with recurrent weights scaled down
    # (x0.5) to deliberately put them in the vanishing-gradient regime
    # demonstrated in gradient_analysis.py -- this is the regime where the
    # vanishing-gradient *problem* actually bites during training, not just
    # at a single forward/backward pass. With "friendlier" (e.g. Xavier,
    # unscaled) initialization this particular easy task is small enough
    # that a vanilla RNN can often still learn it -- see WRITEUP.md for a
    # discussion of why the failure is a matter of degree, not an absolute.
    rnn = VanillaRNN(1, args.hidden_size, 1, seed=1)
    rnn.Wxh *= 0.5
    rnn.Whh *= 0.5

    rnn_clipped = VanillaRNN(1, args.hidden_size, 1, seed=1)
    rnn_clipped.Wxh *= 0.5
    rnn_clipped.Whh *= 0.5

    lstm = LSTM(1, args.hidden_size, 1, seed=1)

    print("Training vanilla RNN (no clipping)...")
    hist_rnn = train_model(rnn, args.seq_len, args.steps, args.batch_size, args.lr,
                            clip_norm=None, seed=1)

    print("Training vanilla RNN (with gradient clipping, max_norm=%.1f)..." % args.clip)
    hist_rnn_clip = train_model(rnn_clipped, args.seq_len, args.steps, args.batch_size, args.lr,
                                 clip_norm=args.clip, seed=1)

    print("Training LSTM (no clipping)...")
    hist_lstm = train_model(lstm, args.seq_len, args.steps, args.batch_size, args.lr,
                             clip_norm=None, seed=1)

    for name, model in [("Vanilla RNN", rnn), ("Vanilla RNN + clipping", rnn_clipped),
                         ("LSTM", lstm)]:
        loss, acc = eval_model(model, args.seq_len)
        print(f"[eval, held-out] {name:24s}  loss={loss:.4f}  acc={acc:.3f}")

    n_clip_events = sum(hist_rnn_clip["clip_triggered"])
    print(f"\nGradient clipping triggered on {n_clip_events}/{args.steps} steps "
          f"for the RNN (max_norm={args.clip}).")

    # --- plots ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    def smooth(x, w=25):
        x = np.array(x)
        if len(x) < w:
            return x
        kernel = np.ones(w) / w
        return np.convolve(x, kernel, mode="valid")

    axes[0].plot(smooth(hist_rnn["loss"]), label="RNN")
    axes[0].plot(smooth(hist_rnn_clip["loss"]), label="RNN + clip")
    axes[0].plot(smooth(hist_lstm["loss"]), label="LSTM")
    axes[0].set_title("Training loss (smoothed)")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("BCE loss")
    axes[0].legend()

    axes[1].plot(smooth(hist_rnn["acc"]), label="RNN")
    axes[1].plot(smooth(hist_rnn_clip["acc"]), label="RNN + clip")
    axes[1].plot(smooth(hist_lstm["acc"]), label="LSTM")
    axes[1].set_title("Training accuracy (smoothed)")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("accuracy")
    axes[1].legend()

    axes[2].semilogy(np.abs(hist_rnn["grad_t0"]) + 1e-12, label="RNN", alpha=0.6)
    axes[2].semilogy(np.abs(hist_rnn_clip["grad_t0"]) + 1e-12, label="RNN + clip", alpha=0.6)
    axes[2].semilogy(np.abs(hist_lstm["grad_t0"]) + 1e-12, label="LSTM", alpha=0.6)
    axes[2].set_title(r"$\|\partial L/\partial h_0\|$ during training")
    axes[2].set_xlabel("step")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig("../results/training_comparison.png", dpi=150)
    print("Saved plot to results/training_comparison.png")


if __name__ == "__main__":
    main()
