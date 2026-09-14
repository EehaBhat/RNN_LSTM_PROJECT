"""
The "signal-in-noise" long-term dependency task.

At t = 0 the network is shown a single bit of information (+1 or -1). Every
subsequent timestep (t = 1 .. T-1) is pure noise, irrelevant to the answer.
At the final timestep, the network must output the sign of the very first
input it ever saw.

To solve this, gradient information from the loss (computed at t = T-1) has
to propagate all the way back to t = 0 -- through T-1 nonlinearities -- for
the network to learn to preserve that signal. With T >= 50 this is exactly
the regime where vanilla RNNs suffer from vanishing gradients, and is a
standard synthetic benchmark for testing long-range memory (in the same
family as the classic "copy task" / "adding problem" used in the original
LSTM and later IRNN / uRNN papers).

Input is 1-dimensional: X[0] = signal in {-1, +1}, X[1:] ~ N(0, noise_std).
Target: y in {0, 1} where 1 corresponds to signal == +1 (binary classification).
"""
import numpy as np


def make_batch(batch_size, seq_len, noise_std=0.2, seed=None):
    """
    Returns:
        X: (seq_len, batch_size, 1) float array
        y: (batch_size, 1) float array of 0/1 targets
    """
    rng = np.random.default_rng(seed)
    signal = rng.choice([-1.0, 1.0], size=(batch_size,))
    noise = rng.normal(0.0, noise_std, size=(seq_len, batch_size))

    X = noise.copy()
    X[0, :] = signal
    X = X[:, :, None]  # (T, B, 1)

    y = (signal > 0).astype(np.float64).reshape(batch_size, 1)
    return X, y
