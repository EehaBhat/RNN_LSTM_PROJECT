"""
Vanilla RNN implemented from scratch with NumPy.

Forward recurrence (per timestep t):
    a_t = x_t @ Wxh.T + h_{t-1} @ Whh.T + bh
    h_t = tanh(a_t)

Readout (applied once, at the final timestep, for the tasks in this project):
    y = h_T @ Why.T + by

Backward pass is full Backpropagation Through Time (BPTT), derived by hand
(no autograd). See WRITEUP.md for the derivation.
"""
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class VanillaRNN:
    def __init__(self, input_size, hidden_size, output_size, seed=0):
        rng = np.random.default_rng(seed)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Xavier-ish init to keep the network trainable at all; this task is
        # about gradient *behavior*, not about squeezing out best accuracy.
        def xavier(fan_in, fan_out):
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            return rng.uniform(-limit, limit, size=(fan_out, fan_in))

        self.Wxh = xavier(input_size, hidden_size)
        self.Whh = xavier(hidden_size, hidden_size)
        self.bh = np.zeros(hidden_size)

        self.Why = xavier(hidden_size, output_size)
        self.by = np.zeros(output_size)

    def params(self):
        return {"Wxh": self.Wxh, "Whh": self.Whh, "bh": self.bh,
                "Why": self.Why, "by": self.by}

    def forward(self, X):
        """
        X: (T, B, input_size)
        Returns:
            y: (B, output_size)   -- readout from the final hidden state
            cache: everything backward() needs
        """
        T, B, _ = X.shape
        H = self.hidden_size

        h_prev = np.zeros((B, H))
        h_states = np.zeros((T, B, H))   # h_t for t = 0..T-1
        a_states = np.zeros((T, B, H))   # pre-activation a_t
        h_hist = np.zeros((T + 1, B, H))  # h_hist[0] = h_{-1} = 0, h_hist[t+1] = h_t
        h_hist[0] = h_prev

        for t in range(T):
            x_t = X[t]
            a_t = x_t @ self.Wxh.T + h_prev @ self.Whh.T + self.bh
            h_t = np.tanh(a_t)
            a_states[t] = a_t
            h_states[t] = h_t
            h_hist[t + 1] = h_t
            h_prev = h_t

        y = h_prev @ self.Why.T + self.by

        cache = {"X": X, "h_hist": h_hist, "a_states": a_states, "y": y}
        return y, cache

    def backward(self, cache, dY):
        """
        dY: (B, output_size) -- dL/dy
        Returns:
            grads: dict of parameter gradients (same keys as params())
            grad_norms: list of length T, ||dL/dh_t|| averaged over the batch,
                        ordered from t=0 (earliest) to t=T-1 (latest). This is
                        what we use to visualize vanishing gradients.
        """
        X = cache["X"]
        h_hist = cache["h_hist"]
        a_states = cache["a_states"]
        T, B, _ = X.shape
        H = self.hidden_size

        grads = {k: np.zeros_like(v) for k, v in self.params().items()}

        h_T = h_hist[T]
        grads["Why"] = dY.T @ h_T
        grads["by"] = dY.sum(axis=0)

        dh_next = dY @ self.Why   # gradient flowing into h_T from the readout
        grad_norms = [0.0] * T

        for t in reversed(range(T)):
            dh = dh_next  # total dL/dh_t (only source here is the future step / readout)
            grad_norms[t] = float(np.linalg.norm(dh) / B)

            a_t = a_states[t]
            da = dh * (1 - np.tanh(a_t) ** 2)   # tanh'(a_t) = 1 - h_t^2

            x_t = X[t]
            h_prev = h_hist[t]  # h_{t-1}

            grads["Wxh"] += da.T @ x_t
            grads["Whh"] += da.T @ h_prev
            grads["bh"] += da.sum(axis=0)

            dh_next = da @ self.Whh   # dL/dh_{t-1} contribution from this step

        return grads, grad_norms

    def sgd_step(self, grads, lr):
        for k in self.params():
            getattr(self, k).__isub__(lr * grads[k]) if False else None
        # explicit (avoid relying on in-place aliasing tricks)
        self.Wxh -= lr * grads["Wxh"]
        self.Whh -= lr * grads["Whh"]
        self.bh -= lr * grads["bh"]
        self.Why -= lr * grads["Why"]
        self.by -= lr * grads["by"]

    def clip_grads(self, grads, max_norm):
        """Global-norm gradient clipping across all parameters (stretch goal)."""
        total_sq = sum(np.sum(g ** 2) for g in grads.values())
        total_norm = np.sqrt(total_sq)
        if total_norm > max_norm:
            scale = max_norm / (total_norm + 1e-8)
            for k in grads:
                grads[k] *= scale
        return total_norm
