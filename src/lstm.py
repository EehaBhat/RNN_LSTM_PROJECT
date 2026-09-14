"""
LSTM cell implemented from scratch with NumPy.

Gate order in the stacked weight matrices is (i, f, g, o) -- input gate,
forget gate, cell/candidate gate, output gate -- which matches PyTorch's
nn.LSTMCell layout, so weights can be copied directly between the two
implementations for a numerical equivalence check (see tests/test_lstm_vs_pytorch.py).

Forward recurrence (per timestep t):
    z_t = x_t @ Wih.T + h_{t-1} @ Whh.T + b        (4H,) pre-activations, split into i,f,g,o
    i_t = sigmoid(z_i)      input gate
    f_t = sigmoid(z_f)      forget gate
    g_t = tanh(z_g)         candidate cell update
    o_t = sigmoid(z_o)      output gate
    c_t = f_t * c_{t-1} + i_t * g_t                 cell state
    h_t = o_t * tanh(c_t)                           hidden state

Readout (applied once, at the final timestep, for the tasks in this project):
    y = h_T @ Why.T + by
"""
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class LSTM:
    def __init__(self, input_size, hidden_size, output_size, seed=0):
        rng = np.random.default_rng(seed)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        H = hidden_size

        def xavier(fan_in, fan_out):
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            return rng.uniform(-limit, limit, size=(fan_out, fan_in))

        # Stacked (4H, in) / (4H, H) matrices, gate order i,f,g,o.
        self.Wih = xavier(input_size, 4 * H)
        self.Whh = xavier(H, 4 * H)
        self.b = np.zeros(4 * H)
        # A common trick (Jozefowicz et al. 2015) is to init the forget-gate
        # bias positive so the network defaults to "remember" early in
        # training, rather than starting from a coin-flip forget gate. We use
        # 3.0 (forget gate ~= 0.95 at init) -- see gradient_analysis.py and
        # WRITEUP.md for a sweep showing exactly how this bias controls how
        # much gradient survives 60 steps.
        self.b[H:2 * H] = 3.0  # forget gate bias

        self.Why = xavier(H, output_size)
        self.by = np.zeros(output_size)

    def params(self):
        return {"Wih": self.Wih, "Whh": self.Whh, "b": self.b,
                "Why": self.Why, "by": self.by}

    def load_gates(self, Wih, Whh, b):
        """Overwrite gate weights directly (used to sync with nn.LSTMCell for testing)."""
        self.Wih = Wih.copy()
        self.Whh = Whh.copy()
        self.b = b.copy()

    def _slice(self, z):
        H = self.hidden_size
        return z[:, 0:H], z[:, H:2 * H], z[:, 2 * H:3 * H], z[:, 3 * H:4 * H]

    def forward(self, X):
        """
        X: (T, B, input_size)
        Returns y: (B, output_size), cache for backward()
        """
        T, B, _ = X.shape
        H = self.hidden_size

        h_prev = np.zeros((B, H))
        c_prev = np.zeros((B, H))

        cache = {
            "X": X,
            "h_hist": np.zeros((T + 1, B, H)),
            "c_hist": np.zeros((T + 1, B, H)),
            "i": np.zeros((T, B, H)), "f": np.zeros((T, B, H)),
            "g": np.zeros((T, B, H)), "o": np.zeros((T, B, H)),
            "tanh_c": np.zeros((T, B, H)),
        }

        for t in range(T):
            x_t = X[t]
            z = x_t @ self.Wih.T + h_prev @ self.Whh.T + self.b
            zi, zf, zg, zo = self._slice(z)
            i_t = sigmoid(zi)
            f_t = sigmoid(zf)
            g_t = np.tanh(zg)
            o_t = sigmoid(zo)
            c_t = f_t * c_prev + i_t * g_t
            tanh_c_t = np.tanh(c_t)
            h_t = o_t * tanh_c_t

            cache["h_hist"][t + 1] = h_t
            cache["c_hist"][t + 1] = c_t
            cache["i"][t] = i_t
            cache["f"][t] = f_t
            cache["g"][t] = g_t
            cache["o"][t] = o_t
            cache["tanh_c"][t] = tanh_c_t

            h_prev, c_prev = h_t, c_t

        y = h_prev @ self.Why.T + self.by
        cache["y"] = y
        return y, cache

    def backward(self, cache, dY):
        """
        dY: (B, output_size)
        Returns grads dict and grad_norms (||dL/dh_t|| per timestep, t=0..T-1),
        same convention as VanillaRNN.backward for a fair side-by-side comparison.
        """
        X = cache["X"]
        h_hist, c_hist = cache["h_hist"], cache["c_hist"]
        i_all, f_all, g_all, o_all = cache["i"], cache["f"], cache["g"], cache["o"]
        tanh_c_all = cache["tanh_c"]
        T, B, _ = X.shape
        H = self.hidden_size

        grads = {k: np.zeros_like(v) for k, v in self.params().items()}

        h_T = h_hist[T]
        grads["Why"] = dY.T @ h_T
        grads["by"] = dY.sum(axis=0)

        dh_next = dY @ self.Why
        dc_next = np.zeros((B, H))
        grad_norms = [0.0] * T

        for t in reversed(range(T)):
            dh = dh_next
            grad_norms[t] = float(np.linalg.norm(dh) / B)

            c_t = c_hist[t + 1]
            c_prev = c_hist[t]
            i_t, f_t, g_t, o_t = i_all[t], f_all[t], g_all[t], o_all[t]
            tanh_c_t = tanh_c_all[t]

            do = dh * tanh_c_t
            dc = dc_next + dh * o_t * (1 - tanh_c_t ** 2)

            df = dc * c_prev
            di = dc * g_t
            dg = dc * i_t
            dc_prev = dc * f_t

            do_raw = do * o_t * (1 - o_t)
            df_raw = df * f_t * (1 - f_t)
            di_raw = di * i_t * (1 - i_t)
            dg_raw = dg * (1 - g_t ** 2)

            dz = np.concatenate([di_raw, df_raw, dg_raw, do_raw], axis=1)

            x_t = X[t]
            h_prev = h_hist[t]

            grads["Wih"] += dz.T @ x_t
            grads["Whh"] += dz.T @ h_prev
            grads["b"] += dz.sum(axis=0)

            dh_next = dz @ self.Whh
            dc_next = dc_prev

        return grads, grad_norms

    def sgd_step(self, grads, lr):
        self.Wih -= lr * grads["Wih"]
        self.Whh -= lr * grads["Whh"]
        self.b -= lr * grads["b"]
        self.Why -= lr * grads["Why"]
        self.by -= lr * grads["by"]

    def clip_grads(self, grads, max_norm):
        total_sq = sum(np.sum(g ** 2) for g in grads.values())
        total_norm = np.sqrt(total_sq)
        if total_norm > max_norm:
            scale = max_norm / (total_norm + 1e-8)
            for k in grads:
                grads[k] *= scale
        return total_norm
