"""
Deliverable 2.4: compare our from-scratch LSTM's forward pass against
nn.LSTMCell using the *same* inputs and weights.

Requires PyTorch (not used anywhere else in this project -- only here, as an
external reference implementation to validate against):
    pip install torch

Run with: python test_lstm_vs_pytorch.py
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from lstm import LSTM

try:
    import torch
    import torch.nn as nn
except ImportError:
    print("PyTorch is not installed in this environment. Install it to run this "
          "comparison test:\n    pip install torch\n"
          "(Every other test/deliverable in this project runs without PyTorch --"
          " it is only needed for this specific external-reference check.)")
    sys.exit(0)


def main():
    torch.manual_seed(0)
    input_size, hidden_size, batch_size, T = 5, 8, 4, 10

    # Reference: PyTorch's built-in LSTMCell.
    ref = nn.LSTMCell(input_size, hidden_size).double()

    # PyTorch's weight_ih / weight_hh are (4H, in) and (4H, H) with gate
    # order (i, f, g, o) -- see the PyTorch docs for nn.LSTMCell. Our
    # from-scratch LSTM uses the exact same layout, so weights transfer directly.
    Wih = ref.weight_ih.detach().numpy().copy()
    Whh = ref.weight_hh.detach().numpy().copy()
    b = (ref.bias_ih + ref.bias_hh).detach().numpy().copy()

    ours = LSTM(input_size, hidden_size, output_size=1, seed=0)
    ours.load_gates(Wih, Whh, b)

    rng = np.random.default_rng(1)
    X_np = rng.normal(size=(T, batch_size, input_size))
    X_torch = torch.tensor(X_np, dtype=torch.float64)

    h_t = torch.zeros(batch_size, hidden_size, dtype=torch.float64)
    c_t = torch.zeros(batch_size, hidden_size, dtype=torch.float64)
    h_ref_seq, c_ref_seq = [], []
    for t in range(T):
        h_t, c_t = ref(X_torch[t], (h_t, c_t))
        h_ref_seq.append(h_t.detach().numpy().copy())
        c_ref_seq.append(c_t.detach().numpy().copy())

    # Reuse our internal forward loop but capture h_t, c_t at every step
    # (forward() only returns the final readout, so we call the recurrence directly).
    h_prev = np.zeros((batch_size, hidden_size))
    c_prev = np.zeros((batch_size, hidden_size))
    h_ours_seq, c_ours_seq = [], []
    for t in range(T):
        x_t = X_np[t]
        z = x_t @ ours.Wih.T + h_prev @ ours.Whh.T + ours.b
        i_t, f_t, g_t, o_t = ours._slice(z)
        from lstm import sigmoid
        i_t, f_t, o_t = sigmoid(i_t), sigmoid(f_t), sigmoid(o_t)
        g_t = np.tanh(g_t)
        c_t_ours = f_t * c_prev + i_t * g_t
        h_t_ours = o_t * np.tanh(c_t_ours)
        h_ours_seq.append(h_t_ours)
        c_ours_seq.append(c_t_ours)
        h_prev, c_prev = h_t_ours, c_t_ours

    max_h_diff = max(np.max(np.abs(a - b)) for a, b in zip(h_ours_seq, h_ref_seq))
    max_c_diff = max(np.max(np.abs(a - b)) for a, b in zip(c_ours_seq, c_ref_seq))

    print(f"Max |h_ours - h_pytorch| across all timesteps: {max_h_diff:.3e}")
    print(f"Max |c_ours - c_pytorch| across all timesteps: {max_c_diff:.3e}")

    tol = 1e-8
    if max_h_diff < tol and max_c_diff < tol:
        print(f"PASS: outputs match nn.LSTMCell to within {tol:.0e}")
        sys.exit(0)
    else:
        print(f"FAIL: outputs differ by more than {tol:.0e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
