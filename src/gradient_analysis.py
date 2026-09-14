"""
Deliverable 2.2 / 2.5: demonstrate vanishing gradients in the vanilla RNN and
show how the LSTM behaves differently on the same long-term dependency task
(seq_len >= 50).

We take one batch from the signal-in-noise task and run a forward + backward
pass through *freshly initialized* (untrained) networks, then plot
||dL/dh_t|| for t = 0 (earliest -- the gradient has to survive the full trip
back to here) through t = T-1 (latest -- right next to the loss).

Freshly initialized networks are used deliberately: this isolates the
*architectural* effect from whatever a specific training run happens to do.

Two design choices matter a lot here and are swept explicitly below, because
they are the whole point of the exercise (see WRITEUP.md section 2.6):

  - RNN: the spectral scale of Whh. With small recurrent weights the
    repeated multiplication by Whh and by tanh'(.) <= 1 shrinks the gradient
    geometrically as it goes back in time -> vanishing gradients. With large
    weights the same repeated multiplication *grows* the gradient instead ->
    exploding gradients. Both are the same underlying mechanism.

  - LSTM: the forget-gate bias at init. The backward path through the cell
    state is dc_{t-1} = dc_t * f_t -- no squashing derivative multiplied in,
    just multiplication by the forget gate's value itself. If f_t stays near
    1, gradient survives almost losslessly across many steps; if f_t is near
    0.5 (unbiased init), it still decays geometrically, just less harshly
    than the RNN's tanh'-multiplied path.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rnn import VanillaRNN
from lstm import LSTM
from data import make_batch
from losses import bce_with_logits


def compute_grad_norms(model, X, y):
    logits, cache = model.forward(X)
    _, dlogits = bce_with_logits(logits, y)
    _, grad_norms = model.backward(cache, dlogits)
    return np.array(grad_norms)


def main():
    seq_len = 60
    batch_size = 32
    hidden_size = 32
    X, y = make_batch(batch_size, seq_len, noise_std=0.2, seed=42)

    # --- main comparison: "typical" vanilla RNN vs. LSTM with a sane init ---
    rnn = VanillaRNN(input_size=1, hidden_size=hidden_size, output_size=1, seed=1)
    rnn.Wxh *= 0.5   # shrink recurrent weight scale -> puts it in the vanishing regime
    rnn.Whh *= 0.5

    lstm = LSTM(input_size=1, hidden_size=hidden_size, output_size=1, seed=1)

    rnn_norms = compute_grad_norms(rnn, X, y)
    lstm_norms = compute_grad_norms(lstm, X, y)

    print("=== Main comparison (T=%d) ===" % seq_len)
    print(f"RNN : ||dL/dh_0|| = {rnn_norms[0]:.3e}   ||dL/dh_{seq_len-1}|| = {rnn_norms[-1]:.3e}"
          f"   ratio(last/first) = {rnn_norms[-1] / (rnn_norms[0] + 1e-30):.3e}")
    print(f"LSTM: ||dL/dh_0|| = {lstm_norms[0]:.3e}   ||dL/dh_{seq_len-1}|| = {lstm_norms[-1]:.3e}"
          f"   ratio(last/first) = {lstm_norms[-1] / (lstm_norms[0] + 1e-30):.3e}")

    t = np.arange(seq_len)
    plt.figure(figsize=(8, 5))
    plt.semilogy(t, rnn_norms + 1e-30, label="Vanilla RNN", marker="o", markersize=3)
    plt.semilogy(t, lstm_norms + 1e-30, label="LSTM", marker="s", markersize=3)
    plt.xlabel("timestep t  (t=0 is 59 steps back from the loss, t=59 is right next to it)")
    plt.ylabel(r"$\|\partial L / \partial h_t\|$  (log scale)")
    plt.title(f"Gradient magnitude vs. distance from the loss (T={seq_len}, untrained networks)")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig("../results/vanishing_gradients.png", dpi=150)
    print("Saved plot to results/vanishing_gradients.png")

    # --- supplementary: forget-gate bias sweep, isolating the mechanism ---
    print("\n=== LSTM forget-gate bias sweep (isolates the mechanism) ===")
    lines = ["forget_bias, mean_forget_gate, grad_t0, grad_t59, ratio"]
    for fbias in [0.0, 1.0, 2.0, 3.0, 5.0]:
        lstm_s = LSTM(input_size=1, hidden_size=hidden_size, output_size=1, seed=1)
        H = lstm_s.hidden_size
        lstm_s.b[H:2 * H] = fbias
        logits, cache = lstm_s.forward(X)
        _, dlogits = bce_with_logits(logits, y)
        _, gn = lstm_s.backward(cache, dlogits)
        gn = np.array(gn)
        fmean = cache["f"].mean()
        ratio = gn[-1] / (gn[0] + 1e-30)
        line = f"{fbias:.1f}, {fmean:.3f}, {gn[0]:.3e}, {gn[-1]:.3e}, {ratio:.3e}"
        print(line)
        lines.append(line)
    with open("../results/forget_bias_sweep.csv", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("Saved sweep to results/forget_bias_sweep.csv")

    # --- supplementary: RNN weight-scale sweep (vanishing vs exploding) ---
    print("\n=== RNN recurrent weight scale sweep (isolates the mechanism) ===")
    lines2 = ["weight_scale, grad_t0, grad_t59, ratio"]
    for scale in [0.5, 1.0, 1.5, 2.0]:
        rnn_s = VanillaRNN(input_size=1, hidden_size=hidden_size, output_size=1, seed=1)
        rnn_s.Wxh *= scale
        rnn_s.Whh *= scale
        gn = compute_grad_norms(rnn_s, X, y)
        ratio = gn[-1] / (gn[0] + 1e-30)
        line = f"{scale:.1f}, {gn[0]:.3e}, {gn[-1]:.3e}, {ratio:.3e}"
        print(line)
        lines2.append(line)
    with open("../results/rnn_scale_sweep.csv", "w") as f:
        f.write("\n".join(lines2) + "\n")
    print("Saved sweep to results/rnn_scale_sweep.csv")


if __name__ == "__main__":
    main()
