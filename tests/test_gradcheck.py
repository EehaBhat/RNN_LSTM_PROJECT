"""
Correctness harness (deliverable 2.1 / 2.3 correctness, referenced by 1.4's
spirit): numerically verify that the hand-derived backward pass (BPTT) for
both VanillaRNN and LSTM matches the true gradient, computed via central
finite differences on the loss.

This is a self-contained ground truth (no external ML library required) --
finite differences directly approximate dL/dparam from the definition of a
derivative, so it is a trusted reference independent of our backward-pass
code.

Run with: python test_gradcheck.py
Exits with code 0 and prints "ALL GRADIENT CHECKS PASSED" on success,
exits with code 1 and prints which checks failed otherwise.
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from rnn import VanillaRNN
from lstm import LSTM
from losses import bce_with_logits


def numerical_grad(model, X, y, param_name, eps=1e-5):
    """Central-difference numerical gradient for one parameter tensor."""
    param = getattr(model, param_name)
    grad = np.zeros_like(param)
    it = np.nditer(param, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        orig = param[idx]

        param[idx] = orig + eps
        logits, _ = model.forward(X)
        loss_plus, _ = bce_with_logits(logits, y)

        param[idx] = orig - eps
        logits, _ = model.forward(X)
        loss_minus, _ = bce_with_logits(logits, y)

        param[idx] = orig
        grad[idx] = (loss_plus - loss_minus) / (2 * eps)
        it.iternext()
    return grad


def analytic_grad(model, X, y):
    logits, cache = model.forward(X)
    _, dlogits = bce_with_logits(logits, y)
    grads, _ = model.backward(cache, dlogits)
    return grads


def relative_error(a, b):
    return np.max(np.abs(a - b) / (np.abs(a) + np.abs(b) + 1e-8))


def check_model(model, X, y, param_names, tol=3e-4, label=""):
    grads = analytic_grad(model, X, y)
    all_ok = True
    for name in param_names:
        num_g = numerical_grad(model, X, y, name)
        ana_g = grads[name]
        err = relative_error(num_g, ana_g)
        status = "OK" if err < tol else "FAIL"
        if err >= tol:
            all_ok = False
        print(f"  [{label}] d{name}: max relative error = {err:.3e}  [{status}]")
    return all_ok


def main():
    rng = np.random.default_rng(0)
    T, B = 6, 3   # small so the O(num_params * T) finite-difference loop is fast
    input_size, hidden_size, output_size = 2, 4, 1

    X = rng.normal(size=(T, B, input_size))
    y = (rng.uniform(size=(B, 1)) > 0.5).astype(np.float64)

    print("Checking VanillaRNN backward pass against numerical gradients...")
    rnn = VanillaRNN(input_size, hidden_size, output_size, seed=0)
    ok_rnn = check_model(rnn, X, y, ["Wxh", "Whh", "bh", "Why", "by"], label="RNN")

    print("\nChecking LSTM backward pass against numerical gradients...")
    lstm = LSTM(input_size, hidden_size, output_size, seed=0)
    ok_lstm = check_model(lstm, X, y, ["Wih", "Whh", "b", "Why", "by"], label="LSTM")

    print()
    if ok_rnn and ok_lstm:
        print("ALL GRADIENT CHECKS PASSED")
        sys.exit(0)
    else:
        print("SOME GRADIENT CHECKS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
