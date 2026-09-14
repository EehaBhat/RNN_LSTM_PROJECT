"""
Correctness harness entry point (see Common Requirements #2).

Runs every correctness check in tests/ and prints a clear pass/fail summary.
Exit code is 0 iff everything that *could* run passed. The PyTorch
comparison (2.4) is skipped with a clear message if torch isn't installed;
it does not count as a failure in that case, but does if torch is installed
and outputs actually differ.

Usage: python run_tests.py
"""
import subprocess
import sys

CHECKS = [
    ("Gradient check: RNN + LSTM backward pass vs. finite differences",
     [sys.executable, "tests/test_gradcheck.py"]),
    ("Forward-pass equivalence: our LSTM vs. torch.nn.LSTMCell",
     [sys.executable, "tests/test_lstm_vs_pytorch.py"]),
]


def main():
    results = []
    for name, cmd in CHECKS:
        print("=" * 70)
        print(name)
        print("=" * 70)
        proc = subprocess.run(cmd)
        results.append((name, proc.returncode))
        print()

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    all_pass = True
    for name, code in results:
        status = "PASS" if code == 0 else "FAIL"
        if code != 0:
            all_pass = False
        print(f"[{status}] {name}")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
