import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def bce_with_logits(logits, targets):
    """
    Numerically stable binary cross-entropy from raw logits.
    logits, targets: (B, 1)
    Returns: scalar loss (mean over batch), dL/dlogits (B, 1)
    """
    B = logits.shape[0]
    # stable BCE-with-logits: max(x,0) - x*y + log(1+exp(-|x|))
    x = logits
    y = targets
    loss = np.maximum(x, 0) - x * y + np.log1p(np.exp(-np.abs(x)))
    loss = loss.mean()

    p = sigmoid(x)
    dlogits = (p - y) / B
    return loss, dlogits


def accuracy(logits, targets):
    preds = (sigmoid(logits) > 0.5).astype(np.float64)
    return float((preds == targets).mean())
