import torch
import numpy as np


def fit_linear_regression(X, y, lr=0.1, steps=500):
    """Fit y ~= X @ w + b with full-batch GD using only autograd.

    Args:
        X: Float tensor (N, D)
        y: Float tensor (N,) or (N, 1)
        lr: learning rate
        steps: number of gradient descent iterations

    Returns:
        w: Float tensor (D,) learned weights (no grad)
        b: Float tensor scalar learned bias (no grad)
    """
    # TODO: ensure y is shape (N,)
    # TODO: initialize w (D,) and b with requires_grad=True
    # TODO: for each step:
    #   - predict, compute MSE loss
    #   - loss.backward()
    #   - manual GD update under torch.no_grad()
    #   - zero gradients
    # TODO: return detached w, b
    w = (torch.randn(X.shape[1]) * 0.1).requires_grad_(True)
    b = torch.tensor(0.0, requires_grad=True)
    for _ in range(steps):
        w.grad, b.grad = None, None
        y_pred = X @ w + b 
        loss = ((y-y_pred)**2).mean()
        loss.backward()
        print(loss)
        print(y_pred)
        print(w,b,w.grad, b.grad)
        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad

    return w.detach(), b.detach()
