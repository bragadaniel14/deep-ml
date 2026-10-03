import numpy as np

def train(X, y, W, b):
    """
    Train linear regression weights on standardized data.
    
    Args:
        X: numpy array of shape (n_samples, n_features) -- standardized features
        y: numpy array of shape (n_samples,) -- standardized targets
        W: numpy array of shape (n_features,) -- initial random weights
        b: float -- initial bias (0.0)
    
    Returns:
        W: numpy array of shape (n_features,) -- trained weights
        b: float -- trained bias
    """
    # TODO: implement your training strategy here
    # You can use ANY approach: gradient descent, normal equation,
    # momentum, adaptive learning rates, mini-batching, etc.
    n_sample, n_features = X.shape
    counter = 0
    max_steps = 1000
    lr = 1e-4
    momentum = 0.
    m_b = 0.
    m_W = np.zeros_like(W)
    eps = 1e-6
    while True:
        counter += 1
        loss = np.mean((y - X @ W - b)**2)
        dloss = 1.
        db = np.mean(-2 *(y - (X @ W) - b))  # scalar
        dW =  - 2 * X.T @ (y - X @ W - b) / n_sample # ((n_features, n_samples) @ n_sample,) ->(n_features, ) 
        if np.abs(db) < eps and np.max(np.abs(dW)) < eps:
            break
        m_b = 0.9 * m_b + 0.1 * db
        m_W = 0.9 * m_W + 0.1 * dW
        b -= m_b * lr
        W -= m_W * lr
        if counter > max_steps:
            break
    return W,b

