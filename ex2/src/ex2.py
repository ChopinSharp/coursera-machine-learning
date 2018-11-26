import numpy as np

def sigmoid(z):
    """
    Implement sigmoid function.
    
    Inputs:
    - z: A numpy array of arbitrary shape.
    
    Returns:
    - g: Result of sigmoid function.
    """
    g = 1 / (1 + np.exp(-z))
    return g


def cost_function(theta, X, y):
    """
    Compute loss for logistic regression.
    
    Inputs:
    - theta: Parameters of shape (M, ).
    - X: Data points.
    - y: Labels.
    
    Returns:
    - J: Cost value.
    - grad: Gradient of theta of shape (M, ).
    """
    h = sigmoid(X @ theta.reshape(-1, 1))
    cost = -np.mean(y * np.log(h) + (1 - y) * np.log(1 - h))
    grad = np.mean((h - y) * X, axis=0, keepdims=True).reshape(-1)
    return cost, grad
