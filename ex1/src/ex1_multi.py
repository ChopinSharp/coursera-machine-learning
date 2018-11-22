import numpy as np
from numpy.linalg import pinv, inv
from src.ex1 import compute_cost


def feature_normalize(X):
    """
    Normalize input feature matrix X.
    
    Inputs:
    - X: A numpy array of shape (N, M), feature matrix to normalize.
         N is the number of the feature vectors, M is the dimension of each 
         feature vector.
    
    Returns:
    - X_normalized: Normalized feature matrix with shape (N, M).
    - mu: Average of the input feature matrix X with shape (1, M).
    - sigma: Standard derivation of the input feature matrix X with shape (1, M).
    """
    mu = np.mean(X, axis=0, keepdims=True)
    sigma = np.std(X, axis=0, keepdims=True)
    X_normalized = (X - mu) / sigma
    return X_normalized, mu, sigma


def gradient_descent_multi(X, y, theta, alpha, num_iters):
    """
    Perform num_iters rounds of gradient descent on parameter theta with
    alpha as learning rate.
    
    Inputs:
    - X: A numpy array of shape (N, M) where N is the scale of training data.
    - y: A numpy array of shape (N, 1).
    - theta: A numpy array of shape (M, 1) which represents a column vector.
    - alpha: Learning rate.
    - num_iters: Rounds of training.
    
    Returns:
    - theta: The parameter vector after training.
    - history_J: A list of the cost J during training.
    """
    theta = theta.copy()
    # dtheta = np.zeros_like(theta)
    history_J = []
    for i in range(num_iters):
        diff = X @ theta - y
        dtheta = np.transpose(np.mean(X * diff, axis=0, keepdims=True))
        theta -= alpha * dtheta
        history_J.append(compute_cost(X, y, theta))
    return theta, history_J


def normal_equation(X, y):
    """
    Compute theta using normal equation.
    
    Inputs:
    - X: A numpy array of shape (N, M) where N is the scale of training data.
    - y: A numpy array of shape (N, 1).
    
    Returns:
    - theta: The parameter vector computed.
    """
    theta = pinv(X) @ y
    return theta
