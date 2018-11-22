import numpy as np

def warm_up_exercise():
    """
    Build a 5x5 identity matrix.
    
    Returns:
    - m: A numpy array containing a 5x5 identity matrix.
    """
    m = np.eye(5)
    return m


def compute_cost(X, y, theta):
    """
    Compute MSE cost given X, y and parameter theta.
    
    Inputs:
    - X: A numpy array of shape (N, 2) where N is the scale of training data.
    - y: A numpy array of shape (N, 1).
    - theta: A numpy array of shape (2, 1) which represents a column vector.
    
    Returns:
    - cost: The MSE cost computed.
    """
    cost = np.mean((X @ theta - y) ** 2) / 2
    return cost


def gradient_descent(X, y, theta, alpha, num_iters):
    """
    Perform num_iters rounds of gradient descent on parameter theta with
    alpha as learning rate.
    
    Inputs:
    - X: A numpy array of shape (N, 2) where N is the scale of training data.
    - y: A numpy array of shape (N, 1).
    - theta: A numpy array of shape (2, 1) which represents a column vector.
    - alpha: Learning rate.
    - num_iters: Rounds of training.
    
    Returns:
    - theta: The parameter vector after training.
    - history_J: A list of the cost J during training.
    """
    theta = theta.copy()
    dtheta = np.zeros((2, 1))
    history_J = []
    for i in range(num_iters):
        diff = (X @ theta - y).reshape(-1)
        dtheta[0][0] = np.mean(diff)
        dtheta[1][0] = np.mean(diff * X[:, 1])
        theta -= alpha * dtheta
        history_J.append(compute_cost(X, y, theta))
    return theta, history_J
    