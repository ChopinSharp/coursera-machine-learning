import numpy as np
from src.ex2 import cost_function

def map_feature(x1, x2):
    """
    Map x1, x2 to [1, x1, x2, x1^2, x1x2, x2^2, x1^3, ... , x2^6].
    
    Inputs:
    - X: Of shape [N, 2].
    
    Returns:
    - feature: Of shape [N, 28].
    """
    N = x1.shape[0]
    feature = np.zeros((N, 28))
    feature[:,  0] = 1
    feature[:,  1] = x1
    feature[:,  2] = x2
    feature[:,  3] = x1 * x1
    feature[:,  4] = x1 * x2
    feature[:,  5] = x2 * x2
    feature[:,  6] = x1 ** 3 
    feature[:,  7] = x1 ** 2 * x2 ** 1
    feature[:,  8] = x1 ** 1 * x2 ** 2
    feature[:,  9] = x2 ** 3
    feature[:, 10] = x1 ** 4
    feature[:, 11] = x1 ** 3 * x2 ** 1
    feature[:, 12] = x1 ** 2 * x2 ** 2
    feature[:, 13] = x1 ** 1 * x2 ** 3
    feature[:, 14] = x2 ** 4
    feature[:, 15] = x1 ** 5
    feature[:, 16] = x1 ** 4 * x2 ** 1
    feature[:, 17] = x1 ** 3 * x2 ** 2
    feature[:, 18] = x1 ** 2 * x2 ** 3
    feature[:, 19] = x1 ** 1 * x2 ** 4
    feature[:, 20] = x2 ** 5
    feature[:, 21] = x1 ** 6
    feature[:, 22] = x1 ** 5 * x2 ** 1
    feature[:, 23] = x1 ** 4 * x2 ** 2
    feature[:, 24] = x1 ** 3 * x2 ** 3
    feature[:, 25] = x1 ** 2 * x2 ** 4
    feature[:, 26] = x1 ** 1 * x2 ** 5
    feature[:, 27] = x2 ** 6
    return feature


def cost_function_reg(theta, X, y, L):
    """
    Mostly the same as src.ex2.cost_function.
    
    New Inputs:
    - L: lambda which is the regularization parameter.
    """
    cost, grad = cost_function(theta, X, y)
    N = X.shape[0]
    cost += L / (2 * N) * np.sum(theta[1:] * theta[1:]) # no theta_0
    grad[1:] += L / N * theta[1:]
    return cost, grad
    
    
    