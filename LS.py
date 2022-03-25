import numpy as np

# Cost function
def LScost(X, y, w):
    return sum(np.square(np.subtract(y, np.matmul(X,w))))

# Fitting weights
def LSfit(X,y):
    return np.asarray(np.dot(np.asmatrix(np.dot(X.T, X)).I, np.dot(X.T, y)))