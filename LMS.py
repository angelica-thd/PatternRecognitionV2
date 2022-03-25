import numpy as np

# Cost function
def LMScost(X, y, w):
    return 1/(y.shape[0]) * sum((np.subtract(y, np.matmul(X,w))) ** 2)

# Fitting weights
def LMSfit(X,y):
    R = np.matmul(np.transpose(X), X).astype(np.float64) / X.shape[0]
    return np.asarray(np.matmul( np.asmatrix(R).I, np.matmul(np.transpose(X), y) / X.shape[0]))
