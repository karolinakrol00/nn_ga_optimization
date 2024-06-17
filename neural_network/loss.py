import numpy as np

def mse(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2))

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true-y_pred))

def binary_cross_entropy(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
