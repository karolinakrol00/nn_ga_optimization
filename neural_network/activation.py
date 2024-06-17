import numpy as np

def tanh(x):
    return np.tanh(x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def ReLU(x):
    return np.maximum(x, 0)

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)