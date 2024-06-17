import math as mt
import numpy as np

class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward_propagation(self, input):
        raise NotImplementedError
    
class FCLayer(Layer):

    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.normal(0, 1, size = (input_size, output_size)) 
        self.bias = np.random.normal(0, 1, size = (1, output_size))

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output
    
class ActivationLayer(Layer):
    def __init__(self, activation):
        self.activation = activation

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output