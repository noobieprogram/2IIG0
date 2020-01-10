import math
import numpy as np
from random import random


class HiddenLayer:
    """
    Represents a hidden layer in the network. It maintains the neurons 
    in it and the activation values for those neurons. 
    Incoming values is a property of the neural layer as they are universal
    to all neurons in the layer.
    """

    def __init__(self, incoming, n=10, activation="relu"):
        # initialize n neurons
        self.neurons = [Neuron(incoming, activation) for i in range(n)]
        self.input = incoming
        self.activation_values = 0

    def set_activation_values(self):
        self.activation_values = [neuron.activation_value for neuron in self.neurons]

    def get_activation_values(self):
        return self.activation_values


class Neuron:
    def __init__(self, incoming, activation, bias=0):
        """
        Each neuron has a certain bias and maintains a set of weights
        corresponding to the each incoming connection. Weights is randomly initialized.
        """
        self.bias = bias
        self.input = incoming
        # random weights for now
        self.weights = [random() for i in range(len(incoming))]
        # value before activation function
        self.value = self.calculate_value(incoming)
        # activation value
        self.activation_value = self.calculate_activation(activation)

    def calculate_value(self, incoming):
        return sum(np.multiply(self.weights, incoming)) - self.bias

    def calculate_activation(self, activation):
        # we decided to use relu as the activation function but this code is just
        # to help scalability
        if activation == "relu":
            return Neuron.relu(self.value)
        elif activation == "sigmoid":
            return Neuron.sigmoid(self.value)

    @classmethod
    def relu(cls, value):
        return max(0.01*value, value)

    @classmethod
    def sigmoid(cls, x):
        return float(1 / (1 + math.exp(-x)))
