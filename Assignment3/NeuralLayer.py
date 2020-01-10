import numpy as np
from random import random


class NeuralLayer:
    """
    Represents a hidden layer in the network. It maintains the neurons 
    in it and the activation values for those neurons. 
    Incoming values is a property of the neural layer and not
    a neuron.
    """

    def __init__(self, incoming, n = 10, activation="relu"):
        # initialize n neurons
        self.neurons = [Neuron(len(incoming), activation) for i in range(n)]
        self.activation = activation
        self.incoming = incoming
        self.activation_values = 0

    def set_activation(self):
        self.activation_values = [neuron.activation for neuron in self.neurons]

    def get_activation(self):
        return self.activation_values

class Neuron:
    def __init__(self, incoming, activation, bias=0):
        """
        Each neuron has a certain bias and maintains a set of weights
        corresponding to the each incoming connection. Weights is randomly initialized.
        """
        self.bias = bias
        self.weights = [random() for i in range(incoming)]
        # value before activation function
        self.value = self.calculate_value(incoming)
        # activation value
        self.activation = self.activation(activation)

    def calculate_value(self, incoming) -> None:
        return sum(np.multiply(self.weights, incoming)) - self.bias

    def calculate_activation(self, activation) -> None:
        # we decided to use relu as the activation function but this code is just
        # to help scalability
        if activation == "relu":
            return Neuron.relu(self.value)

    @classmethod
    def relu(cls, value: int) -> float:
        return max(0.0, value)
