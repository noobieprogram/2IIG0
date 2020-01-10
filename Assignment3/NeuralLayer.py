from random import random


class NeuralLayer:
    num_neurons = 10
    activation = 'relu'

    def __index__(self):
        neurons = [Neuron(10) for i in range(NeuralLayer.num_neurons)]

    def update(self, activation):
        pass


class Neuron:

    def __init__(self, incoming, bias=0):
        """
        Each neuron has a certain bias and maintains a set of weights
        corresponding to the each incoming connection. Weights is randomly initialized.
        """
        self.bias = bias
        # initialize random weights for all incoming connections
        self.weights = [random() for i in range(incoming)]
        self.value = 0

    def calculate_value(self, incoming_values) -> None:
        if len(incoming_values) != len(self.weights):
            raise Exception("Incorrect input provided")

        value = 0
        for i in range(incoming_values):
            value += self.weights[i]*incoming_values[i]

        self.value = value

    def calculate_activation(self, activation):
        # we decided to use relu as the activation function but this code is just
        # to help scalability
        if activation == 'relu':
            return Neuron.relu(self.value)

    @classmethod
    def relu(cls, value: int) -> float:
        return max(0.0, value)