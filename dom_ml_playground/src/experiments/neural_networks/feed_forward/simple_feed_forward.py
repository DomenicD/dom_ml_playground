import math
import sys
import numpy
from src.experiments.neural_networks.neurons import ReceiveAllNeuron


class SimpleFeedForwardNN(object):
    def __init__(self, normalizers, nHidden = 5):
        self.input = ReceiveAllNeuron(normalizers['in'])
        # TODO: Implement the activation functions using your own methods, then
        # make one that uses the theano methods.
        self.hidden = [ReceiveAllNeuron(sigmoid) for n in range(nHidden)]
        self.output = ReceiveAllNeuron(normalizers['out'])

        # Connect the neurons
        for h in self.hidden:
            self.input.connectTo(h)
            h.connectTo(self.output)


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(x))

def normalizers(range, offset):
    return {
        'in': lambda x: (x + offset) / range,
        'out': lambda x: (x * range) - offset
        }


if (__name__ == '__main__'):
    net = SimpleFeedForwardNN(normalizers(2, 1))
    testFn = lambda x: math.sin(x)

# TASK
# 1) [Done] Implement a simple feed forward neural network
# 2) Implement a backpropagation training algorithm for the simple network
# 3) Train simple network to represent a simple forth degree quadratic function


"""
    http://www.iro.umontreal.ca/~bengioy/dlbook/mlp.html#pf2
    f?(x) = b + V sigmoid(c + W x),
"""