import math
from code.src.neurons.activation_functions.activation_function import ActivationFunction



class SigmoidActivation(ActivationFunction):
    def __init__(self):
        ActivationFunction.__init__(
            self,
            lambda x: sigmoid(x),
            # This could be optimized as the derivative
            # will always be called after the fn.
            # And will be called with the same x value.
            lambda x: sigmoid(x) * (1 - sigmoid(x)))


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))