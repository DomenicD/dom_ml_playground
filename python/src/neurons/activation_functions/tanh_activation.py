import math
from python.src.neurons.activation_functions.activation_function import ActivationFunction



class TanhActivation(ActivationFunction):
    def __init__(self):
        ActivationFunction.__init__(
            self,
            lambda x: math.tanh(x),
            # This could be optimized as the derivative
            # will always be called after the fn.
            # And will be called with the same x value.
            lambda x: 1 - math.pow(math.tanh(x), 2))
