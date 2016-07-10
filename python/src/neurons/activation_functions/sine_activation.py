import math
from python.src.neurons.activation_functions.activation_function import ActivationFunction



class SineActivation(ActivationFunction):
    def __init__(self):
        ActivationFunction.__init__(
            self,
            lambda x: x*math.sin(x),
            # This could be optimized as the derivative
            # will always be called after the fn.
            # And will be called with the same x value.
            lambda x: x*math.cos(x) + math.sin(x))
