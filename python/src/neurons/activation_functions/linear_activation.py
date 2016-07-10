from python.src.neurons.activation_functions.activation_function import ActivationFunction



class LinearActivation(ActivationFunction):
    def __init__(self):
        ActivationFunction.__init__(
            self, lambda x: x, lambda x: 1.0)
