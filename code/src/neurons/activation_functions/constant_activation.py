from code.src.neurons.activation_functions.activation_function import ActivationFunction



class ConstantActivation(ActivationFunction):
    def __init__(self):
        ActivationFunction.__init__(self, lambda x: x, lambda x: 0)
