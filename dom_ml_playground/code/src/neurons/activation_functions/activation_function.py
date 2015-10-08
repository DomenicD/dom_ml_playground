class ActivationFunction(object):
    def __init__(self, fn, derivative):
        self.fn = fn
        self.derivative = derivative