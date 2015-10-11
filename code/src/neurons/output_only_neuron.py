from code.src.neurons.neurons import Neuron
from code.src.neurons.activation_functions.constant_activation import ConstantActivation




class OutputOnlyNeuron(Neuron):
    """This nueron has no inbound connections and always fires the same value.
    """
    def __init__(self, activation = ConstantActivation()):
        Neuron.__init__(self, activation)


    def add_inbound(self, connection):
        """Keep other neurons from connecting to this neuron.
        """
        return False


    def on_signal_received(self, value, connection_id):
        self.output = self.activation.fn(value)
        self.fire()
