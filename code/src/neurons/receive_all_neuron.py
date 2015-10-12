from code.src.neurons.neurons import Neuron
from code.src.neurons.activation_functions.sigmoid_activation import SigmoidActivation



# This type is like most models. It requires all of the
# signals to have been received before sending any of
# its connections.
class ReceiveAllNeuron(Neuron):
    def __init__(self, activation = SigmoidActivation()):
        # Call inherited class.
        Neuron.__init__(self, activation)
        self.signal_receivedTracker = {}


    def on_reset(self):
        # gets the neruon ready to fire again
        for key in self.signal_receivedTracker:
            self.signal_receivedTracker[key] = False


    def on_inbound_added(self, connection):
        self.signal_receivedTracker[connection.id] = False


    def on_inbound_removed(self, connection):
        self.signal_receivedTracker.pop(connection.id, None)


    def on_signal_received(self, value, connection_id):
        # If all signals have been received, the output is set
        # and the node fires.         
        if (len(self.signal_receivedTracker) and 
            self.signal_receivedTracker[connection_id]): return

        if (connection_id):
            self.signal_receivedTracker[connection_id] = True

        if (not self.allSignalsReceived()): return
               
        self.output = self.activation.fn(self.accumulated_input_signals)
        self.fire()        


    def allSignalsReceived(self):
        return all(self.signal_receivedTracker.values())
