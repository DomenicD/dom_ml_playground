from code.src.neurons.neurons import Neuron
from code.src.neurons.activation_functions.sigmoid_activation import SigmoidActivation



# This type is like most models. It requires all of the
# signals to have been received before sending any of
# its connections.
class ReceiveAllNeuron(Neuron):
    def __init__(self, activation = SigmoidActivation()):
        # Call inherited class.
        Neuron.__init__(self, activation)
        self.signalReceivedTracker = {}


    def onReset(self):
        # gets the neruon ready to fire again
        for key in self.signalReceivedTracker:
            self.signalReceivedTracker[key] = False


    def onInboundAdded(self, connection):
        self.signalReceivedTracker[connection.id] = False


    def onInboundRemoved(self, connection):
        self.signalReceivedTracker.pop(connection.id, None)


    def onSignalReceived(self, value, connectionId):
        # If all signals have been received, the output is set
        # and the node fires.         
        if (len(self.signalReceivedTracker) and 
            self.signalReceivedTracker[connectionId]): return

        if (connectionId):
            self.signalReceivedTracker[connectionId] = True

        if (not self.allSignalsReceived()): return
               
        self.output = self.activation.fn(self.accumulatedInputSignals)
        self.fire()        


    def allSignalsReceived(self):
        return all(self.signalReceivedTracker.values())