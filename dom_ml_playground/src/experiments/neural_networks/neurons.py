import math
import uuid
from enum import Enum

class ActivationFunction(object):
    def __init__(self, fn, derivative):
        self.fn = fn
        self.derivative = derivative


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

class Normalizer(object):
    def __init__(self, range, offset):
        self.input = ActivationFunction(
            lambda x: (x + offset) / range,
            lambda x: x)

        self.output = ActivationFunction(
            lambda x: (x * range) - offset,
            lambda x: x)
    


class NeuronConnection(object):
    def __init__(self, sender, receiver):
        self.id = uuid.uuid4().hex
        self.sender = sender
        self.receiver = receiver
        self._weight = 1.0
        self.weight = self._weight
        self.signalSent = 0.0
        self.signalReceived = 0.0
        

    @property
    def weight(self):
        return self._weight


    @weight.setter
    def weight(self, value):
        self._prior_weight = self.weight
        self._weight = value


    @property
    def prior_weight(self):
        return self._prior_weight


    @property
    def weight_difference(self):
        return self.weight - self.prior_weight


    @property
    def weight_change(self):
        return weight_difference / self.prior_weight


    def sendSignal(self, value):
        self.signalReceived = value
        self.signalSent = value * self.weight        
        self.receiver.receiveSignal(self.signalSent, self.id)

    def disconnect(self):
        self.sender.removeOutbound(self)
        self.receiver.removeInbound(self)


class NeuronType(Enum):
    UNKNOWN = 0,
    INPUT = 1,
    OUTPUT = 2,
    HIDDEN = 3


class Neuron(object):
    # __metaclass__ = ABCMeta
    
    def __init__(self, activation):
        self.id = uuid.uuid4().hex        
        self.activation = activation
        self.outConnections = set()
        self.inConnections = set()
        self.accumulatedInputSignals = 0.0
        self.output = 0.0
        self.error = 0.0


    @property
    def out_connections_list(self):
        return list(self.outConnections)


    @property
    def in_connections_list(self):
        return list(self.inConnections)


    @property
    def type(self):
        if (len(self.outConnections) == 0 and
            len(self.inConnections) == 0):
            return NeuronType.UNKNOWN

        if len(self.outConnections) == 0:
            return NeuronType.OUTPUT

        if len(self.inConnections) == 0:
            return NeuronType.INPUT

        return NeuronType.HIDDEN


    def fire(self):
        for c in self.outConnections:
            c.sendSignal(self.output)


    # This creates a one way connection from this neuron to the passed in one.
    def connectTo(self, node):
        connection = NeuronConnection(self, node)
        self.addOutbound(connection)
        node.addInbound(connection)        
        return connection    


    def addOutbound(self, connection):
        self.outConnections.add(connection)
        self.onOutboundAdded(connection)


    def addInbound(self, connection):
        self.inConnections.add(connection)
        self.onInboundAdded(connection)    


    def removeOutbound(self, connection):
        self.outConnections.remove(connection)
        self.onOutboundRemoved(connection)


    def removeInbound(self, connection):
        self.inConnections.remove(connection)
        self.onInboundRemoved(connection)


    def receiveSignal(self, value, connectionId = None):
        self.accumulatedInputSignals += value
        self.onSignalReceived(value, connectionId)

    def reset(self):
        self.accumulatedInputSignals = 0.0
        self.output = 0.0
        self.error = 0.0
        self.onReset()

    # Overridable event methods
    def onInboundAdded(self, connection): pass
    def onInboundRemoved(self, connection): pass
    def onOutboundAdded(self, connection): pass
    def onOutboundRemoved(self, connection): pass
    def onSignalReceived(self, value, connectionId): pass
    def onReset(self): pass


            
    
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