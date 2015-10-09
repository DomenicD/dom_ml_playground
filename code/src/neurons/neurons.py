import math
import uuid
from enum import Enum    
from code.src.neurons.neuron_connection import NeuronConnection



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
