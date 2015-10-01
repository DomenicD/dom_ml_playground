import uuid

class NeuronActivation(object):
    def __init__(self, fn, derivative):
        self.fn = fn
        self.derivative = derivative


class NeuronConnection(object):
    def __init__(self, sender, receiver):
        self.id = uuid.uuid4().hex
        self.sender = sender
        self.receiver = receiver
        self.weight = 1.0
        self.signalSent = 0.0
        self.signalReceived = 0.0


    def sendSignal(self, value):
        self.signalReceived = value
        self.signalSent = value * self.weight        
        self.receiver.receiveSignal(self.signalSent, self.id)

    def disconnect(self):
        self.sender.removeOutbound(self)
        self.receiver.removeInbound(self)



class Neuron(object):
    # __metaclass__ = ABCMeta
    
    def __init__(self, activation):
        self.id = uuid.uuid4().hex        
        self.activation = activation
        self.outConnections = set()
        self.inConnections = set()
        self.reset()


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


            
    
# This type is like most models. It requires all of the signals to have been received
# before sending any of its connections.
class ReceiveAllNeuron(Neuron):
    def __init__(self, activation):
        # Call inherited class.
        Neuron.__init__(self, activation)
        self.weighedAverage = 0.0
        self.signalReceivedTracker = {}


    def onReset(self):
        # gets the neruon ready to fire again
        self.weighedAverage = 0.0
        self.output = 0.0
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

        self.weighedAverage += value

        if (not self.allSignalsReceived()): return
               
        self.output = self.activation.fn(self.weighedAverage)
        self.fire()        


    def allSignalsReceived(self):
        return all(self.signalReceivedTracker.values())