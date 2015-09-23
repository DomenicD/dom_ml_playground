from abc import ABCMeta, abstractmethod
import sys
import uuid
import timeit
import numpy
import theano
import theano.tensor as T


class NeuronConnection(object):
    def __init__(self, sender, receiver):
        self.id = uuid.uuid4()
        self.sender = sender
        self.receiver = receiver
        self.weight = 0.0
        self.signalSent = 0.0
        self.signalReceived = 0.0

    def sendSignal(self, value):
        self.signalSent = value
        self.signalReceived = self.weight * value
        self.receiver.receiveSignal(self.id, self.signalReceived)



class Neuron(object):
    # __metaclass__ = ABCMeta
    
    def __init__(self, activationFn, bias):
        self.bias = bias
        self.activationFn = activationFn
        self.connections = set()
        self.accumulatedSignals = 0.0        
        self.output = 0.0

    def fire(self):
        for c in self.connections:
            c.sendSignal(c.output)

    # This creates a one way connection from this neuron to the passed in one.
    def connectTo(self, node):
        connection = NeuronConnection(self, node)
        self.connections.add(connection)
        self.onConnectionAdded(connection)

        # This creates a one way connection from this neuron to the passed in one.
    def removeConnection(self, connection):
        self.connections.remove(connection)
        self.onConnectionRemoved(connection)

    def receiveSignal(self, connectionId, value):
        self.accumulatedSignals += value
        self.onSignalReceived(connectionId, value)

    def onConnectionAdded(self, connection): pass

    def onConnectionRemoved(self, connection): pass
    
    def onSignalReceived(self, connectionId, value): pass

    
        
    
# This type is like most models. It requires all of the signals to have been received
# before sending any of its connections.
class ReceiveAllNeuron(Neuron):
    def __init__(self, activationFn, bias):
        self.weighedAverage = 0.0
        self.signalReceivedMap = {}


    def reset(self):
        # gets the neruon ready to fire again
        self.weighedAverage = 0.0
        for key in self.signalReceivedMap:
            self.signalReceivedMap[key] = False


    def onConnectionAdded(self, connection):
        self.signalReceivedMap[connection.id] = False


    def onConnectionRemoved(self, connection):
        self.signalReceivedMap.pop(connection.id, None)


    def onSignalReceived(self, connectionId, value):
        # If all signals have been received, the output is set
        # and the node fires.         
        if (self.signalReceivedMap[connectionId]): return

        self.signalReceivedMap[connectionId] = True
        self.weighedAverage += value

        if (not self.allSignalsReceived()): return
               
        self.output = self.activationFn(self.bias + self.weighedAverage)
        self.fire()        


    def allSignalsReceived(self):
        return all(self.signalReceivedMap.values())



class NetConfig(object):
    def __init__(self):
        self.inputCount = 0
        self.hiddenCount = 0
        self.outputCount = 0
        self.inputActivationFn = None # TODO: choose a default
        self.hiddenActivationFn = None # TODO: choose a default
        self.outputActivationFn = None # TODO: choose a default


class SimpleFeedForwardNN(object):
    def __init__(self, config):
        self.inputLayer = self.createLayer(config.inputCount,
                                           config.inputActivationFn)

        self.hiddenLayer = self.createLayer(config.hiddenCount,
                                           config.hiddenActivationFn)

        self.outputLayer = self.createLayer(config.outputCount,
                                           config.outputActivationFn)

    def createLayer(self, count, activationFn):
        layer = []
        for i in range(count):
            # TODO: Need to have a special input node that has no
            #       incoming nodes or bias value.
            layer.append(Neuron()) 



"""
    http://www.iro.umontreal.ca/~bengioy/dlbook/mlp.html#pf2
    f?(x) = b + V sigmoid(c + W x),
"""