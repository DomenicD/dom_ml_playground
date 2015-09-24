﻿from abc import ABCMeta, abstractmethod
import math
import sys
import uuid
import timeit
import numpy
# import theano
# import theano.tensor as T


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
    
    def __init__(self, activationFn, bias = 0.0):
        self.bias = bias
        self.activationFn = activationFn
        self.outConnections = set()
        self.inConnections = set()
        self.accumulatedSignals = 0.0        
        self.output = 0.0


    def fire(self):
        for c in self.outConnections:
            c.sendSignal(c.output)


    # This creates a one way connection from this neuron to the passed in one.
    def connectTo(self, node):
        connection = NeuronConnection(self, node)
        self.addOutbound(connection)
        node.addInbound(connection)
        self.onOutboundAdded(connection)

    def addOutbound(self, connection):
        self.outConnections.add(connection)
        self.onOutboundAdded(connection)


    def addInbound(self, connection):
        self.inConnections.add(connection)
        self.onInboundAdded(connection)


    def removeConnection(self, connection):
        connection.sender.removeOutbound(connection)
        connection.receiver.removeInbound(connection)


    def removeOutbound(self, connection):
        self.outConnections.remove(connection)
        self.onOutboundRemoved(connection)


    def removeInbound(self, connection):
        self.inConnections.remove(connection)
        self.onInboundRemoved(connection)


    def receiveSignal(self, connectionId, value):
        self.accumulatedSignals += value
        self.onSignalReceived(connectionId, value)


    def onInboundAdded(self, connection): pass


    def onInboundRemoved(self, connection): pass


    def onOutboundAdded(self, connection): pass


    def onOutboundRemoved(self, connection): pass

    
    def onSignalReceived(self, connectionId, value): pass

    
        
    
# This type is like most models. It requires all of the signals to have been received
# before sending any of its connections.
class ReceiveAllNeuron(Neuron):
    def __init__(self, activationFn, bias = 0.0):
        # Call inherited class.
        Neuron.__init__(self, activationFn, bias)
        self.weighedAverage = 0.0
        self.signalReceivedTracker = {}


    def reset(self):
        # gets the neruon ready to fire again
        self.weighedAverage = 0.0
        for key in self.signalReceivedTracker:
            self.signalReceivedTracker[key] = False


    def onInboundConnectionAdded(self, connection):
        self.signalReceivedTracker[connection.id] = False


    def onInboundRemoved(self, connection):
        self.signalReceivedTracker.pop(connection.id, None)


    def onSignalReceived(self, connectionId, value):
        # If all signals have been received, the output is set
        # and the node fires.         
        if (self.signalReceivedTracker[connectionId]): return

        self.signalReceivedTracker[connectionId] = True
        self.weighedAverage += value

        if (not self.allSignalsReceived()): return
               
        self.output = self.activationFn(self.bias + self.weighedAverage)
        self.fire()        


    def allSignalsReceived(self):
        return all(self.signalReceivedTracker.values())



class NetConfig(object):
    def __init__(self):
        self.inputCount = 0
        self.hiddenCount = 0
        self.outputCount = 0
        self.inputActivationFn = None # TODO: choose a default
        self.hiddenActivationFn = None # TODO: choose a default
        self.outputActivationFn = None # TODO: choose a default


class SimpleFeedForwardNN(object):
    def __init__(self, normalizers, nHidden = 5):
        self.input = ReceiveAllNeuron(normalizers['in'])
        # TODO: Implement the activation functions using your own methods, then
        # make one that uses the theano methods.
        self.hidden = [ReceiveAllNeuron(sigmoid) for n in range(nHidden)]
        self.output = ReceiveAllNeuron(normalizers['out'])

        # Connect the neurons
        for h in self.hidden:
            self.input.connectTo(h)
            h.connectTo(self.output)


def sigmoid(x): 1.0 / (1.0 + math.exp(x))

def normalizers(range, offset):
    return {
        'in': lambda x: (x + offset) / range,
        'out': lambda x: (x * range) - offset
        }


if (__name__ == '__main__'):
    net = SimpleFeedForwardNN(normalizers(2, 1))
    testFn = lambda x: math.sin(x)

# TASK
# 1) Implement a simple feed forward neural network
# 2) Implement a backpropagation training algorithm for the simple network
# 3) Train the network to represent a simple forth degree quadratic function


"""
    http://www.iro.umontreal.ca/~bengioy/dlbook/mlp.html#pf2
    f?(x) = b + V sigmoid(c + W x),
"""