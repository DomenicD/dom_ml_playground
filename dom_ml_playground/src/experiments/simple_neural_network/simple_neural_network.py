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
    __metaclass__ = ABCMeta
    
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

    def receiveSignal(self, connectionId, value):
        self.accumulatedSignals += value
        self.onSignalReceived(connectionId, value)

    @abstractmethod # TODO: Create a set of optionally overridable methods.
    def onSignalReceived(self, connectionId, value): pass       
        
    
# This type is like most models. It requires all of the signals to have been received
# before sending any of its connections.
class ReceiveAllNeuron(Neuron):
    def reset(self):
        pass
        # gets the neruon ready to fire again

    def onSignalReceived(self):    
        # TODO: After each signal is recieved, check to see if all the connections
        # have been recieved. If they have, then execute the code below.
        # note: don't need to compute weighedAverage anymore    
        self.weighedAverage = 0.0
        
        for nodeIndex in range(len(self.incoming)):
            self.weighedAverage += (self.weights[nodeIndex] *
                               self.incoming[nodeIndex].output)
        
        self.output = self.activationFn(self.bias + self.weighedAverage)



class NetConfig(object):
    def __init__(self):
        self.inputCount = 0
        self.hiddenCount = 0
        self.outputCount = 0
        self.inputActivationFn = None # TODO: choose a default
        self.hiddenActivationFn = None # TODO: choose a default
        self.outputActivationFn = None # TODO: choose a default

class SimpleNeuralNetwork(object):
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