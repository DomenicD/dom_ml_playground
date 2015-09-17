import sys
import timeit
import numpy
import theano
import theano.tensor as T


class NeuralNode(object):
    def __init__(self, incoming, bias, outgoing, activationFn):                
        self.incoming = incoming
        self.bias = bias
        self.outgoing = outgoing
        self.activationFn = activationFn
        self.threshold = 0.0
        self.weights = [0] * len(self.incoming)
        self.weighedAverage = 0.0        
        self.output = 0.0
        
    def readIncoming(self):        
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
            layer.append(NeuralNode()) 



"""
    http://www.iro.umontreal.ca/~bengioy/dlbook/mlp.html#pf2
    f?(x) = b + V sigmoid(c + W x),
"""