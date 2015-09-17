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
        
        self.output = self.activationFn(self.bias.output +
                                        self.weighedAverage)


"""
    f?(x) = b + V sigmoid(c + W x),
"""