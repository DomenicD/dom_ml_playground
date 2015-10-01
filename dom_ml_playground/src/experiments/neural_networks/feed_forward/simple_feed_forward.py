import math
import sys
import numpy
from src.experiments.neural_networks.neurons import ReceiveAllNeuron


class NeuralNetwork(object):
    def __init__(self):
        self.layers = []
    
    @property
    def inputs(self):
        nLayers = self._layers.count
        if nLayers > 0: return self.layers[0]
        else: return []        


    @property
    def outputs(self):
        nLayers = self.layers.count
        if nLayers > 0: return self.layers[nLayers - 1]
        else: return []        


    def addLayer(self, layer):
        self.layers.append(layer)

    def createLayer(self, count, neuronConstructor):
        return [neuronConstructor() for n in range(count)]
        
    def prepairForInput(self): pass



class SimpleFeedForwardNN(NeuralNetwork):
    def __init__(self, normalizer, structure = [1, 5, 1]):
        # Call inherited class.
        NeuralNetwork.__init__(self)
        lastLayerIndex = len(structure) - 1

        for i in range(structure):
            nNeurons = structure[i]
            neuronConstructor = None
            if i == 0:
                neuronConstructor = (
                    lambda : ReceiveAllNeuron(normalizer.input))
            elif i == lastLayerIndex:
                neuronConstructor = (
                    lambda : ReceiveAllNeuron(normalizer.output))
            else:
                neuronConstructor = (
                    lambda : ReceiveAllNeuron(sigmoid))

            self.addLayer(
                self.createLayer(nNeurons, neuronConstructor))
            
            self.connectNeurons()

    
    def connectNeurons(self):
        # Connect the neurons
        for iLayer in range(self.layers.count - 1):
            for sender in self.layers[iLayer]:
                for reciever in self.layers[iLayer + 1]:
                    sender.connectTo(reciever)


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(x))

class Normalizer(object):
    def __init__(self, range, offset):
        self.range = range
        self.offset = offset
    
    
    def input(self, x):
        return (x + self.offset) / self.range
    
        
    def output(self, x):
        return (x * range) - offset


if (__name__ == '__main__'):
    net = SimpleFeedForwardNN(normalizers(2, 1))
    testFn = lambda x: math.sin(x)

# TASK
# 1) [Done] Implement a simple feed forward neural network
# 2) Implement a backpropagation training algorithm for the simple network
# 3) Train simple network to represent a simple forth degree quadratic function


"""
    http://www.iro.umontreal.ca/~bengioy/dlbook/mlp.html#pf2
    f?(x) = b + V sigmoid(c + W x),
"""