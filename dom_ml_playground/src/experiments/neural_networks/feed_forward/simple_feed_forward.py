import sys
import numpy
from src.experiments.neural_networks.neurons import (
    ReceiveAllNeuron, SigmoidActivation)
from src.experiments.neural_networks.neural_network import (
    NeuralNetwork)

class SimpleFeedForwardNN(NeuralNetwork):
    def __init__(self, normalizer, structure = [1, 5, 1]):
        # Call inherited class.
        NeuralNetwork.__init__(self)
        nLayers = len(structure)
        lastLayerIndex = nLayers - 1

        for i in range(nLayers):
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
                    lambda : ReceiveAllNeuron(SigmoidActivation()))

            self.addLayer(
                self.createLayer(nNeurons, neuronConstructor))
            
            self.connectNeurons()

    
    def connectNeurons(self):
        # Connect the neurons
        for iLayer in range(self.nLayers - 1):
            for sender in self.layers[iLayer]:
                for reciever in self.layers[iLayer + 1]:
                    sender.connectTo(reciever)


    def prepairForInput(self):
        for neuron in self.neurons:
            neuron.reset()


# TASK
# 1) [Done] Implement a simple feed forward neural network
# 2) Implement a backpropagation training algorithm for the simple network
# 3) Train simple network to represent a simple forth degree quadratic function


"""
    http://www.iro.umontreal.ca/~bengioy/dlbook/mlp.html#pf2
    f?(x) = b + V sigmoid(c + W x),
"""