import sys
import numpy
from code.src.data_processing.normalizer import Normalizer
from code.src.neural_networks.neural_network import NeuralNetwork
from code.src.neurons.receive_all_neuron import ReceiveAllNeuron



class FeedForwardNN(NeuralNetwork):
    def __init__(self, normalizer = Normalizer(), structure = [1, 5, 1]):        
        NeuralNetwork.__init__(self) # Call inherited class.
        self.normalizer = normalizer

        num_layers = len(structure)
        last_layer_index = num_layers - 1

        for i in range(num_layers):
            num_neurons = structure[i]
            neuron_constructor = None
            # May offer ability to configure activation functions.
            if i == 0: 
                neuron_constructor = (
                    lambda : ReceiveAllNeuron())
            elif i == last_layer_index:
                neuron_constructor = (
                    lambda : ReceiveAllNeuron())
            else:
                neuron_constructor = (
                    lambda : ReceiveAllNeuron())

            self.add_layer(
                self.create_layer(num_neurons, neuron_constructor))
            
            self.connect_neurons()

    
    def connect_neurons(self):
        # Connect the neurons
        for iLayer in range(self.num_layers - 1):
            for sender in self.layers[iLayer]:
                for reciever in self.layers[iLayer + 1]:
                    sender.connect_to(reciever)


    def prepair_for_input(self):
        for neuron in self.neurons:
            neuron.reset()


    def receive_inputs(self, inputs):
        self.prepair_for_input()
        for i in range(len(inputs)):
            input = inputs[i]
            neuron = self.input_layer[i]
            neuron.receive_signal(self.normalizer.norm_input(input))

        return [self.normalizer.norm_output(output.output)
                for output in self.output_layer]



# TASK
# 1) [Done] Implement a simple feed forward neural network
# 2) Implement a backpropagation training algorithm for the simple network
# 3) Train simple network to represent a simple forth degree quadratic function


"""
    http://www.iro.umontreal.ca/~bengioy/dlbook/mlp.html#pf2
    f?(x) = b + V sigmoid(c + W x),
"""