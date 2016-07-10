import sys
import numpy
from python.src.data_processing.normalizer import Normalizer
from python.src.neural_networks.neural_network import NeuralNetwork
from python.src.neurons.receive_all_neuron import ReceiveAllNeuron
from python.src.neurons.output_only_neuron import OutputOnlyNeuron
from python.src.neurons.activation_functions.linear_activation import LinearActivation



class FeedForwardNN(NeuralNetwork):
    def __init__(self, normalizer = Normalizer(), structure = [1, 5, 1],
                 has_bias_nodes = True, is_regression = False):        
        NeuralNetwork.__init__(self, normalizer)

        self.structure = structure
        self.has_bias_nodes = has_bias_nodes
        self.is_regression = is_regression
        self.bias_nodes = []

        num_layers = len(structure)
        last_layer_index = num_layers - 1

        for layer_index in range(num_layers):
            num_neurons = structure[layer_index]            

            if has_bias_nodes and layer_index != last_layer_index:
                num_neurons += 1

            self.add_layer(self.create_layer(
                num_neurons, lambda neuron_index: self.neuron_constructor(
                    last_layer_index, layer_index, neuron_index)))
            
        self.connect_neurons()


    @property
    def num_inputs(self):
        input_count = NeuralNetwork.num_inputs.fget(self)
        if self.has_bias_nodes:
            input_count -= 1
        return input_count


    def neuron_constructor(self, last_layer_index, current_layer_index,
                           neuron_index):
        
        if (self.has_bias_nodes and
            current_layer_index != last_layer_index and 
            neuron_index == self.structure[current_layer_index]):
            
            bias_node = OutputOnlyNeuron()
            self.bias_nodes.append(bias_node)
            return bias_node
        
        #if self.is_regression and current_layer_index == last_layer_index:
        #    return ReceiveAllNeuron(activation=LinearActivation())

        return ReceiveAllNeuron()


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
        if len(inputs) != self.num_inputs:
            raise ValueError("Inputs lenth must equal num_inputs") 
        
        self.prepair_for_input()

        if self.has_bias_nodes:
            for bias in self.bias_nodes:
                bias.receive_signal(1)
        
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