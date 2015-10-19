from code.src.neural_networks.neural_network_utils import NeuralNetworkUtils
from code.src.neurons.neurons import NeuronType
import math
import random


# http://home.agh.edu.pl/~vlsi/AI/backp_t_en/backprop.html
# http://www.cse.unsw.edu.au/~cs9417ml/MLP2/


class TrainingResult(object):
    def __init__(self, epochs, error):
        self.epochs = epochs
        self.error = error



# wonder how to compute the contribution of each node
# during feed forward phase?
class Backpropagator(object):
    def teach(self, neural_network, expectations,
              acceptable_error = .001, max_iterations = 100,
              time_limit = None, learning_rate = 0.5, callback_rate = 10,
              callback_func = None):
        epochs = 0
        error = 0.0
        within_acceptable_error = False
        
        while (epochs < max_iterations and not within_acceptable_error):
            sample = list(expectations)
            #random.shuffle(sample)
            for expectation in sample:                
                self.learn(neural_network, expectation, learning_rate)

            epochs += 1

            if epochs % callback_rate == 0:
                error = 0.0
                for exp in expectations:
                    error += self.calculate_error(
                        neural_network.receive_inputs(exp.inputs),
                        exp.outputs)
                within_acceptable_error = error < acceptable_error
                if callback_func != None:
                    callback_func(neural_network, expectations,
                                  TrainingResult(epochs, error))
            
        return TrainingResult(epochs, error)


    def calculate_error(self, actual, correct):
        errors = [math.pow(actual[i] - correct[i], 2)
                  for i in range(len(actual))]
        
        return sum(errors)


    def learn(self, neural_network, expectation, learning_rate):
        """This is the actual backpropagation part of the code
        here is where we will perform one propagation iteration
        adjusting the networks's node weights
        """        

        neural_network.receive_inputs(expectation.inputs)
        denorm_expectation = [
            neural_network.normalizer.denorm_output(exp_out)
            for exp_out in expectation.outputs]

        NeuralNetworkUtils.OutputBreadthTraversal(
            neural_network,
            lambda neuron: self.propagate_errors(
                neural_network, neuron, denorm_expectation, learning_rate))


    def propagate_errors(self, network, neuron, expectations,
                         learning_rate):     
           
        if neuron.type == NeuronType.OUTPUT:
            index = network.neuron_index(neuron).index
            expectation = expectations[index]
            neuron.error = self.output_error(
                neuron, expectation)
        else:
            # Calculate the error for the current layer
            neuron.error = self.hidden_error(neuron)

            # Adjust the weights of the prior layer
            for outbound in neuron.out_connections:
                self.adjust_weight(outbound, learning_rate)
    

    
    def adjust_weight(self, connection, learning_rate):        
        connection.weight += (learning_rate * 
                              connection.receiver.error * 
                              connection.signal_received)
        

    def output_error(self, neuron, expectation):        
        return ((expectation - neuron.output) * 
                neuron.activation.derivative(
                    neuron.accumulated_input_signals))


    def hidden_error(self, neuron):
        weightedErrorSum = reduce(
            lambda sum, c: sum + c.weight * c.receiver.error, 
            neuron.out_connections, 0.0)

        return (weightedErrorSum *
                neuron.activation.derivative(
                    neuron.accumulated_input_signals))

    