from code.src.neural_networks.neural_network_utils import NetworkUtils
from code.src.neurons.neurons import NeuronType
import math


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
              time_limit = None):
        epochs = 0
        within_acceptable_error = False
        error = 0.0
        while (epochs < max_iterations and not within_acceptable_error):
            error = 0.0
            for expectation in expectations:
                neural_network.receive_inputs(expectation.inputs)
                self.learn(neural_network, expectation.outputs, .5)
                error += sum(map(lambda arr: math.pow(arr[0] - arr[1], 2),
                                 zip(neural_network.receive_inputs(
                                     expectation.inputs),
                                     expectation.outputs)))

            epochs += 1
            within_acceptable_error = all(
                [expectation < acceptable_error 
                 for output in neural_network.output_layer])

        return TrainingResult(epochs, error)
    

    def learn(self, neural_network, expectation, learning_rate = 1):
        """This is the actual backpropagation part of the code
        here is where we will perform one propagation iteration
        adjusting the networks's node weights
        """        
        action = lambda neuron: self.propagate_errors(
            neural_network, neuron, expectation, learning_rate)

        NetworkUtils.OutputBreadthTraversal(neural_network,
                                            action)


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
                              connection.signalSent)
        

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

    