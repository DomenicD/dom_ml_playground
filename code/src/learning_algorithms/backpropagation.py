from code.src.neural_networks.neural_network_utils import NetworkUtils
from code.src.neurons.neurons import NeuronType


# http://home.agh.edu.pl/~vlsi/AI/backp_t_en/backprop.html
# http://www.cse.unsw.edu.au/~cs9417ml/MLP2/


# wonder how to compute the contribution of each node
# during feed forward phase?
class Expectation(object):
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs




class Backpropagator(object):
    def teach(self, neural_network, expectations,
              acceptable_error = .001, max_iterations = 100,
              time_limit = None):
        for i in range(max_iterations):
            self.lesson(neural_network, expectations, .5)

    
    def lesson(self, neural_network, expectations, learning_rate = 1):
        for expectation in expectations:
            neural_network.prepair_for_input()
            for i in range(len(neural_network.input_layer)):
                neuron = neural_network.input_layer[i]
                input = expectation.inputs[i]
                neuron.receive_signal(input)
            learn(neural_network, expectation.outputs, learning_rate)

    

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

    