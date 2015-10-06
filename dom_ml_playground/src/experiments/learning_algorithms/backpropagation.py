from src.experiments.neural_networks.neural_network_utils import NetworkUtils
from src.experiments.neural_networks.neurons import NeuronType

# http://home.agh.edu.pl/~vlsi/AI/backp_t_en/backprop.html
# http://www.cse.unsw.edu.au/~cs9417ml/MLP2/


# wonder how to compute the contribution of each node
# during feed forward phase?
class Expectation(object):
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs




class Backpropagator(object):
    def teach(self, neuralNetwork, trainingData,
              acceptableError = .001, maxIterations = 100,
              timeLimit = None):
        for i in range(maxIterations):
            self.lesson(neuralNetwork, trainingData, .5)

    
    def lesson(self, neuralNetwork, trainingData, learning_rate):
        for expectation in trainingData:
            neuralNetwork.prepairForInput()
            for i in range(len(neuralNetwork.inputLayer)):
                neuron = neuralNetwork.inputLayer[i]
                input = expectation.inputs[i]
                neuron.receiveSignal(input)
            learn(neuralNetwork, expectation.outputs, learning_rate)

    

    def learn(self, neuralNetwork, expectations, learning_rate):
        """This is the actual backpropagation part of the code
        here is where we will perform one propagation iteration
        adjusting the networks's node weights
        """
        # TODO: Use the NetworkUtils breadth traversal
        # to perform backpropagation. Need to make an
        # 'action' that checks the node type and adjusts
        # the weights of its 'outbound' connections.        
        action = lambda neuron: self.propagate_errors(
            neuralNetwork, neuron, expectations, learning_rate)

        NetworkUtils.OutputBreadthTraversal(neuralNetwork,
                                            action)


    def propagate_errors(self, network, neuron, expectations,
                         learning_rate):     
           
        if neuron.type == NeuronType.OUTPUT:
            neuron_index = network.neuron_index(neuron)
            expectation = expectations[neuron_index]
            neuron.error = self.outputError(
                neuron, expectation)
        else:
            # Calculate the error for the current layer
            neuron.error = self.hiddenError(neuron)

            # Adjust the weights of the prior layer
            for outbound in neuron.outConnections:
                self.adjust_weight(outbound, learning_rate)
    

    
    def adjust_weight(self, connection, learning_rate):        
        connection.weight += (learning_rate * 
                              connection.receiver.error * 
                              connection.signalReceived)
        

    def outputError(self, neuron, expectation):        
        return ((expectation - neuron.output) * 
                neuron.activation.derivative(
                    neuron.accumulatedInputSignals))


    def hiddenError(self, neuron):
        weightedErrorSum = reduce(
            lambda sum, c: sum + c.weight * c.receiver.error, 
            neuron.outConnections)

        return (weightedErrorSum *
                neuron.activation.derivative(
                    neuron.accumulatedInputSignals))

    