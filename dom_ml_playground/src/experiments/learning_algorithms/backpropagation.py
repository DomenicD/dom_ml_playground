
# http://home.agh.edu.pl/~vlsi/AI/backp_t_en/backprop.html
# http://www.cse.unsw.edu.au/~cs9417ml/MLP2/


# wonder how to compute the contribution of each node
# during feed forward phase?
class NeuronStat(object):
    def __init__(self, nodeId):
        self.nodeId = nodeId
        self.reset()
        

    def reset(self):
        self.error = None
        self.weightDelta = None


class Expectation(object):
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs




class Backpropagator(object):
    def __init__(self, learnRate):
        self.learnRate = learnRate
        self.nodeStats = {}


    def teach(self, neuralNetwork, trainingData,
              acceptableError = .001, maxIterations = 1000,
              timeLimit = None):
        outputNodes = neuralNetwork.outputLayer        
        # compute error at node
        # for each parent
        #    pass error up to parent node
        #    adjust weight with parent node

    
    def session(self, neuralNetwork, trainingData):
        pass



    def lesson(self, neuralNetwork, expectation):
        neuralNetwork.prepairForInput()
        for i in range(len(neuralNetwork.inputLayer)):
            neuron = neuralNetwork.inputLayer[i]
            input = expectation.inputs[i]
            neuron.receiveSignal(input)
        learn(neuralNetwork, expectation.outputs, actualOutputs)

    

    def learn(self, neuralNetwork, expectations):
        """This is the actual backpropagation part of the code
        here is where we will perform one propagation iteration
        adjusting the networks's node weights
        """
        # TODO: Use the NetworkUtils breadth traversal
        # to perform backpropagation. Need to make an
        # 'action' that checks the node type and adjusts
        # the weights of its 'outbound' connections.        
        for i in range(len(neuralNetwork.outputLayer)):
            neuron = neuralNetwork.outputLayer[i]
            expectation = expectations[i]            
            neuron.error = self.outputError(
                neuron, expectation)



    
    def weightAdjustment(self, connection, learningRate):        
        return (learningRate * 
                connection.receiver.error * 
                connection.signalReceived)
        

    def outputError(self, neuron, expectation):        
        return ((expectation - neuron.output) * 
                neuron.activation.derivative(
                    neuron.accumulatedInputSignals))


    def hiddenError(self, neuron, expectation):
        weightedErrorSum = reduce(
            lambda sum, c: sum + c.weight * c.receiver.error, 
            neuron.outConnections)

        return (weightedErrorSum *
                neuron.activation.derivative(
                    neuron.accumulatedInputSignals))

    