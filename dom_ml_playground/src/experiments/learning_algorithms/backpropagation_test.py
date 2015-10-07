﻿import unittest
from src.experiments.learning_algorithms.backpropagation import Backpropagator
from src.experiments.neural_networks.neurons import ReceiveAllNeuron

class BackpropagatorTest(unittest.TestCase):
    def test_outputError(self):
        backpropagator = Backpropagator()
        neuron = ReceiveAllNeuron()
        neuron.receiveSignal(-0.607)        
        error = backpropagator.outputError(neuron, 0.78)
        self.assertAlmostEqual(error, 0.09754925)




    def test_hiddenError(self):
        backpropagator = Backpropagator()
        output = ReceiveAllNeuron()
        hidden_1 = ReceiveAllNeuron()
        hidden_2 = ReceiveAllNeuron()
        
        hidden_1.connectTo(output)
        hidden_1.out_connections_list[0].weight = -0.75
        
        hidden_2.connectTo(output)
        hidden_2.out_connections_list[0].weight = -0.25

        # sigmoid(0.434719) == 0.6070000
        hidden_1.receiveSignal(0.434719)
        hidden_2.receiveSignal(0.434719)

        # sigmoid(0.434719) * -0.75 + sigmoid(0.434719) * -0.25 == -0.607
        # sigmoid(-0.607) == 0.352744
        # 0.78 - 0.352744 == 0.3527438
        output.error = backpropagator.outputError(output, 0.78)
        self.assertAlmostEqual(output.error, 0.09754925)

        hidden_1.error = backpropagator.hiddenError(hidden_1)
        hidden_2.error = backpropagator.hiddenError(hidden_2)

        # 0.6070000 * (1 - 0.6070000)*(-0.75 * 0.09754925)
        self.assertAlmostEqual(hidden_1.error, -0.01745285)

        # 0.6070000 * (1 - 0.6070000)*(-0.25 * 0.09754925)
        self.assertAlmostEqual(hidden_2.error, -0.00581762)

        


    def test_adjust_weight(self):
        backpropagator = Backpropagator()
        output = ReceiveAllNeuron()
        hidden_1 = ReceiveAllNeuron()
        hidden_2 = ReceiveAllNeuron()
        
        hidden_1.connectTo(output)
        hidden_1.out_connections_list[0].weight = -0.75
        
        hidden_2.connectTo(output)
        hidden_2.out_connections_list[0].weight = -0.25

        hidden_1.receiveSignal(0.434719)
        hidden_2.receiveSignal(0.434719)

        output.error = 0.09754925
        hidden_1.error = -0.01745285
        hidden_2.error = -0.00581762

        connection_1 = hidden_1.out_connections_list[0]
        backpropagator.adjust_weight(connection_1, 1.1)
        # -0.75 + (1.1 * 0.09754925 * -0.45525)
        self.assertAlmostEqual(connection_1.weight, -0.7988502)

        connection_2 = hidden_2.out_connections_list[0]
        backpropagator.adjust_weight(connection_2, 1.1)
        # -0.25 + (1.1 * 0.09754925 * -0.15175)
        self.assertAlmostEqual(connection_2.weight, -0.26628341)

