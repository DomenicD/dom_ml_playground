﻿import unittest
from code.src.learning_algorithms.backpropagation import Backpropagator
from code.src.neurons.receive_all_neuron import ReceiveAllNeuron
from code.src.data_processing.expectation import Expectation
from code.src.data_processing.normalizer import Normalizer
from code.src.neural_networks.feed_forward import FeedForwardNN
import math



class BackpropagatorTest(unittest.TestCase):
    def test_outputError(self):
        backpropagator = Backpropagator()
        neuron = ReceiveAllNeuron()
        neuron.receive_signal(-0.607)        
        error = backpropagator.output_error(neuron, 0.78)
        self.assertAlmostEqual(error, 0.09754925)


    def test_hiddenError(self):
        backpropagator = Backpropagator()
        output = ReceiveAllNeuron()
        hidden_1 = ReceiveAllNeuron()
        hidden_2 = ReceiveAllNeuron()
        
        hidden_1.connect_to(output)
        hidden_1.out_connections_list[0].weight = -0.75
        
        hidden_2.connect_to(output)
        hidden_2.out_connections_list[0].weight = -0.25

        # sigmoid(0.434719) == 0.6070000
        hidden_1.receive_signal(0.434719)
        hidden_2.receive_signal(0.434719)

        # sigmoid(0.434719) * -0.75 + sigmoid(0.434719) * -0.25 == -0.607
        # sigmoid(-0.607) == 0.352744
        # 0.78 - 0.352744 == 0.3527438
        output.error = backpropagator.output_error(output, 0.78)
        self.assertAlmostEqual(output.error, 0.09754925)

        hidden_1.error = backpropagator.hidden_error(hidden_1)
        hidden_2.error = backpropagator.hidden_error(hidden_2)

        # 0.6070000 * (1 - 0.6070000)*(-0.75 * 0.09754925)
        self.assertAlmostEqual(hidden_1.error, -0.01745285)

        # 0.6070000 * (1 - 0.6070000)*(-0.25 * 0.09754925)
        self.assertAlmostEqual(hidden_2.error, -0.00581762)

        
    def test_adjust_weight(self):
        backpropagator = Backpropagator()
        output = ReceiveAllNeuron()
        hidden_1 = ReceiveAllNeuron()
        hidden_2 = ReceiveAllNeuron()
        
        hidden_1.connect_to(output)
        hidden_1.out_connections_list[0].weight = -0.75
        
        hidden_2.connect_to(output)
        hidden_2.out_connections_list[0].weight = -0.25

        hidden_1.receive_signal(0.434719)
        hidden_2.receive_signal(0.434719)

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


    def test_propagate_errors(self):
        backpropagator = Backpropagator()
        normalizer = Normalizer(in_max = 100, out_max = 200)
        network = FeedForwardNN(normalizer, [1, 2, 1])
        expectation = [148]
        result = network.receive_inputs([74])

        backpropagator.learn(network, expectation)


    def test_teach(self):
        backpropagator = Backpropagator()
        normalizer = Normalizer(in_min = 0, in_max = 100,
                                out_min = -1, out_max = 1)
        network = FeedForwardNN(normalizer, [1, 3, 1])
        expectations = [Expectation([i], [math.sin(i)])
                        for i in range(100)]
        result = backpropagator.teach(network, expectations)
        
        # TODO(domenicd): Need to update Backpropagator to account for
        #                 normalized output, or offer a way to get
        #                 denormalized output from the neural network.
        self.assertEqual(result.error, .0001)
        self.assertEqual(result.epochs, 50)
