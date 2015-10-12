import unittest
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
        self.assertAlmostEqual(connection_1.weight, -0.68486637)

        connection_2 = hidden_2.out_connections_list[0]
        backpropagator.adjust_weight(connection_2, 1.1)
        # -0.25 + (1.1 * 0.09754925 * -0.15175)
        self.assertAlmostEqual(connection_2.weight, -0.18486637)


    def test_propagate_errors(self):
        backpropagator = Backpropagator()
        normalizer = Normalizer(in_max = 100, out_max = 200)
        network = FeedForwardNN(normalizer, [1, 2, 1])
        expectation = Expectation([50], [148])
        result = network.receive_inputs(expectation.inputs)

        backpropagator.learn(network, expectation, 1)


    def test_calculate_error(self):
        backpropagator = Backpropagator()
        error = backpropagator.calculate_error([5.5, 1, -8], [5, -4, 2.5])
        self.assertAlmostEqual(error, 135.5)


    def test_learn(self):
        backpropagator = Backpropagator()
        normalizer = Normalizer(in_min = 1, in_max = 15,
                                out_min = -1, out_max = 1,
                                norm_min = -3, norm_max = 3)
        
        network = FeedForwardNN(normalizer, [1, 2, 2, 1])
        network.randomize_connection_weights(seed = 74)
        neurons = network.neurons        
        expectation = Expectation([1], [0.8415])

        error = backpropagator.calculate_error(
            network.receive_inputs(expectation.inputs),
            expectation.outputs)

        for i in range(20):
            last_error = error
            backpropagator.learn(network, expectation, 1.5)
            actual = network.receive_inputs(expectation.inputs)
            print(actual)
            error = backpropagator.calculate_error(actual, expectation.outputs)
            self.assertLess(error, last_error)


    def test_teach_acceptable_error(self):
        backpropagator = Backpropagator()
        normalizer = Normalizer(in_min = -15, in_max = 15,
                                out_min = -30, out_max = 30,
                                norm_min = -2, norm_max = 2)
        
        network = FeedForwardNN(normalizer, [1, 3, 1])
        
        network.randomize_connection_weights(seed = 74)

        expectations = [Expectation([i], [2*i])
                        for i in range(-5, 5)]

        result = backpropagator.teach(network,
                                      expectations, 
                                      learning_rate = 1.5,
                                      max_iterations = 2000,
                                      acceptable_error = .5)
        
        self.assertLessEqual(result.error, .5)

        errors = 0
        for exp in expectations:
            errors += backpropagator.calculate_error(
                network.receive_inputs(exp.inputs), exp.outputs)

        self.assertLessEqual(errors, .5)
        

    def test_teach_max_iterations(self):
        backpropagator = Backpropagator()
        normalizer = Normalizer(in_min = -15, in_max = 15,
                                out_min = -30, out_max = 30,
                                norm_min = -2, norm_max = 2)
        
        network = FeedForwardNN(normalizer, [1, 2, 2, 1])
        
        network.randomize_connection_weights(seed = 74)

        expectations = [Expectation([i], [2*i])
                        for i in range(-5, 5)]
        
        result = backpropagator.teach(network,
                                      expectations, 
                                      learning_rate = 1.5,
                                      max_iterations = 123,
                                      acceptable_error = 0)
        
        self.assertEqual(result.epochs, 123)
    