import unittest
from code.src.neural_networks.neural_network_utils import NeuralNetworkUtils
from code.src.neural_networks.feed_forward import FeedForwardNN

class NeuralNetworkUtilsTest(unittest.TestCase):
    def test_output_breadth_traversal(self):
        network = FeedForwardNN(structure = [1, 3, 5, 4, 2])
        NeuralNetworkUtils.OutputBreadthTraversal(
            network, lambda neuron: _set_error(neuron, -44))
        self.assertTrue(all([n.error == -44 for n in network.neurons]))


def _set_error(neuron, error):
    neuron.error = error
