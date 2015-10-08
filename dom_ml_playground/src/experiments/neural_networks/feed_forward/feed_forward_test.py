import unittest
from src.experiments.neural_networks.neural_network import Normalizer
from src.experiments.neural_networks.feed_forward.feed_forward import (
    FeedForwardNN)

class SimpleFeedForwardNNTest(unittest.TestCase):
    def test_input_passes_through_network(self):
        normalizer = Normalizer(2, 1)
        network = FeedForwardNN(normalizer, [2, 4, 5, 3])

        for input in network.input_layer:
            input.receiveSignal(5.0)

        for output in network.output_layer:
            self.assertNotAlmostEqual(output.output, 0.0)



        sentSignals = [connection.signalSent 
                       for layer in network.layers 
                       for node in layer 
                       for connection in node.outConnections]
        
        self.assertTrue(s != 0.0 for s in sentSignals)


if __name__ == '__main__':
    unittest.main()
