import unittest
from simple_feed_forward import SimpleFeedForwardNN
from src.experiments.neural_networks.neurons import Normalizer

class SimpleFeedForwardNNTest(unittest.TestCase):
    def test_input_passes_through_network(self):
        normalizer = Normalizer(2, 1)
        network = SimpleFeedForwardNN(normalizer, [2, 4, 5, 3])

        for input in network.inputs:
            input.receiveSignal(5.0)

        for output in network.outputs:
            self.assertNotAlmostEqual(output.output, 0.0)



        sentSignals = [connection.signalSent 
                       for layer in network.layers 
                       for node in layer 
                       for connection in node.outConnections]
        
        self.assertTrue(s != 0.0 for s in sentSignals)


if __name__ == '__main__':
    unittest.main()
