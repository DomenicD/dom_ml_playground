import unittest
from simple_feed_forward import SimpleFeedForwardNN, normalizers


class SimpleFeedForwardNNTest(unittest.TestCase):
    def test_input_passes_through_network(self):
        inOutFuncs = normalizers(2, 1)
        network = SimpleFeedForwardNN(inOutFuncs)

        network.input.receiveSignal(5.0)

        self.assertNotAlmostEqual(network.output.output, 0.0)
        connections = (list(network.input.outConnections) + 
                       list(network.output.outConnections))

        for h in network.hidden:
            connections += h.outConnections

        sentSignals = [c.signalSent for c in connections]
        self.assertTrue(s != 0.0 for s in sentSignals)


if __name__ == '__main__':
    unittest.main()
