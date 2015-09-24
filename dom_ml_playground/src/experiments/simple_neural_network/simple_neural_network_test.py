import unittest
import simple_neural_network as snn

class ReceiveAllNeuronTest(unittest.TestCase):
    def test_connectTo(self):
        sender = snn.ReceiveAllNeuron(lambda x: x)
        receiver = snn.ReceiveAllNeuron(lambda x: x)
        sender.connectTo(receiver)

        self.assertEqual(len(sender.outConnections), 1)
        self.assertEqual(len(receiver.inConnections), 1)
        self.assertSetEqual(sender.outConnections,
                         receiver.inConnections)

if __name__ == '__main__':
    unittest.main()
