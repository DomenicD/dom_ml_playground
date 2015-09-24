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


    def test_removeConnection(self):
        sender = snn.ReceiveAllNeuron(lambda x: x)
        receiver = snn.ReceiveAllNeuron(lambda x: x)
        connection = sender.connectTo(receiver)

        connection.disconnect()

        self.assertEqual(len(sender.outConnections), 0)
        self.assertEqual(len(receiver.inConnections), 0)
        self.assertSetEqual(sender.outConnections,
                         receiver.inConnections)


    def test_wait_for_all_signals(self):
        sender_1 = snn.ReceiveAllNeuron(lambda x: x)
        sender_2 = snn.ReceiveAllNeuron(lambda x: x)
        receiver = snn.ReceiveAllNeuron(lambda x: x)

        sender_1.connectTo(receiver)
        sender_2.connectTo(receiver)

        sender_1.onSignalReceived(2.0)
        self.assertEqual(receiver.output, 0.0)
        self.assertFalse(receiver.allSignalsReceived())
        
        sender_2.onSignalReceived(3.0)
        self.assertEqual(receiver.output, 5.0)
        self.assertTrue(receiver.allSignalsReceived())

        sender_1.reset()
        sender_2.reset()
        receiver.reset()

        self.assertEqual(sender_1.output, 0.0)
        self.assertEqual(sender_2.output, 0.0)
        self.assertEqual(receiver.output, 0.0)
        self.assertFalse(receiver.allSignalsReceived())





if __name__ == '__main__':
    unittest.main()
