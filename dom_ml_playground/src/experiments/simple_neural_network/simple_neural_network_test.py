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

        sender_1.receiveSignal(2.0)
        self.assertEqual(receiver.output, 0.0)
        self.assertFalse(receiver.allSignalsReceived())
        
        sender_2.receiveSignal(3.0)
        self.assertEqual(receiver.output, 5.0)
        self.assertTrue(receiver.allSignalsReceived())

        sender_1.reset()
        sender_2.reset()
        receiver.reset()

        self.assertEqual(sender_1.output, 0.0)
        self.assertEqual(sender_2.output, 0.0)
        self.assertEqual(receiver.output, 0.0)
        self.assertFalse(receiver.allSignalsReceived())


    def test_connection(self):
        sender = snn.ReceiveAllNeuron(lambda x: x)
        receiver = snn.ReceiveAllNeuron(lambda x: x)
        connection = sender.connectTo(receiver)

        self.assertEqual(connection.sender, sender)
        self.assertEqual(connection.receiver, receiver)
        self.assertEqual(connection.weight, 1.0)
        self.assertEqual(connection.signalSent, 0.0)
        self.assertEqual(connection.signalReceived, 0.0)

        connection.weight = 0.5
        sender.receiveSignal(7.4)

        self.assertEqual(connection.weight, 0.5)
        self.assertEqual(connection.signalSent, 3.7)
        self.assertEqual(connection.signalReceived, 7.4)


class SimpleFeedForwardNNTest(unittest.TestCase):
    def test_input_passes_through_network(self):
        normalizers = snn.normalizers(2, 1)
        network = snn.SimpleFeedForwardNN(normalizers)

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
