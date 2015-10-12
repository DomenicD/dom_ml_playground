import unittest
from code.src.neurons.receive_all_neuron import ReceiveAllNeuron
from code.src.neurons.activation_functions.activation_function import (
    ActivationFunction)

class ReceiveAllNeuronTest(unittest.TestCase):
    def test_connectTo(self):
        sender = ReceiveAllNeuron(self.simple_activation())
        receiver = ReceiveAllNeuron(self.simple_activation())
        sender.connect_to(receiver)

        self.assertEqual(len(sender.out_connections), 1)
        self.assertEqual(len(receiver.in_connections), 1)
        self.assertSetEqual(sender.out_connections,
                         receiver.in_connections)


    def test_removeConnection(self):
        sender = ReceiveAllNeuron(self.simple_activation())
        receiver = ReceiveAllNeuron(self.simple_activation())
        connection = sender.connect_to(receiver)

        connection.disconnect()

        self.assertEqual(len(sender.out_connections), 0)
        self.assertEqual(len(receiver.in_connections), 0)
        self.assertSetEqual(sender.out_connections,
                         receiver.in_connections)


    def test_wait_for_all_signals(self):
        sender_1 = ReceiveAllNeuron(self.simple_activation())
        sender_2 = ReceiveAllNeuron(self.simple_activation())
        receiver = ReceiveAllNeuron(self.simple_activation())

        sender_1.connect_to(receiver)
        sender_2.connect_to(receiver)

        sender_1.receive_signal(2.0)
        self.assertEqual(receiver.output, 0.0)
        self.assertFalse(receiver.allSignalsReceived())
        
        sender_2.receive_signal(3.0)
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
        sender = ReceiveAllNeuron(self.simple_activation())
        receiver = ReceiveAllNeuron(self.simple_activation())
        connection = sender.connect_to(receiver)

        self.assertEqual(connection.sender, sender)
        self.assertEqual(connection.receiver, receiver)
        self.assertEqual(connection.weight, 1.0)
        self.assertEqual(connection.signal_sent, 0.0)
        self.assertEqual(connection.signal_received, 0.0)

        connection.weight = 0.5
        sender.receive_signal(7.4)

        self.assertEqual(connection.weight, 0.5)
        self.assertEqual(connection.signal_sent, 3.7)
        self.assertEqual(connection.signal_received, 7.4)


    def test_sigmoid_activation(self):
        neuron = ReceiveAllNeuron()        
        neuron.receive_signal(-0.607)
        self.assertAlmostEqual(neuron.output, 0.3527438)


    def simple_activation(self):
        return ActivationFunction(lambda x: x, lambda x: 1)


if __name__ == '__main__':
    unittest.main()
