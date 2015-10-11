import unittest
from code.src.data_processing.normalizer import Normalizer
from code.src.neural_networks.feed_forward import FeedForwardNN
from code.src.neurons.output_only_neuron import OutputOnlyNeuron

class SimpleFeedForwardNNTest(unittest.TestCase):
    def test_input_passes_through_network(self):
        normalizer = Normalizer(2, 1)
        network = FeedForwardNN(normalizer, [2, 4, 5, 3])

        for input in network.input_layer:
            input.receive_signal(5.0)

        for output in network.output_layer:
            self.assertNotAlmostEqual(output.output, 0.0)



        sentSignals = [connection.signalSent 
                       for layer in network.layers 
                       for node in layer 
                       for connection in node.out_connections]
        
        self.assertTrue(s != 0.0 for s in sentSignals)


    def test_bias_nodes(self):
        network = FeedForwardNN(structure = [2, 4, 5, 3])

        bias_1 = network.layers[0][-1]
        bias_2 = network.layers[1][-1]
        bias_3 = network.layers[2][-1]

        self.assertTrue(type(bias_1) is OutputOnlyNeuron)
        self.assertTrue(type(bias_2) is OutputOnlyNeuron)
        self.assertTrue(type(bias_3) is OutputOnlyNeuron)
        self.assertFalse(type(network.layers[3][-1]) is OutputOnlyNeuron)

        self.assertEqual(len(bias_1.in_connections_list), 0)
        self.assertEqual(len(bias_2.in_connections_list), 0)
        self.assertEqual(len(bias_3.in_connections_list), 0)

        self.assertEqual(len(bias_1.out_connections_list), 4)
        self.assertEqual(len(bias_2.out_connections_list), 5)
        self.assertEqual(len(bias_3.out_connections_list), 3)



    def test_bias_nodes_off(self):
        network = FeedForwardNN(structure = [2, 4, 5, 3],
                                has_bias_nodes = False)

        self.assertFalse(type(network.layers[0][-1]) is OutputOnlyNeuron)
        self.assertFalse(type(network.layers[1][-1]) is OutputOnlyNeuron)
        self.assertFalse(type(network.layers[2][-1]) is OutputOnlyNeuron)
        self.assertFalse(type(network.layers[3][-1]) is OutputOnlyNeuron)


    def test_calculation(self):
        normalizer = Normalizer(in_min = 1, in_max = 15,
                                out_min = -1, out_max = 1,
                                norm_min = -3, norm_max = 3)
        
        network = FeedForwardNN(normalizer, [1, 2, 2, 1])
        neurons = network.neurons

        outputs = network.receive_inputs([1])
        

        self.assertEqual(len(outputs), 1)
        # Nodes 1, 4, and 7 are bias nodes
        self.assertAlmostEqual(neurons[0].output, 0.04742587318)
        self.assertAlmostEqual(neurons[1].output, 1)
        self.assertAlmostEqual(neurons[2].output, 0.7402802899)
        self.assertAlmostEqual(neurons[3].output, 0.7402802899)
        self.assertAlmostEqual(neurons[4].output, 1)
        self.assertAlmostEqual(neurons[5].output, 0.9227677584)
        self.assertAlmostEqual(neurons[6].output, 0.9227677584)
        self.assertAlmostEqual(neurons[7].output, 1)
        self.assertAlmostEqual(neurons[8].output, 0.9450874486)
        self.assertAlmostEqual(outputs[0], 0.8901748973)




if __name__ == '__main__':
    unittest.main()
