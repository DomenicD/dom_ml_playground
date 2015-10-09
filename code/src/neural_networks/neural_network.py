from code.src.neurons.neurons import NeuronType



class NeuronIndex(object):
    def __init__(self, layer, index):
        self.layer = layer
        self.index = index



class NeuralNetwork(object):
    def __init__(self):
        self.layers = []
    
    @property
    def num_layers(self):
        return len(self.layers)


    @property
    def input_layer(self):        
        if self.num_layers > 0: return self.layers[0]
        else: return []        


    @property
    def output_layer(self):        
        if self.num_layers > 0: return self.layers[self.num_layers - 1]
        else: return []


    @property
    def hidden_layers(self):
        end = self.num_layers - 2
        return self.layers[1 : end]


    @property
    def neurons(self):
        return [neuron
                for layer in self.layers
                for neuron in layer]


    def add_layer(self, layer):
        self.layers.append(layer)


    def create_layer(self, count, neuronConstructor):
        return [neuronConstructor() for n in range(count)]


    def neuron_index(self, neuron):
        if neuron.type == NeuronType.OUTPUT:
            return NeuronIndex(self.num_layers - 1,
                               self.output_layer.index(neuron))
        
        elif neuron.type == NeuronType.INPUT:
            return NeuronIndex(0, self.input_layer.index(neuron))
        
        elif neuron.type == NeuronType.HIDDEN:            
            for layerIndex in range(len(self.hidden_layers)):
                for neuronIndex in range(len(hidden_layers[layerIndex])):
                    if hidden_layers[layerIndex][neuronIndex] == neuron:
                        return NeuronIndex(layerIndex + 1, neuronIndex)
        
        raise IndexError()

        
    def prepair_for_input(self): pass
    def receive_inputs(self, inputs): pass
