import math
import uuid
from enum import Enum    
from code.src.neurons.neuron_connection import NeuronConnection



class NeuronType(Enum):
    UNKNOWN = 0,
    INPUT = 1,
    OUTPUT = 2,
    HIDDEN = 3



class Neuron(object):
    # __metaclass__ = ABCMeta
    
    def __init__(self, activation):
        self.id = uuid.uuid4().hex        
        self.activation = activation
        self.out_connections = set()
        self.in_connections = set()
        self.accumulated_input_signals = 0.0
        self.output = 0.0
        self.error = 0.0


    @property
    def out_connections_list(self):
        return list(self.out_connections)


    @property
    def in_connections_list(self):
        return list(self.in_connections)


    @property
    def type(self):
        if (len(self.out_connections) == 0 and
            len(self.in_connections) == 0):
            return NeuronType.UNKNOWN

        if len(self.out_connections) == 0:
            return NeuronType.OUTPUT

        if len(self.in_connections) == 0:
            return NeuronType.INPUT

        return NeuronType.HIDDEN


    def fire(self):
        for c in self.out_connections:
            c.sendSignal(self.output)


    # This creates a one way connection from this neuron to the passed in one.
    def connect_to(self, node):
        connection = NeuronConnection(self, node)
        if node.add_inbound(connection):
            self.add_outbound(connection)
            return connection    
        return None


    def add_outbound(self, connection):
        self.out_connections.add(connection)
        self.on_outbound_added(connection)


    def add_inbound(self, connection):
        self.in_connections.add(connection)
        self.on_inbound_added(connection)
        return True


    def remove_outbound(self, connection):
        self.out_connections.remove(connection)
        self.on_outbound_removed(connection)


    def remove_inbound(self, connection):
        self.in_connections.remove(connection)
        self.on_inbound_removed(connection)


    def receive_signal(self, value, connection_id = None):
        self.accumulated_input_signals += value
        self.on_signal_received(value, connection_id)


    def reset(self):
        self.accumulated_input_signals = 0.0
        self.output = 0.0
        self.error = 0.0
        self.on_reset()


    # Overridable event methods
    def on_inbound_added(self, connection): pass
    def on_inbound_removed(self, connection): pass
    def on_outbound_added(self, connection): pass
    def on_outbound_removed(self, connection): pass
    def on_signal_received(self, value, connection_id): pass
    def on_reset(self): pass
