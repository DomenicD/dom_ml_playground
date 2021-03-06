﻿import uuid



class NeuronConnection(object):
    def __init__(self, sender, receiver):
        self.id = uuid.uuid4().hex
        self.sender = sender
        self.receiver = receiver
        self._weight = 1.0
        self.weight = self._weight
        self.signal_sent = 0.0
        self.signal_received = 0.0
        

    @property
    def weight(self):
        return self._weight


    @weight.setter
    def weight(self, value):
        self._prior_weight = self.weight
        self._weight = value


    @property
    def prior_weight(self):
        return self._prior_weight


    @property
    def weight_difference(self):
        return self.weight - self.prior_weight


    @property
    def weight_change(self):
        return weight_difference / self.prior_weight


    def sendSignal(self, value):
        self.signal_received = value
        self.signal_sent = value * self.weight        
        self.receiver.receive_signal(self.signal_sent, self.id)

    def disconnect(self):
        self.sender.remove_outbound(self)
        self.receiver.remove_inbound(self)
