import numpy as np

class Module:
    def __init__(self):
        self._parameters = {}
        self._gradients = {}
        self.training = True

    def forward(self, *args):
        raise NotImplementedError

    def backward(self, *args):
        raise NotImplementedError

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def parameters(self):
        return self._parameters

    def gradients(self):
        return self._gradients
