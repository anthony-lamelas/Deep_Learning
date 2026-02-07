from ..module import Module
import numpy as np

class ReLU(Module):
    def __init__(self):
        super().__init__()
        self.mask = None

    def forward(self, x):
        self.mask = (x > 0)
        return x * self.mask

    def backward(self, dLdy):
        return dLdy * self.mask

class Tanh(Module):
    def __init__(self):
        super().__init__()
        self.out = None

    def forward(self, x):
        self.out = np.tanh(x)
        return self.out

    def backward(self, dLdy):
        return dLdy * (1 - self.out**2)

class Sigmoid(Module):
    def __init__(self):
        super().__init__()
        self.out = None

    def forward(self, x):
        self.out = 1 / (1 + np.exp(-x))
        return self.out

    def backward(self, dLdy):
        return dLdy * self.out * (1 - self.out)
