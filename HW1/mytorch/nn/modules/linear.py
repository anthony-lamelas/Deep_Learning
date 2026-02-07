from ..module import Module
import numpy as np

class Linear(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # He Initialization
        std = np.sqrt(2.0 / in_features)
        self.W = np.random.randn(out_features, in_features) * std
        self.b = np.zeros(out_features)
        
        self._parameters['weight'] = self.W
        self._parameters['bias'] = self.b
        
        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.b)
        
        self._gradients['weight'] = self.grad_W
        self._gradients['bias'] = self.grad_b
        
        self.x = None

    def forward(self, x):
        self.x = x
        return x @ self.W.T + self.b

    def backward(self, dLdy):
        # dLdW = dLdy^T @ x -> (out, batch) @ (batch, in) = (out, in)
        self.grad_W[:] = dLdy.T @ self.x / self.x.shape[0]
        
        # dLdb = sum(dLdy, axis=0) -> (out,)
        self.grad_b[:] = np.sum(dLdy, axis=0) / self.x.shape[0]
        
        # dLdx = dLdy @ W -> (batch, out) @ (out, in) = (batch, in)
        return dLdy @ self.W
