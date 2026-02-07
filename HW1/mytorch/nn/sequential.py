from .module import Module

class Sequential(Module):
    def __init__(self, *args):
        super().__init__()
        self.layers = list(args)

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, dLdy):
        for layer in reversed(self.layers):
            dLdy = layer.backward(dLdy)
        return dLdy

    def parameters(self):
        params = {}
        for i, layer in enumerate(self.layers):
            for name, param in layer.parameters().items():
                params[f"{i}.{name}"] = param
        return params

    def gradients(self):
        grads = {}
        for i, layer in enumerate(self.layers):
            for name, grad in layer.gradients().items():
                grads[f"{i}.{name}"] = grad
        return grads
