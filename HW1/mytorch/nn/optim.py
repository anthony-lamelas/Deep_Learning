class SGD:
    def __init__(self, model_params, lr=0.1):
        pass

    def step(self, model):
        # Update rule: w = w - lr * grad
        params = model.parameters()
        grads = model.gradients()
        
        for name in params:
            # name is like "0.weight"
            if name in grads:
                params[name] -= self.lr * grads[name]
    
    # Redefine to standard interface
    def __init__(self, model, lr=0.1):
        self.model = model
        self.lr = lr
        
    def step(self):
        params = self.model.parameters()
        grads = self.model.gradients()
        for k in params:
            params[k] -= self.lr * grads[k]

    def zero_grad(self):
        # Not strictly needed if we overwrite grads in backward, but good practice
        pass
