from ..module import Module
import numpy as np

class MSELoss(Module):
    def forward(self, pred, target):
        self.diff = pred - target
        return np.mean(self.diff**2)

    def backward(self):
        return 2 * self.diff / self.diff.size

class SoftmaxCrossEntropy(Module):
    def forward(self, logits, target_indices):
        
        # numerical stability
        max_logits = np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits - max_logits)
        self.probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        batch_size = logits.shape[0]
        
        # Check if target_indices is one-hot encoded
        if target_indices.ndim > 1 and target_indices.shape[1] > 1:
            self.target_indices = np.argmax(target_indices, axis=1).flatten().astype(int)
        else:
            self.target_indices = target_indices.flatten().astype(int) 
        
        log_probs = -np.log(self.probs[np.arange(batch_size), self.target_indices])
        return np.mean(log_probs)

    def backward(self):
        batch_size = self.probs.shape[0]
        grad = self.probs.copy()
        grad[np.arange(batch_size), self.target_indices] -= 1
        return grad 
