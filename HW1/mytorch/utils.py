import numpy as np

def gradient_check(model, loss_fn, x, y, epsilon=1e-5):
    """
    Checks the gradients of the model using numerical approximation.
    """
    # 1. Forward Pass
    y_pred = model.forward(x)
    loss_analytic = loss_fn.forward(y_pred, y)
    
    # 2. Backward Pass
    grad_loss = loss_fn.backward()
    model.backward(grad_loss)
    
    # Get analytic gradients
    grads_analytic = model.gradients()
    params = model.parameters()
    
    max_rel_error = 0.0
    
    for name, param in params.items():
        if name not in grads_analytic:
            continue
            
        grad_analytic = grads_analytic[name]
        grad_num = np.zeros_like(param)
        
        # Iterate over each element of the parameter
        it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            orig_val = param[idx]
            
            # f(x + epsilon)
            param[idx] = orig_val + epsilon
            L_plus = loss_fn.forward(model.forward(x), y)
            
            # f(x - epsilon)
            param[idx] = orig_val - epsilon
            L_minus = loss_fn.forward(model.forward(x), y)
            
            # Central Difference
            grad_num[idx] = (L_plus - L_minus) / (2 * epsilon)
            
            # Restore parameter
            param[idx] = orig_val
            it.iternext()

        # rel_error = |analytic - num| / (|analytic| + |num| + epsilon)
        numerator = np.abs(grad_analytic - grad_num)
        denominator = np.abs(grad_analytic) + np.abs(grad_num) + 1e-8
        rel_error = numerator / denominator
        
        max_rel_error = max(max_rel_error, np.max(rel_error))

    return max_rel_error
