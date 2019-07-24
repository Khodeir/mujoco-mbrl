from dm_control.mujoco import wrapper
from torch.autograd.gradcheck import zero_gradients

import torch
import numpy as np


# Could probably batch this over the entire trajectory?
def compute_jacobian(inputs, output):
    """
    Jacobian is of d(outputs)/d(inputs) for each pair. These should both be Tensors in a computation graph
    that have requires_grad=True.
    In the context of a trajectory, inputs is (x_t, y_t), outputs is f(x_t, y_t) where f is an NN model for example'''
    """
    assert inputs.requires_grad

    num_inputs = inputs.size()[-1]
    num_outputs = output.size()[-1]
    
    jacobian = torch.zeros(num_outputs, num_inputs)
    # Need to iterate over each output to get one
    for i in range(num_outputs):
        zero_gradients(inputs)
        grad_mask = torch.zeros(num_outputs)
        grad_mask[i] = 1
        output.backward(grad_mask, retain_graph=True)
        jacobian[i] = inputs.grad

    return jacobian

