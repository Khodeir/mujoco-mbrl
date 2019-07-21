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


def sample_reacher_state():
    state_dim = 4
    state = np.zeros(state_dim)

    state[0] = np.random.uniform(low=-np.pi, high=np.pi)  # 2*np.pi*np.random.rand()- np.pi
    state[1] = np.random.uniform(low=-2.8, high=2.8)  # Avoid infeasible goals
    state[2] = np.random.uniform(low=-3, high=3)
    state[3] = np.random.uniform(low=-3, high=3)

    return state


def sample_swimmer_state():
    # Assuming swimmer3
    state_dim = 10
    state = np.zeros(state_dim)

    state[2] = np.random.uniform(low=-3, high=3)

    return state


def sample_cheetah_state():
    state_dim = 18
    state = np.zeros(state_dim)
    state[1] = np.random.uniform(-0.2, 0.2)  # Vertical Height
    if state[1] > 0.05:
        state[2] = np.random.uniform(-3.14, 3.14)  # torso angle (0, 50)
    else:
        if np.random.uniform() < 0.72:
            state[2] = np.random.uniform(-3.14, -1.5)  # torso angle (-50, 0)
        else:
            state[2] = np.random.uniform(2.5, 3.14)
        
    state[3] = np.random.uniform(-0.5236, 1.0472)  # bthigh (-30, 60) = (-0.5236, 1.0472)
    state[4] = np.random.uniform(-0.8727, 0.8727)  # bshin (-50, 50) = (-0.8727, 0.8727)
    state[5] = np.random.uniform(-4.0143, 0.8727)  # bfoot (-230, 50) = (-4.0143, 0.8727)
    state[6] = np.random.uniform(-0.9948, 0.0070)  # fthigh (-57, .4) = (-0.9948, 0.0070)
    state[7] = np.random.uniform(-1.2217, 0.8727)  # fshin (-70, 50) = (-1.2217, 0.8727)
    state[8] = np.random.uniform(-0.4887, 0.4887)  # ffoot (-28, 28) = (-0.4887, 0.4887)
    state[9:] = np.random.uniform(-3, 3, 9)  # Velocities

    return state


def sample_walker_state():
    state_dim = 18
    state = np.zeros(state_dim)   
    # state[0] = np.random.uniform() ## horizontal position
    # state[1] = 0#np.random.uniform(0, 0.05) ## vertical position
    state[2] = np.random.uniform(-.1, .1)  # clock rotation of main body
    
    hip_rot = np.random.uniform(-0.15, 0.15)
    state[3] = hip_rot  # right_hip (-20, 100)   = (-0.3491, 1.7452)
    state[4] = np.random.uniform(-.3, 0)  # right_knee (-150, 0)   = (-2.6178 , 0)
    state[5] = np.random.uniform(-0.1, 0.1)  # right_ankle (-45, 45)  = (-0.7854, 0.7854)
    state[6] = -hip_rot  # np.random.uniform(-0.3491, 1.) ## left_hip (-20, 100)   = (-0.3491, 1.7452)
    state[7] = np.random.uniform(-.3, 0)  # left_knee (-150, 0)   = (-2.6178 , 0)
    state[8] = np.random.uniform(-0.1, 0.1)  # left_ankle (-45, 45)  = (-0.7854, 0.7854)
    
    # state[9:] = np.random.uniform(-0.04, 0.04, 9) ## Velocities

    return state


def sample_hopper_state():
    state_dim = 14
    state = np.zeros(state_dim)   
    # state[0] = np.random.uniform() ## horizontal position
    state[1] = -0.078789  # np.random.uniform(0, 0.05) ## vertical position
    state[2] = np.random.uniform(-.01, .01)  # clock rotation of main body
    state[3] = np.random.uniform(-.01, .01)  # clock rotation of main body right_hip (-20, 100) = (-0.3491, 1.7452)
    state[4] = np.random.uniform(-.01, .01)  # hip (-170, 10)   = (-2.6178 , 0)
    state[5] = np.random.uniform(0.1, 0.12)  # knee (5, 150)  = (-0.7854, 0.7854)
    state[6] = np.random.uniform(-0.01, 0.01)  # ankle (-45, 45)  = (-0.7854, 0.7854)
    
    state[7:] = np.random.uniform(-0.01, 0.01, 7)  # Velocities

    return state


def sample_humanoid_state():
    state_dim = 55
    state = np.zeros(state_dim)  # Removed 0,1 and 3,4,5,6

    state[2] = 1.3  # np.random.uniform(1.45, 1.55) ## Vertical Position
    state[7] = np.random.uniform(-0.7854, 0.7854)  # abdomen_z (-45, 45) = (-0.7854, 0.7854)
    state[8] = np.random.uniform(-1.3089, 0.5236)  # abdomen_y (-75, 30) = (-1.3089, 0.5236)
    state[9] = np.random.uniform(-0.6109, 0.6109)  # abdomen_x (-35, 35) = (-0.6109, 0.6109)

    state[10] = np.random.uniform(-0.4363, 0.0873)  # right_hip_x (-25, 5) = (-0.4363, 0.0873)
    state[11] = np.random.uniform(-1.0472, 0.6109)  # right_hip_z (-60, 35) = (-1.0472, 0.6109)
    state[12] = np.random.uniform(-1.9199, 0.3491)  # right_hip_y (-110, 20) = (-1.9199, 0.3491)
    state[13] = np.random.uniform(-2.7925, 0.0349)  # right_knee (-160, 2) = (-2.7925, 0.0349)
    state[14] = np.random.uniform(-0.8727, 0.8727)  # right_ankle_y (-50, 50) = (-0.8727, 0.8727)
    state[15] = np.random.uniform(-0.8727, 0.8727)  # right_ankle_x (-50, 50) = (-0.8727, 0.8727)

    state[16] = np.random.uniform(-0.4363, 0.0873)  # left_hip_x (-25, 5) = (-0.4363, 0.0873)
    state[17] = np.random.uniform(-1.0472, 0.6109)  # left_hip_z (-60, 35) = (-1.0472, 0.6109)
    state[18] = np.random.uniform(-1.9199, 0.3491)  # left_hip_y (-110, 20) = (-1.9199, 0.3491)
    state[19] = np.random.uniform(-2.7925, 0.0349)  # left_knee (-160, 2) = (-2.7925, 0.0349)
    state[20] = np.random.uniform(-0.8727, 0.8727)  # left_ankle_y (-50, 50) = (-0.8727, 0.8727)
    state[21] = np.random.uniform(-0.8727, 0.8727)  # left_ankle_x (-50, 50) = (-0.8727, 0.8727)

    state[22] = np.random.uniform(-1.4835, 1.0472)  # right_shoulder1 (-85, 60) = (-1.4835, 1.0472)
    state[23] = np.random.uniform(-1.4835, 1.0472)  # right_shoulder2 (-85, 60) = (-1.4835, 1.0472)
    state[24] = np.random.uniform(-1.5708, 0.8727)  # right_elbow (-90, 50) = (-1.5708, 0.8727)

    state[25] = np.random.uniform(-1.0472, 1.4835)  # left_shoulder1 (-60, 85) = (-1.0472, 1.4835)
    state[26] = np.random.uniform(-1.0472, 1.4835)  # left_shoulder2 (-60, 85) = (-1.0472, 1.4835)
    state[27] = np.random.uniform(-1.5708, 0.8727)  # left_elbow (-90, 50) = (-1.5708, 0.8727)

    # state[34:] = 0.1*np.random.uniform(-0.1, 0.1, 21)  # Velocities

    return state
