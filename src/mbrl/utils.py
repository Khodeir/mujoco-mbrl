import os
from PIL import Image
import subprocess
import shutil

from torch.autograd.gradcheck import zero_gradients

import torch


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


class Recorder:
    # recorder.record_frame(env.physics.render(camera_id=0), t)

    def __init__(self, experiment_name, count):
        self.experiment_name = experiment_name
        self.count = count
        os.makedirs('frames', exist_ok=True)
        os.makedirs('frames/{}'.format(self.experiment_name), exist_ok=True)
        os.makedirs(os.path.join('frames', '{}-{}'.format(self.experiment_name, self.count)))

    def record_frame(self, image_data, timestep):
        img = Image.fromarray(image_data, 'RGB')
        fname = os.path.join('frames', '{}-{}'.format(self.experiment_name, self.count), 'frame-%.10d.png' % timestep)
        img.save(fname)

    def make_movie(self):
        frames = 'frames/{}-{}/frame-%010d.png'.format(self.experiment_name, self.count)
        movie = 'frames/{}/{}.mp4'.format(self.experiment_name, self.count)
        string = "ffmpeg -framerate 24 -y -i {} -r 30 -pix_fmt yuv420p {}".format(frames, movie)
        subprocess.call(string.split())
        shutil.rmtree('frames/{}-{}'.format(self.experiment_name, self.count))
