import os
from PIL import Image
import subprocess
from tempfile import TemporaryDirectory

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
    def __init__(self):
        self.frames = []
    def __enter__(self):
        self.tmpdir = TemporaryDirectory()
        return self

    def __exit__(self, exc, value, tb):
        self.tmpdir.cleanup()

    def record_frame(self, image_data, timestep):
        img = Image.fromarray(image_data, 'RGB')
        self.frames.append(img)
        fname = os.path.join(self.tmpdir.name, 'frame-%.10d.png' % timestep)
        img.save(fname)

    def make_movie(self, final_path):
        frames = os.path.join(self.tmpdir.name, 'frame-%010d.png')
        movie = '{}.mp4'.format(final_path)
        string = "ffmpeg -framerate 24 -y -i {} -r 30 -pix_fmt yuv420p {}".format(frames, movie)
        subprocess.call(string.split())

