import torch
from torch import nn
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
torch.manual_seed(0) # Set for testing purposes

def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28)):

    image_unflat = image_tensor.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()


def get_generator_block(input_dim, output_dim):

    return nn.Sequential(

        nn.Linear(input_dim, output_dim),
        nn.BatchNorm1d(output_dim),

        nn.ReLU(inplace=True)
    )

def get_noise(n_samples, z_dim, device="cpu"):
    return torch.randn(n_samples, z_dim, device=device)


def get_discriminator_block(input_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.LeakyReLU(0.2)
    )