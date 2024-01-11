from Discriminator import Discriminator
from Generateur import Generator
from gan import get_generator_block, get_noise, get_discriminator_block
from torch import nn
import torch


def test_gen_block(in_features, out_features, num_test=1000):
    block = get_generator_block(in_features, out_features)

    # Check the three parts
    assert len(block) == 3
    assert type(block[0]) == nn.Linear
    assert type(block[1]) == nn.BatchNorm1d
    assert type(block[2]) == nn.ReLU

    # Check the output shape
    test_input = torch.randn(num_test, in_features)
    test_output = block(test_input)
    assert tuple(test_output.shape) == (num_test, out_features)
    assert test_output.std() > 0.55
    assert test_output.std() < 0.65


test_gen_block(25, 12)
test_gen_block(15, 28)
print("Success!")

def test_generator(z_dim, im_dim, hidden_dim, num_test=10000):
    gen = Generator(z_dim, im_dim, hidden_dim).get_gen()

    assert len(gen) == 6
    test_input = torch.randn(num_test, z_dim)
    test_output = gen(test_input)

    assert tuple(test_output.shape) == (num_test, im_dim)
    assert test_output.max() < 1, "Make sure to use a sigmoid"
    assert test_output.min() > 0, "Make sure to use a sigmoid"
    assert test_output.min() < 0.5, "Don't use a block in your solution"
    assert test_output.std() > 0.05, "Don't use batchnorm here"
    assert test_output.std() < 0.15, "Don't use batchnorm here"

test_generator(5,10,20)
test_generator(20, 8, 24)
print("Success!")


def test_get_noise(n_samples, z_dim, device="cpu"):
    noise = get_noise(n_samples, z_dim, device)

    assert tuple(noise.shape) == (n_samples, z_dim)
    assert  torch.abs(noise.std() - torch.tensor(1.0)) < 0.01
    assert str(noise.device).startswith(device)

test_get_noise(1000, 100, "cpu")
if torch.cuda.is_available():
    test_get_noise(1000, 32, 'cuda')
print('Success!')

def test_disc_block(in_features, out_features, num_test=10000):
    block = get_discriminator_block(in_features, out_features)

    assert len(block) == 2
    test_input = torch.randn(num_test, in_features)
    test_output = block(test_input)


    assert tuple(test_output.shape) == (num_test, out_features)

    assert -test_output.min() / test_output.max() > 0.1
    assert -test_output.min() / test_output.max() < 0.3
    assert test_output.std() > 0.3
    assert test_output.std() < 0.5

test_disc_block(25,12)
test_disc_block(15,18)
print("Success!")


def test_discriminator(z_dim, hidden_dim, num_test=100):
    disc = Discriminator(z_dim, hidden_dim).get_disc()

    assert len(disc) == 4


    test_input = torch.randn(num_test, z_dim)
    test_output = disc(test_input)
    assert tuple(test_output.shape) == (num_test, 1)


    assert not isinstance(disc[-1], nn.Sequential)

test_discriminator(5,10)
test_discriminator(20, 8)
print("Success!")

