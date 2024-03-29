import torch

from gan import get_noise


def get_disc_loss(gen, disc, criterion, real, num_images, z_dim, device):
    noise = torch.randn(num_images, z_dim, device=device)
    fake = gen(noise)

    disc_fake_pred = disc(fake.detach())
    disc_fake_labels = torch.zeros_like(disc_fake_pred, device=device)
    disc_fake_loss = criterion(disc_fake_pred, disc_fake_labels)

    disc_real_pred = disc(real)
    disc_real_labels = torch.ones_like(disc_real_pred, device=device)
    disc_real_loss = criterion(disc_real_pred, disc_real_labels)


    disc_loss = (disc_real_loss+disc_fake_loss)/2

    return disc_loss


def get_gen_loss(gen, disc, criterion, num_images, z_dim, device):
    noise = get_noise(num_images, z_dim, device)
    fake_images = gen(noise)

    disc_fake_pred = disc(fake_images)
    gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))

    return gen_loss