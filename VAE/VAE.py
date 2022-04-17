import torch.nn as nn
from torch.nn import functional as F
import torch
import os

from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.autograd import Variable

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets


class VAE(nn.Module):
    def __init__(self, in_channels: int, latent_dim: int):
        super(VAE, self).__init__()
        models = []

        # ------------------
        # Build the Encoder
        # ------------------
        latent_dims = [256, 512]
        for dim in latent_dims:
            models.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=dim,
                              kernel_size=(3, 3), stride=(1,), padding='same'),
                    nn.InstanceNorm2d(num_features=dim),
                    nn.LeakyReLU(negative_slope=0.2)
                )
            )
            in_channels = dim
            models.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=dim,
                              kernel_size=(3, 3), stride=(2,), padding=(1, 1)),
                    nn.InstanceNorm2d(num_features=dim),
                    nn.LeakyReLU(negative_slope=0.2)
                )
            )
            in_channels = dim

        models.append(
            nn.Sequential(
                nn.Conv2d(512, out_channels=1024,
                          kernel_size=(3, 3), stride=(1,), padding='same'),
                nn.LeakyReLU(negative_slope=0.2)
            )
        )
        models.append(
            nn.Sequential(
                nn.Conv2d(1024, out_channels=1024,
                          kernel_size=(3, 3), stride=(2,), padding=(1, 1)),
                nn.LeakyReLU(negative_slope=0.2),
            )
        )

        # Some instances
        self.encoder = nn.Sequential(*models)
        self.skip_encoder = nn.Sequential(*models[-2:])
        self.fc_mu = nn.Linear(16 * 1024, latent_dim)
        self.fc_var = nn.Linear(16 * 1024, latent_dim)
        self.latent_dim = latent_dim
        # -----------------------
        # Build the discriminator
        # ------------------------
        models = [nn.Conv2d(in_channels=3, out_channels=64,
                            kernel_size=(1, 1), stride=(1,), padding='same')]
        latent_dims = [128, 256, 512, 512, 512]
        input_channels = 64

        for latent_dim in latent_dims:
            for _ in range(2):
                models.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels=input_channels, out_channels=latent_dim,
                                  kernel_size=(3, 3), stride=(1,), padding=(1, 1)),
                        nn.InstanceNorm2d(num_features=latent_dim),
                        nn.LeakyReLU(negative_slope=0.2)
                    )
                )
                input_channels = latent_dim
            models.append(nn.Upsample(scale_factor=0.5, mode='bilinear'))

        models.append(
            nn.Sequential(
                nn.Conv2d(in_channels=512, out_channels=512,
                          kernel_size=(3, 3), stride=(1,), padding='same'),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Conv2d(in_channels=512, out_channels=512,
                          kernel_size=(4, 4), stride=(1,), padding='valid'),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Flatten(),
                nn.Linear(in_features=512, out_features=1)
            )
        )
        self.discriminator = nn.Sequential(*models)

    def decoder(self, content_feat, style_feat):
        """
        We want to build the complex structure of the
        decoder here
        """

        def get_model(input_channel: int, output_channels: int, feat, is_upsample: bool):
            if is_upsample:
                models = [nn.Upsample(scale_factor=2, mode='biliear')]
            else:
                models = []
            models.append(nn.Sequential(
                nn.Conv2d(in_channels=input_channel, out_channels=output_channels,
                          kernel_size=(3, 3), stride=(1,), padding='same'),
                nn.LeakyReLU(negative_slope=0.2)
            ))
            return nn.Sequential(*models)

        input_channels = 512
        latent_dims = [512, 512, 256, 256, 128]
        for latent_dim in latent_dims:
            for i in range(2):
                model = get_model(input_channels, latent_dim, content_feat, i == 0)
                content_feat = model(content_feat)
                content_feat = adaptive_instance_normalization(content_feat, style_feat)
                input_channels = latent_dim
        content_feat = nn.Conv2d(in_channels=latent_dims[-1], out_channels=3,
                                 kernel_size=(1, 1), stride=(1,), padding='same')(content_feat)
        content_feat = nn.Tanh()(content_feat)
        return content_feat

    def encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, content_feat, style_feat):
        content_feat = torch.nn.Linear(in_features=100, out_features=16 * 1024)
        content_feat = content_feat.view(-1, 512, 4, 4)
        result = self.decoder(content_feat, style_feat)
        return result

    def discrime(self, input):
        return self.discriminator(input)

    def generate_style_feat(self, input):
        layers = [32, 64, 128, 128]
        models = []
        in_channels = self.latent_dim
        for layer in layers:
            models.append(nn.Linear(in_features=in_channels, out_features=layer))
            in_channels = layer
        return nn.Sequential(*models)(input)

    def re_parameterize(self, mu, log_var):
        """
        :param mu: (Tensor) Mean of the latent Gaussian
        :param log_var: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std, dtype=torch.float32)
        return eps * std + mu

    def forward(self, input):
        mu, log_val = self.encode(input)
        z = self.re_parameterize(mu, log_val)
        style_feat = self.generate_style_feat(z)
        result = self.decode(z, style_feat)
        return [result, input, mu, log_val]


# -----------------------------
# AdaIn layer helper functions
# ----------------------------

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (size == 4)  # (N, C, W, H)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_std, feat_mean


def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_std, style_mean = calc_mean_std(feat=style_feat)
    content_std, content_mean = calc_mean_std(feat=content_feat)

    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


def load_data():
    # FIXME
    return [0]


if __name__ == '__main__':
    os.makedirs("./vae_data", exist_ok=True)
    dataloader = load_data()

    gan_loss_func = torch.nn.BCELoss()
    # Create a VAE
    vae = VAE(in_channels=32, latent_dim=100)

    # Optimizer
    Optimizer = Adam(vae.parameters(), lr=0.001, betas=(0, 0.99))

    for epoch in range(50):
        for i, (imgs, _) in enumerate(dataloader):
            # Build the computation Graph
            real_imgs = nn.Upsample(scale_factor=0.25, mode='bilinear')(imgs.type(torch.FloatTensor))
            input_imgs = Variable(real_imgs, requires_grad=True)

            shape1 = (real_imgs.size()[0], 1)
            valid = Variable(torch.ones(size=shape1, dtype=torch.float32), requires_grad=False)
            fake = Variable(torch.zeros(size=shape1, dtype=torch.float32), requires_grad=False)

            # -------
            #  Train
            # -------
            # Compute g_loss
            Optimizer.zero_grad()
            g_res = vae.forward(input_imgs)
            generate_imgs = g_res[0]
            mu, log_var = g_res[2:]
            g_loss1 = gan_loss_func(vae.discrime(generate_imgs), valid)  # GAN generator loss
            # KL loss
            g_loss2 = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
            g_loss = g_loss1 + g_loss2

            # Compute d_loss
            true_to_false = gan_loss_func(vae.discrime(generate_imgs), valid)
            false_to_true = gan_loss_func(vae.discrime(generate_imgs), fake)

            d_loss = (true_to_false + false_to_true) / 2

            loss = g_loss + d_loss
            loss.backward()
            Optimizer.step()

            if i % 500 == 0:
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, 50, i, len(dataloader), d_loss.item(), g_loss.item())
                )
