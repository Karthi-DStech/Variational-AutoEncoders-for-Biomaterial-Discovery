from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.networks import BaseNetwork


class VanillaEncoder(BaseNetwork):
    """
    This class implements the Basic Vanilla Encoder for the VAE
    """

    def __init__(
        self,
        encoder_neurons: int,
        latent_dim: int,
        image_size: int,
        in_channels: int,
    ) -> None:
        """
        Initializes the VanillaEncoder Class

        Parameters
        ----------

        encoder_neurons: int
            The number of starting neurons in the encoder layer

        latent_dim: int
            The number of neurons for latent dimensions

        out_channels: int
            The number of output channels

        image_size: int
            The size of the image

        in_channels: int
            The number of input channels

        """

        super().__init__()
        self._name = "VanillaEncoder"
        self._input_dim = image_size * image_size * in_channels

        self.linear1 = nn.Linear(self._input_dim, encoder_neurons)  # Input Img ->1024
        self.linear2 = nn.Linear(
            self.linear1.out_features, self.linear1.out_features // 2
        )  # 1024 -> 512
        self.linear3 = nn.Linear(
            self.linear2.out_features, self.linear2.out_features // 2
        )  # 512 -> 256
        self.linear4 = nn.Linear(
            self.linear3.out_features, self.linear3.out_features // 2
        )  # 256 -> 128

        self.mu = nn.Linear(self.linear4.out_features, latent_dim)  # 128 -> 20
        self.logvar = nn.Linear(self.linear4.out_features, latent_dim)  # 128 -> 20

        """
        The latent_dim is the bottleneck layer which calculates the mean and log variance of the input image
        """

    def _reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for VAE

        Parameters
        ----------
        mu: torch.Tensor
            The mean of the latent space

        logvar: torch.Tensor
            The log variance of the latent space

        Returns
        -------
        torch.Tensor
            The reparameterized tensor
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the network

        Parameters
        ----------
        x: torch.Tensor
            The input image

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            The reparameterized tensor, mean and log variance

        """
        x = x.view(
            x.shape[0], -1
        )  # Flatten the input image. Shape: (batch_size, input_dim)
        x = F.leaky_relu(self.linear1(x), 0.2)
        x = F.leaky_relu(self.linear2(x), 0.2)
        x = F.leaky_relu(self.linear3(x), 0.2)
        x = F.leaky_relu(self.linear4(x), 0.2)
        mu = self.mu(x)
        logvar = self.logvar(x)
        z = self._reparameterize(mu, logvar)

        return z, mu, logvar
