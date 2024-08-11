import torch
import torch.nn as nn
import torch.nn.functional as F

from model.networks import BaseNetwork


class VanillaDecoder(BaseNetwork):
    """
    This class implements the Basic Vanilla Decoder for the VAE
    """

    def __init__(
        self,
        latent_dim: int,
        decoder_neurons: int,
        out_channels: int,
        image_size: int,
    ) -> None:
        """
        Initializes the VanillaDecoder Class

        Parameters
        ----------

        latent_dim: int
            The number of neurons for latent dimensions

        decoder_neurons: int
            The number of neurons in the decoder layer

        out_channels: int
            The number of output channels

        image_size: int
            The size of the image

        """

        """
        The linear layers for the VanillaDecoder class. We are using 4 linear layers and we have to define 
        the number of decoder neurons in the first linear layer.
        """
        super().__init__()
        self._name = "VanillaDecoder"
        self._image_size = image_size
        self._out_channels = out_channels
        self._out_size = image_size * image_size * out_channels

        self.linear1 = nn.Linear(latent_dim, decoder_neurons)  # 20 -> 128
        self.linear2 = nn.Linear(
            self.linear1.out_features, self.linear1.out_features * 2
        )  # 128 -> 256
        self.linear3 = nn.Linear(
            self.linear2.out_features, self.linear2.out_features * 2
        )  # 256 -> 512
        self.linear4 = nn.Linear(
            self.linear3.out_features, self.linear3.out_features * 2
        )  # 512 -> 1024
        self.linear5 = nn.Linear(
            self.linear4.out_features, self._out_size
        )  # 1024 -> output image size

        """
        The final linear layer is the output of the image size.
        """

    def forward(self, z):
        """
        The forward pass of the VanillaDecoder class

        Parameters
        ----------
        z: torch.Tensor
            The input tensor to the network

        Returns
        -------
        torch.Tensor
            The output tensor of the network
        """
        x = F.leaky_relu(self.linear1(z), 0.2)
        x = F.leaky_relu(self.linear2(x), 0.2)
        x = F.leaky_relu(self.linear3(x), 0.2)
        x = F.leaky_relu(self.linear4(x), 0.2)

        x = torch.tanh(self.linear5(x))

        return x.view(-1, self._out_channels, self._image_size, self._image_size)

    """
    The forward pass of the VanillaDecoder class
    """
