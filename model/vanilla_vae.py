import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from call_methods import make_network
from model.encoders import VanillaEncoder
from model.models import BaseModel
from utils.losses import kld_loss


class VanillaVAE(BaseModel):
    """
    This class implements the VanillaVAE model.
    """

    def __init__(self, opt: argparse.Namespace) -> None:
        """
        Initializes the VanillaVAE class.

        Parameters
        ----------
        opt: argparse.Namespace
            The training options
        """
        super().__init__(opt)
        self._name = "VanillaVAE"
        self._networks = None
        self._create_networks()
        self._print_num_params()

        if self._is_train:
            self._make_loss()
            self._make_optimizer()

    def _create_networks(self) -> None:
        """
        Creates the networks of the model
        """
        self._encoder = make_network(
            network_name="VanillaEncoder",
            encoder_neurons=self._opt.encoder_neurons,
            latent_dim=self._opt.latent_dim,
            image_size=self._opt.image_size,
            in_channels=self._opt.in_channels,
        )

        self._encoder.init_weights(self._opt.init_type)

        self._decoder = make_network(
            network_name="VanillaDecoder",
            latent_dim=self._opt.latent_dim,
            decoder_neurons=self._opt.decoder_neurons,
            out_channels=self._opt.out_channels,
            image_size=self._opt.image_size,
        )

        self._decoder.init_weights(self._opt.init_type)
        self._networks = [self._encoder, self._decoder]
        self._send_to_device(self._networks)

    def _make_loss(self) -> None:
        """
        Creates the loss function for the model
        """

        self._recon_loss = nn.MSELoss()

        self._kld_loss = kld_loss

    def _forward_vae(self) -> torch.Tensor:
        """
        Forward pass of the VAE

        Returns
        -------
        torch.Tensor
            The overall loss of the VAE
        """

        z, mu, logvar = self._encoder(self._real_images)

        self._reconstruction = self._decoder(z)

        self._re_loss = self._recon_loss(self._reconstruction, self._real_images)

        self._k_loss = self._kld_loss(mu, logvar)

        self._loss = (1 - self._opt.beta * self._re_loss) + (
            self._opt.beta * self._k_loss
        )

        return self._loss

    def _get_current_performance(self, do_visualization: bool = False) -> None:
        """
        Gets the current performance of the model

        Parameters
        ----------
        do_visualization: bool
            Whether to visualize the images

        Returns
        -------
        None
        """
        self._current_vae_performance = {
            "Reconstruction Loss": self._re_loss.item(),
            "KLD Loss": self._k_loss.item(),
            "Overall Loss": self._loss.item(),
        }

        self._current_performance = {**self._current_vae_performance}

        if do_visualization:
            self.vis_data = self._get_generated_image(n_samples=self._opt.n_vis_samples)

        vae_lr = self._optimizer.param_groups[0]["lr"]

        self.performance = {"losses": self._current_performance, "lr": vae_lr}
