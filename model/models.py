import argparse
from typing import Dict, Tuple, Union

import torch
import torch.nn as nn


class BaseModel(nn.Module):
    """
    This class is an abstract class for models
    """

    def __init__(self, opt: argparse.Namespace) -> None:
        """
        Initializes the BaseModel class

        Parameters
        ----------
        opt: argparse.Namespace
            The options to initialize the model

        """

        super().__init__()

        self._opt = opt
        self._name = "BaseModel"
        self._is_train = self._opt.is_train
        self._encoder = torch.nn.Module()
        self._decoder = torch.nn.Module()
        self._networks = [self._encoder, self._decoder]
        self._reconstruction = torch.Tensor()
        self.performance: Dict = {}

        self._get_device()

    def __str__(self) -> str:
        """
        Returns the string representation of the model
        """
        return self._name

    def _create_networks(self) -> None:
        """
        Creates the networks of the model

        Raises
        ------
        NotImplementedError
            if the method is not implemented
        """

        raise NotImplementedError

    def _make_loss(self) -> None:
        """
        Makes the loss for the model

        Raises
        ------
        NotImplementedError"""

        raise NotImplementedError

    def _forward_vae(self) -> torch.Tensor:
        """
        Forward pass of the vae

        Raises
        ------
        NotImplementedError
            if the method is not implemented
        """

        raise NotImplementedError()

    def _get_current_performance(self) -> None:
        """
        Gets the current performance of the model

        Raises
        ------
        NotImplementedError
            if the method is not implemented
        """

        raise NotImplementedError

    @property
    def name(self) -> str:
        """
        Returns the name of the model
        """
        return self._name

    def train(self, do_visualization: bool = False) -> None:
        """
        Trains the model

        Parameters
        ----------
        train_generator: bool
            Whether to train the generator
        do_visualization: bool
            Whether to visualize the performance

        Returns
        -------
        None
        """
        self._encoder.train()
        self._decoder.train()

        self._loss = self._forward_vae()
        self._optimizer.zero_grad()
        self._loss.backward()
        self._optimizer.step()
        self._get_current_performance(do_visualization)

    def _print_num_params(self) -> None:
        """
        Prints the number of parameters of the model

        Raises
        ------
        ValueError
            If the networks are not created yet
        """
        if self._networks is None:
            raise ValueError("Networks are not created yet")
        else:
            for network in self._networks:
                all_params, trainable_params = network.get_num_params()
                print(
                    f"{network.name} has {all_params/1e3:.1f}K parameters ({trainable_params/1e3:.1f}K trainable)"
                )

    def _make_optimizer(self) -> None:
        """
        Makes the optimizer for the model

        Raises
        ------
        NotImplementedError
            if the optimizer is not implemented
        """
        params = list(self._encoder.parameters()) + list(self._decoder.parameters())

        if self._opt.optimizer == "adam":
            self._optimizer = torch.optim.Adam(
                params,
                lr=self._opt.lr,
                betas=(self._opt.adam_beta1, self._opt.adam_beta2),
            )

        elif self._opt.optimizer == "rmsprop":
            self._optimizer = torch.optim.RMSprop(
                params,
                lr=self._opt.lr,
                weight_decay=self._opt.weight_decay,
            )

        else:
            raise NotImplementedError(f"{self._opt.optimizer} is not implemented yet")

    def set_input(self, data: torch.Tensor) -> None:
        """
        Sets the input of the model
        Parameters
        ----------
        data: torch.Tensor
            The input data
        Returns
        -------
        None
        """
        self._real_images, self._real_labels = data
        self._real_images = self._real_images.to(self._device).float()
        self._real_labels = self._real_labels.to(self._device).long()
        self._gen_labels = (
            torch.randint(0, self._opt.n_classes, (self._real_images.size(0),))
            .to(self._device)
            .long()
        )

    def _get_device(self) -> None:
        """
        Gets the device to train the model
        """
        self._device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        print(f"Using device: {self._device}")

    def _send_to_device(
        self, data: Union[torch.Tensor, list]
    ) -> Union[torch.Tensor, list]:
        """
        Sends the data to the device

        Parameters
        ----------
        data: torch.Tensor
            The data to send to the device

        Returns
        -------
        torch.Tensor
            The data in the device
        """
        if isinstance(data, list):
            return [x.to(self._device) for x in data]
        else:
            return data.to(self._device)

    def save_networks(self, path: str, epoch: Union[int, str]) -> None:
        """
        Saves the networks

        Parameters
        ----------
        path: str
            The path to save the networks
        epoch: Union[int, str]
            The current epoch

        Returns
        -------
        None
        """
        for network in self._networks:
            network.save(path, epoch)

    def _get_generated_image(self, n_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gets the generated images

        Parameters
        ----------
        n_samples: int
            The number of samples to save

        Returns
        -------
        torch.Tensor
            The generated data
        """

        self._vis_images = self._reconstruction[:n_samples]
        self._vis_labels = self._gen_labels[:n_samples]

        mean = self._opt.normalization_values["mean"]
        std = self._opt.normalization_values["std"]
        self._vis_images = self._vis_images * std + mean
        self.vis_data = (self._vis_images, self._vis_labels)
        return self.vis_data
