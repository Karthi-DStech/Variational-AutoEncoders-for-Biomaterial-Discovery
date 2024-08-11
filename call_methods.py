import argparse
from typing import Union

import torch

from data.datasets import BaseDataset
from model.models import BaseModel


def make_model(model_name: str, *args, **kwargs) -> BaseModel:
    """
    Create a model from a given name and arguments

    Parameters
    ----------
    model_name: str
        The name of the model to create
    args: list
        The arguments to pass to the model constructor
    kwargs: dict
        The keyword arguments to pass to the model constructor

    Returns
    -------
    model: BaseModel
        The created model

    Raises
    ------
    ValueError
        If the model name is invalid
    """
    model = None
    if model_name.lower() == "vanilla vae":
        from model.vanilla_vae import VanillaVAE

        model = VanillaVAE(*args, **kwargs)

    else:
        raise ValueError(f"Invalid model name: {model_name}")
    print(f"Model {model_name} was created")
    return model


def make_network(network_name: str, *args, **kwargs) -> torch.nn.Module:
    """
    Create a network from a given name and arguments

    Parameters
    ----------
    network_name: str
        The name of the network to create
    args: list
        The arguments to pass to the network constructor
    kwargs: dict
        The keyword arguments to pass to the network constructor

    Returns
    -------
    network: torch.nn.Module
        The created network

    Raises
    ------
    ValueError
        If the network name is invalid
    """
    network = None
    if network_name.lower() == "vanillaencoder":
        from model.encoders import VanillaEncoder

        network = VanillaEncoder(*args, **kwargs)

    elif network_name.lower() == "vanilladecoder":
        from model.decoder import VanillaDecoder

        network = VanillaDecoder(*args, **kwargs)

    else:
        raise ValueError(f"Invalid network name: {network_name}")
    print(f"Network {network_name} was created")
    return network


def make_dataset(dataset_name: str, opt: argparse.Namespace, *args, **kwargs):
    """
    Creates a dataset from the given dataset name

    Parameters
    ----------
    dataset_name: str
        The name of the dataset to create
    opt: argparse.Namespace
        The training options
    *args: list
        The arguments to pass to the dataset constructor
    **kwargs: dict
        The keyword arguments to pass to the dataset constructor

    Returns
    -------
    dataset: BaseDataset
        The created dataset
    """
    dataset = None
    if dataset_name.lower() == "biological":
        from data.topographies import BiologicalObservation

        dataset = BiologicalObservation(opt, *args, **kwargs)

    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}")

    make_dataloader(
        dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers
    )
    dataset.print_dataloader_info()

    print(f"Dataset {dataset_name} was created")
    return dataset


def make_dataloader(dataset: BaseDataset, *args, **kwargs) -> None:
    """
    Creates a dataloader from the given dataset

    Parameters
    ----------
    dataset: torch.utils.data.Dataset
        The dataset to create the dataloader from
    *args: list
        The arguments to pass to the dataloader constructor
    **kwargs: dict
        The keyword arguments to pass to the dataloader constructor

    Returns
    -------
    None
    """
    dataset.dataloader = torch.utils.data.DataLoader(dataset, *args, **kwargs)
