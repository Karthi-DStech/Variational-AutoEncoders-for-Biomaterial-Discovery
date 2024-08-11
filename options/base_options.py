import argparse
import ast
from typing import Dict, Union


class BaseOptions:
    """
    This class defines options used during all types of experiments.
    It also implements several helper functions such as parsing, printing, and saving the options.
    """

    def __init__(self) -> None:
        self._parser = argparse.ArgumentParser()

        self._initialized = False  # Initially, Flags are set to false

        self._float_or_none = self.float_or_none
        self._list_or_none = self.list_or_none

    def initialize(self) -> None:
        """
        Initializes the BaseOptions class

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        self._parser.add_argument(
            "-en",
            "--experiment_name",
            type=str,
            default="1_Vanilla_VAE",
            help="Name of the experiment!",
        )

        self._parser.add_argument(
            "--dataset_name",
            type=str,
            required=False,
            default="biological",
            help="dataset name",
            choices=["biological"],
        )

        self._parser.add_argument(
            "--image_folder",
            type=str,
            default="../Datasets/Topographies/raw/FiguresStacked Same Size 4X4",
            help="Path for loading the image folder (dataset)",
        )

        self._parser.add_argument(
            "--label_path",
            type=str,
            default="../Datasets/biology_data/TopoChip/AeruginosaWithClass.csv",
            help="Path to the label file",
        )

        self._parser.add_argument(
            "--log_dir",
            type=str,
            required=False,
            default="./logs",
            help="Change to path to the log directory",
        )

        self._parser.add_argument(
            "--image_size",
            type=int,
            required=False,
            default=224,
            help="image size",
        )

        self._parser.add_argument(
            "--image_scale",
            type=int,
            default=1,
            help="Set the image into 1 for grayscale",
        )

        self._parser.add_argument(
            "--normalization_values",
            type=lambda x: ast.literal_eval(x),
            required=False,
            default={"mean": 0.5, "std": 0.5},
            help="dataset parameters",
        )

        self._parser.add_argument(
            "--img_type", type=str, required=False, default="png", help="image type"
        )

        self._parser.add_argument(
            "--n_classes",
            type=int,
            required=False,
            default=5,
            help="number of classes for conditional VAEs",
        )

        self._parser.add_argument(
            "-bs",
            "--batch_size",
            type=int,
            default=32,
            help="Number of images per batch",
        )

        self._parser.add_argument(
            "--in_channels",
            type=int,
            required=False,
            default=1,
            help="number of input channels",
        )

        self._parser.add_argument(
            "--out_channels",
            type=int,
            required=False,
            default=1,
            help="number of output channels",
        )

        self._parser.add_argument(
            "--num_workers",
            type=int,
            required=False,
            default=4,
            help="number of workers",
        )

        self._parser.add_argument(
            "--save_image_frequency",
            type=int,
            required=False,
            default=100,
            help="save image frequency",
        )

        self._parser.add_argument(
            "--print_freq", type=int, required=False, default=1, help="print frequency"
        )

        self._parser.add_argument(
            "--model_save_frequency",
            type=int,
            required=False,
            default=50,
            help="model save frequency",
        )

        self._parser.add_argument(
            "--n_vis_samples",
            type=int,
            default=32,
            help="number of samples to visualize",
        )

        self._parser.add_argument(
            "--seed", type=int, required=False, default=1221, help="random seed"
        )

        self._initialized = True
        self._is_train = False

    def parse(self) -> argparse.Namespace:  # To save the commands
        """
        Parameters
        ----------
        None

        Returns
        -------
        opt: argparse.Namespace
            The parsed arguments

        """

        if not self._initialized:
            self.initialize()

        self._opt = self._parser.parse_args()
        self._opt.is_train = self._is_train

        args = vars(self._opt)
        self._print(args)

        return self._opt

    def _print(self, args: Dict) -> None:
        """
        Prints the arguments passed to the script

        Parameters
        ----------
        args: dict
            The arguments to print

        Returns
        -------
        None
        """

        print("------------ Options -------------")
        for k, v in args.items():
            print(f"{str(k)}: {str(v)}")
        print("-------------- End ---------------")

    def float_or_none(self, value: str) -> Union[float, None]:
        """
        Converts a string to float or None

        Parameters
        ----------
        value: str
            The value to convert

        Returns
        -------
        float
            The converted value
        """

        if value.lower() == "none":
            return None
        try:
            return float(value)
        except ValueError:
            raise argparse.ArgumentTypeError(
                "Invalid float or 'none' value: {}".format(value)
            )

    def list_or_none(self, value: str) -> Union[list, None]:
        """
        Converts a string to list or None

        Parameters
        ----------
        value: str
            The value to convert

        Returns
        -------
        list
            The converted value
        """

        if value.lower() == "none":
            return None
        try:
            return ast.literal_eval(value)
        except ValueError:
            raise argparse.ArgumentTypeError(
                "Invalid list or 'none' value: {}".format(value)
            )


# TODO:
# - Set Image Dimensions
