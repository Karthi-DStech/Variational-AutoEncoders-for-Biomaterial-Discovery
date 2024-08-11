from options.base_options import BaseOptions


class TrainOptions(BaseOptions):
    """Train options"""

    def __init__(self) -> None:
        super().__init__()

    def initialize(self) -> None:
        """Initializes the TrainOptions class"""

        BaseOptions.initialize(self)

        self._parser.add_argument(
            "--model_name",
            type=str,
            required=False,
            default="Vanilla VAE",
            help="model name",
            choices=["Vanilla VAE"],
        )

        self._parser.add_argument(
            "--n_epochs",
            type=int,
            default=200,
            help="Number of epochs",
        )

        self._parser.add_argument(
            "--latent_dim",
            type=int,
            required=False,
            default=20,
            help="The latent dimension neurons for the encoder-decoder",
        )

        self._parser.add_argument(
            "--decoder_neurons",
            type=int,
            default=128,
            help="Number of starting neurons for Decoder",
        )

        self._parser.add_argument(
            "--encoder_neurons",
            type=int,
            default=1024,
            help="Number of starting neurons for Encoder",
        )

        self._parser.add_argument(
            "--init_type",
            type=str,
            required=False,
            default="xavier_normal",
            help="initialization type",
            choices=["normal", "xavier_normal", "kaiming_normal"],
        )

        self._parser.add_argument(
            "--optimizer",
            type=str,
            required=False,
            default="rmsprop",
            help="optimizer",
            choices=["adam", "rmsprop"],
        )

        self._parser.add_argument(
            "--loss_function",
            type=str,
            required=False,
            default="MSELoss",
            help="loss function",
        )

        self._parser.add_argument(
            "--lr",
            type=float,
            required=False,
            default=0.0003,
            help="learning rate",
        )

        self._parser.add_argument(
            "--weight_decay",
            type=int,
            required=False,
            default=0.0001,
            help="number of epochs",
        )

        self._parser.add_argument(
            "--adam_beta1",
            type=float,
            required=False,
            default=0.5,
            help="values for adam beta1",
        )
        self._parser.add_argument(
            "--adam_beta2",
            type=float,
            required=False,
            default=0.999,
            help=" values for adam beta2",
        )
        self._parser.add_argument(
            "--beta",
            type=float,
            required=False,
            default=0.5,
            help="beta value for the VAE",
        )

        self._is_train = True
