import torch


def kld_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    Kullback-Leibler divergence loss

    Parameters
    ----------
    mu : torch.Tensor
        Mean of the latent space
    logvar : torch.Tensor
        Logarithm of the variance of the latent space

    Returns
    -------
    torch.Tensor
        Kullback-Leibler divergence loss
    """

    return torch.mean(
        -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1), dim=0
    )
