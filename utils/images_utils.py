from torchvision import transforms


def get_transform(img_size: int, mean: float, std: float) -> transforms.Compose:
    """
    Gets the transforms for the dataset

    Parameters
    ----------
    img_size: int
        The size of the image
    mean: float
        The mean of the dataset
    std: float
        The standard deviation of the dataset

    Returns
    -------
    transforms.Compose
        The transforms for the dataset
    """
    transform = transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize((mean,), (std,)),
        ]
    )
    return transform
