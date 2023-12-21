import subprocess

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import MNIST


def get_mnist_dataloaders(
    data_path: str = "save/data",
    batch_size: int = 32,
) -> tuple[DataLoader, DataLoader, Dataset, Dataset]:
    """Get the MNIST dataloaders.  This is used as an example dataset.  You will have to modify this to work with your own data.

    Args:
        batch_size (int): Batch size to retrieve on each call to `__next__()`.

    Returns:
        tuple[DataLoader, DataLoader, Dataset, Dataset]: The iterators to return.

    .. warning:: Be Careful!
        `DataLoader` and `Dataset` are very different types of iterators.  A `DataLoader` will return a batch tensor whereas a `Dataset` will return a single tensor.  The user has to be aware of what kind of iterator he is using or risk causing problems.
    """

    # Make dirs
    subprocess.run(["mkdir", "-p", data_path])

    # Apply transformations to data to ensure that data is normalized between -1 and 1
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    # Download the MNIST data and apply transforms
    dataset_train = MNIST(root=data_path, train=True, download=True, transform=transform)
    dataset_val = MNIST(root=data_path, train=False, download=True, transform=transform)

    # Shuffle the data loaders
    dataloader_train = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_val = DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=False)

    return dataloader_train, dataloader_val, dataset_train, dataset_val
