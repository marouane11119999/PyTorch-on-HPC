from pathlib import Path
import torch
import torchvision                                                       
import torchvision.transforms as transforms
import os.path


from pathlib import Path
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset

def get_dataset(path: Path):
    """
    Load Fashion MNIST dataset and return DataLoaders.

    Args:
        path (Path): Directory to store/load dataset.
        batch_size (int): Batch size for DataLoader.
        num_workers (int): Number of worker processes for data loading.

    Returns:
        train_loader (DataLoader), test_loader (DataLoader)
    """

    # Define transforms: convert to tensor + normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # mean=0.5, std=0.5
    ])

    # Load train/test datasets
    train_dataset = torchvision.datasets.FashionMNIST(
        root=path,
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = torchvision.datasets.FashionMNIST(
        root=path,
        train=False,
        download=True,
        transform=transform
    )
    return train_dataset, test_dataset

def get_dataloaders(train_dataset: Dataset, test_dataset: Dataset , batch_size: int = 64, num_workers: int = 2):
    # Wrap in DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, test_loader