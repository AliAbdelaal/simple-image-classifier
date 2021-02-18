import os
from typing import Tuple
from torch.utils.data.dataset import Dataset
import torchvision
from torchvision import transforms


class DataWrapper():
    """A wrapper for torchvision datasets.
    """

    @staticmethod
    def get_datasets()->dict:
        """Get supported datasets and it's corresponding function.

        Returns
        -------
        dict
            A key would represent the dataset name
            A value would represent the dataset getter function.
        """
        return {
            "fashion-mnist": DataWrapper.get_fashion_data,
            "cifar10": DataWrapper.get_cifar10_data,
        }
    
    @staticmethod
    def get_input_channels()->dict:
        """get input channels for each of the available datasets. 

        Returns
        -------
        dict
            A key would represent the dataset name
            A value would represent the number of input channels for the given dataset.
            
        """
        return {
            "fashion-mnist": 1,
            "cifar10": 3,
        }


    @staticmethod
    def get_fashion_data()->Tuple[Dataset, Dataset]:
        """get train and test data from fashion mnist dataset

        Returns
        -------
        Tuple[Dataset, Dataset]
            train_set and test_set Dataset objects.
        """
        train_set = torchvision.datasets.FashionMNIST(
        "./data", download=True, transform=transforms.Compose([transforms.ToTensor()]))
        test_set = torchvision.datasets.FashionMNIST(
        "./data", download=True, train=False, transform=transforms.Compose([transforms.ToTensor()]))
        return train_set, test_set

    @staticmethod
    def get_cifar10_data()->Tuple[Dataset, Dataset]:
        """get train and test data from CIFAR10 dataset

        Returns
        -------
        Tuple[Dataset, Dataset]
            train_set and test_set Dataset objects.
        """
        train_set = torchvision.datasets.CIFAR10(
        "./data", download=True, transform=transforms.Compose([transforms.ToTensor()]))
        test_set = torchvision.datasets.CIFAR10(
        "./data", download=True, train=False, transform=transforms.Compose([transforms.ToTensor()]))
        return train_set, test_set
