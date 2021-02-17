import os
from typing import Tuple
import pandas as pd
from torch.utils.data.dataset import Dataset
import torchvision
from torchvision import transforms


class DataWrapper():

    @staticmethod
    def get_datasets():
        return {
            "fashion-mnist": DataWrapper.get_fashion_data,
            "cifar10": DataWrapper.get_cifar10_data,
            "emnist": DataWrapper.get_emnist_data
        }
    
    @staticmethod
    def get_input_channels():
        return {
            "fashion-mnist": 1,
            "cifar10": 3,
            "emnist": 1
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

    @staticmethod
    def get_emnist_data()->Tuple[Dataset, Dataset]:
        """get train and test data from emnist dataset

        Returns
        -------
        Tuple[Dataset, Dataset]
            train_set and test_set Dataset objects.
        """
        train_set = torchvision.datasets.EMNIST(
        "./data", download=True, transform=transforms.Compose([transforms.ToTensor()]))
        test_set = torchvision.datasets.EMNIST(
        "./data", download=True, train=False, transform=transforms.Compose([transforms.ToTensor()]))
        return train_set, test_set
    