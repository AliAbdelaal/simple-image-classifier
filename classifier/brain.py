import io
import os
import json
from pathlib import Path
from typing import List, Union
from PIL import Image
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, dataloader
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, classification_report

from classifier.model import Classifier


class Brain():
    """Create a Brain object

    Parameters
    ----------
    base_classifier : nn.Module, optional
        The base classifier to be used, by default Classifier
    epochs : int, optional
        Number of epochs, by default 5
    batch_size : int, optional
        The batch size to be used, by default 100
    learning_rate : float, optional
        The learning rate to be used, by default .001
    """

    def __init__(self, base_classifier: nn.Module = Classifier, epochs: int = 5, batch_size: int = 100, learning_rate: float = .001) -> None:
        self.__base_clf = base_classifier
        self.__trained = False
        self.__epochs = epochs
        self.__batch_size = batch_size
        self.__learning_rate = learning_rate
        self.__device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.__transformation = transforms.Compose([transforms.Resize(28),
                                            transforms.CenterCrop(28),
                                            transforms.ToTensor()])
        self.__data_shape = (28,28)

    def fit(self, train_data: Dataset, input_channels: int = 1):
        """fit the brain's classifier on the given dataset.

        Parameters
        ----------
        train_data : Dataset
            The train dataset, for the model to train on.
        input_channels : int, optional
            The number of color channels of the images, for colored images use 3, by default 1
        """
        # add the transformation to the dataset.
        train_data.transform = self.__transformation

        # save the model properties
        self.__input_channels = input_channels
        self.__labels = train_data.classes

        # instantiate a classifier class and send it to the available device
        self.__clf = self.__base_clf(len(self.__labels), self.__input_channels)
        self.__clf.to(self.__device)

        # wrap training data into loader
        train_data = DataLoader(train_data, batch_size=self.__batch_size)

        # training
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(self.__clf.parameters(), lr=self.__learning_rate)

        for epoch in tqdm(range(self.__epochs)):
            running_loss = 0.0
            with tqdm(enumerate(train_data, 0), total=len(train_data)) as batch:
                for i, data in batch:
                    # get the inputs; data is a list of [inputs, labels]
                    inputs, labels = data[0].to(
                        self.__device), data[1].to(self.__device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward + backward + optimize
                    outputs = self.__clf(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()

                    # update progress bar
                    if not i % 50:
                        acc = accuracy_score(
                            labels.cpu().numpy(),
                            torch.max(outputs, 1)[1].cpu().numpy()
                        )
                        batch.set_postfix(
                            acc=f'{acc:.3f}', loss=f'{running_loss:.3f}')

        self.__clf.eval()
        self.__trained = True
        print(f'done training classifier with acc of {acc:.2f}')

    def prepare_image(self, image_bytes:bytes)->torch.Tensor:
        """transfer an image to tensor, by applying the transformation to it
        - resizing the image
        - crop the center of it
        - transfer it to tensor.

        Parameters
        ----------
        image_bytes : bytes
            The image data.

        Returns
        -------
        torch.Tensor
            The image tensor.
        """
        image = Image.open(io.BytesIO(image_bytes))
        # if the image is PNG and includes ALPHA channel remove it.
        if image.mode == 'RGBA':
            image = image.convert("RGB")
        return self.__transformation(image)

    def predict_image(self, image_data:bytes)->dict:
        """get prediction for the given image data in form of dictionary that represent
        each class and it's probability.

        Parameters
        ----------
        image_data : bytes
            image data.

        Returns
        -------
        dict
            Predictions dictionary represent each class with it's probability and the winning
            class as follows
            {
            "prediction": {
                "label": "class_1", # winning class
                "conf": 0.6868124604225159
            },
            "class_2": 0.0003579219337552786,
            "class_3": 0.00016830475942697376,
            ...
            }
        """
        image = self.prepare_image(image_data)
        return self.predict_tensor(image)

    def predict_batch(self, data: torch.Tensor)->List[str]:
        """predict a batch of tensors (for internal evaluation)

        Parameters
        ----------
        data : torch.Tensor
            A tensor of shape (n, C, W, H)

        Returns
        -------
        List[str]
            list of labels for corresponding inputs
        """
        with torch.no_grad():
            predictions=self.__clf(data.to(self.__device))
            predictions=torch.max(predictions, 1)[1].cpu()
            return [self.__labels[i] for i in predictions]        

    def predict_tensor(self, data: torch.Tensor)->dict:
        """predict the classes for the given tensor in form of dictionary.

        Parameters
        ----------
        data : torch.Tensor
            The input image data, should be in shape (W,H) or (C, W, H).

        Returns
        -------
        dict
            Predictions dictionary represent each class with it's probability and the winning
            class as follows
            {
            "prediction": {
                "label": "class_1", # winning class
                "conf": 0.6868124604225159
            },
            "class_2": 0.0003579219337552786,
            "class_3": 0.00016830475942697376,
            ...
            }
        """

        with torch.no_grad():
            # in case of one image with no channels
            if len(data.shape) == 2:
                data = data.unsqueeze(0).unsqueeze(0)

            # in case 3 dims and model is trained using 1 channel
            if len(data.shape) == 3 and self.__input_channels == 1:
                data = data[0, :, :].unsqueeze(0).unsqueeze(0)
            
            # extend the dims to be of rank 4
            if len(data.data)  == 3:
                data = data.unsqueeze(0)

            predictions = self.__clf(data.to(self.__device))
            max_label_idx = torch.argmax(predictions)
            predictions = torch.softmax(predictions, 1).cpu().squeeze().numpy().tolist()
            res = {}
            res['prediction'] = {
                'label': self.__labels[max_label_idx],
                'conf': predictions[max_label_idx]
            }
            for idx, label in enumerate(self.__labels):
                res[label] = predictions[idx]
            return res

    def export_model(self, model_path: Union[Path, str] = 'checkpoint/'):
        """export the trained model to the given path.

        Parameters
        ----------
        model_path : Union[Path, str], optional
            The directory to save the model to, if not exists it will be created, by default 'checkpoint/'
        """
        if not self.is_trained():
            raise Exception("Model is not trained")

        if not os.path.exists(model_path):
            os.mkdir(model_path)

        # export configs
        configs = {
            "channels": self.__input_channels,
            "image_shape": self.__data_shape,
            "labels": self.__labels
        }
        with open(Path(model_path)/Path("configs.json"), "w+") as file:
            json.dump(configs, file, indent=2)

        # export the model
        torch.save(self.__clf.state_dict(), Path(model_path)/Path('clf.pt'))
        print(f'model exported successfully to {model_path}')

    def load_model(self, model_path: Union[Path, str] = 'checkpoint/'):
        """load a trained classifier into the brain inplace.

        Parameters
        ----------
        model_path : Union[Path, str], optional
            The directory that includes the model state dictionary and configs, by default 'checkpoint/'

        Raises
        ------
        FileNotFoundError
            If the directory is not there.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError("No exported model was found")
        configs = {}
        with open(Path(model_path)/Path("configs.json"), 'r') as file:
            configs = json.load(file)
        self.__data_shape = configs['image_shape']
        self.__input_channels = configs['channels']
        self.__labels = configs['labels']
        model_dict = torch.load(Path(model_path)/Path("clf.pt"))
        self.__clf = self.__base_clf(len(self.__labels), self.__input_channels)
        self.__clf.load_state_dict(model_dict)
        self.__clf.to(self.__device)
        self.__clf.eval()
        self.__trained = True
        print("Model loaded successfully.")

    def is_trained(self)->bool:
        """Check whether the brain's classifier is trained.

        Returns
        -------
        bool
            The state of the classifier.
        """
        return self.__trained

    def get_supported_classes(self)->list:
        """get the model's supported classes. 

        Returns
        -------
        list
            The list of class labels that this brain supports.
        """
        if self.is_trained():
            return self.__labels
        else:
            return []