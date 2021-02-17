import io
import os
import json
from pathlib import Path
from typing import Union
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

        train_data.transform = self.__transformation

        self.__input_channels = input_channels

        # save labels
        self.__labels = train_data.classes

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

    def prepare_image(self, image_bytes):
        my_transforms = transforms.Compose([transforms.Resize(self.__data_shape[0]),
                                            transforms.CenterCrop(
                                                self.__data_shape[0]),
                                            transforms.ToTensor()])
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode == 'RGBA':
            image = image.convert("RGB")
        return my_transforms(image)

    def predict_image(self, image_data):
        image = self.prepare_image(image_data)
        return self.predict_tensor(image)

    def predict_batch(self, data: torch.Tensor):
        """predict a batch of tensors (for internal evaluation)

        Parameters
        ----------
        data : torch.Tensor
            A tensor of shape (n, C, W, H)

        Returns
        -------
        list
            list of labels for corresponding inputs
        """
        with torch.no_grad():
            predictions=self.__clf(data.to(self.__device))
            predictions=torch.max(predictions, 1)[1].cpu()
            return [self.__labels[i] for i in predictions]        

    def predict_tensor(self, data: torch.Tensor)->dict:
        # in case of one image with no channels
        with torch.no_grad():
            if len(data.shape) == 2:
                data = data.unsqueeze(0).unsqueeze(0)

            # in case 3 dims and model is trained using 1 channel
            if len(data.shape) == 3 and self.__input_channels == 1:
                data = data[0, :, :].unsqueeze(0).unsqueeze(0)
            
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

    def is_trained(self):
        return self.__trained

    def get_supported_classes(self):
        if self.is_trained():
            return self.__labels
        else:
            return []