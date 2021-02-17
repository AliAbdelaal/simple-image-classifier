import torch.nn as nn
import torch.nn.functional as F

class Classifier(nn.Module):
    """A Multi-class Image classifier 
    """

    def __init__(self, labels_count:int, input_channels:int=3):
        super().__init__()
        # the model will work with 28x28 pics W_in = 28
        self.conv1 = nn.Conv2d(input_channels, 6, 5) # W_out = 24
        self.pool = nn.MaxPool2d(2, 2) # W_out = 12
        self.conv2 = nn.Conv2d(6, 16, 5) # W_out = 8
        self.fc1 = nn.Linear(16*4*4, 120) # W_in = 256
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, labels_count)

    def forward(self, x):
        # change any photo to 28x28
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*4*4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
