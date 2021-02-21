# from flask import Flask
import torchvision
from torchvision import transforms
from sklearn.metrics import classification_report

from MultiClassImageClassifier import Brain

# app = Flask(__name__)
brain = Brain(epochs=5)
try:
    brain.load_model()
except:
    print("No saved model was found, training a new one.")
    train_set = torchvision.datasets.FashionMNIST(
        "./data", download=True, transform=transforms.Compose([transforms.ToTensor()]))
    brain.fit(train_set, 1)
    # export the model
    brain.export_model()


test_set = torchvision.datasets.FashionMNIST(
    "./data", download=True, train=False, transform=transforms.Compose([transforms.ToTensor()]))


preds = brain.predict_batch(test_set.data.view(-1, 1, 28, 28).float())
y_true = [test_set.classes[i] for i in test_set.targets]

print(classification_report(y_true, preds))
# test on some images
image = open("/media/ali/Data/dev/vector-ai/project/data/tmp-imgs/dress.jpg", "rb").read()
print(brain.predict_image(image))

