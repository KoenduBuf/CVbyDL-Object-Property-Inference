
from torch import nn
from networks import NETWORKS
from torchvision import models

def resnet(outc): # needs normalization and image size 224 btw
    model_ft = models.resnet18(pretrained=True)     # Get pretrained model
    num_ftrs = model_ft.fc.in_features              # Replace last layer
    model_ft.fc = nn.Linear(num_ftrs, outc)
    return model_ft

NETWORKS["ResNet"] = resnet
