#!/usr/bin/env python3

# Load and show some images

import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from random import randrange
import sys

from utils.DataSet import *
from utils.TrainValidate import *


transforms_test = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
transforms_train = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomVerticalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.RandomErasing(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

classes = FruitImageDataset.DEFAULT_TYPES
train_set = FruitImageDataset("../images/auto128x128",
    transform=transforms_train)
test_set  = train_set.split_1_in_n(10)
test_set.transform = transforms_test
print_summary_tables( (train_set, "TRAIN"), (test_set, "TEST") )



# show random images and print labels
# showloader = torch.utils.data.DataLoader(train_set,
#         batch_size=4, shuffle=True, num_workers=2)
# images, labels = iter(showloader).next()
# print(' '.join(classes[lbl] for lbl in labels))
# grid_img = torchvision.utils.make_grid(images)
# grid_img = grid_img / 2 + 0.5 # unnormalize
# plt.imshow(np.transpose(grid_img.numpy(), (1, 2, 0)))
# plt.show()
# sys.exit(0)


# Define the NN, This one is from:
# https://godatadriven.com/blog/how-to-build-your-first-image-classifier-using-pytorch/
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv = nn.Conv2d(3, 18, kernel_size=3, stride=1, padding=1)
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
#         self.fc1 = nn.Linear(18 * 16 * 16, 64)
#         self.fc2 = nn.Linear(64, len(classes)) # end with amount of classes
#
#     def forward(self, x):
#         x = F.relu(self.conv(x))
#         x = self.pool(x)
#         x = x.view(-1, 18 * 16 * 16)
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x



# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
# class Net(nn.Module):
    # def __init__(self):
        # super().__init__()
        # self.conv1 = nn.Conv2d(3, 6, 5)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.conv2 = nn.Conv2d(6, 16, 5)
        # self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, 10)

    # def forward(self, x):
        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        # x = torch.flatten(x, 1) # flatten all dimensions except batch
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        # return x

# model = Net()

# model = nn.Sequential(
#     nn.Conv2d(3, 6, 5), nn.ReLU(),   # 3 * 64 * 64 ->  6 * 60 * 60
#     nn.MaxPool2d(2, 2),              #             ->  6 * 30 * 30
#     nn.Conv2d(6, 16, 7), nn.ReLU(),  #             -> 16 * 24 * 24
#     nn.MaxPool2d(2, 2),              #             -> 16 * 12 * 12
#     nn.Conv2d(16, 16, 5), nn.ReLU(), #             -> 16 *  8 *  8
#     nn.MaxPool2d(2, 2),              #             -> 16 *  4 *  4
#     nn.Flatten(1),
#     nn.Linear(16 * 4 * 4, 120), nn.ReLU(),
#     nn.Linear(120, 84), nn.ReLU(),
#     nn.Linear(84, 10)
# )

model = nn.Sequential(
    nn.Conv2d(3, 6, 5), nn.ReLU(),   # 3 * 128 * 128 ->  6 * 124 * 124
    nn.MaxPool2d(2, 2),              #               ->  6 *  62 *  62
    nn.Conv2d(6, 12, 7), nn.ReLU(),  #               -> 12 *  56 *  56
    nn.MaxPool2d(2, 2),              #               -> 12 *  28 *  28
    nn.Conv2d(12, 6, 5), nn.ReLU(), #                ->  6 *  24 *  24
    nn.MaxPool2d(2, 2),              #               ->  6 *  12 *  12
    nn.Flatten(1),
    nn.Linear(6 * 12 * 12, 120), nn.ReLU(),
    nn.Linear(120, 84), nn.ReLU(),
    nn.Linear(84, 7)
)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# cross_validate(model, criterion, optimizer, train_set)
for r in range(15):
    train(model, criterion, optimizer, train_set, epochs = 4)
    validate(model, test_set)

print("\n")
validate(model, test_set, show_for_classes=
    FruitImageDataset.DEFAULT_TYPES)
