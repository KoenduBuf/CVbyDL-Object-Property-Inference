#!/usr/bin/env python3

import torch
from torch import nn
from utils.TrainValidate import *
from utils.WeightEstimate import *
from networks import *

# Get the datasets
device   = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
datasets = get_datasets(lambda fi: (fi.weight - WEIGHT_MIN) / WEIGHT_RANGE, 224, device)

# Setup the model that we want to train
# model = nn.Sequential(
#     nn.Conv2d(3, 6, 5), nn.ReLU(),   # 3 * 128 * 128 ->  6 * 124 * 124
#     nn.MaxPool2d(2, 2),              #               ->  6 *  62 *  62
#     nn.Conv2d(6, 12, 7), nn.ReLU(),  #               -> 12 *  56 *  56
#     nn.MaxPool2d(2, 2),              #               -> 12 *  28 *  28
#     nn.Conv2d(12, 6, 5), nn.ReLU(),  #               ->  6 *  24 *  24
#     nn.MaxPool2d(2, 2),              #               ->  6 *  12 *  12
#     nn.Flatten(1),
#     nn.Linear(6 * 12 * 12, 120), nn.ReLU(),
#     nn.Linear(120, 84), nn.ReLU(),
#     nn.Linear(84, 1),
#     nn.Flatten(0) # make (n,1) into (n) shape
# )

model = get_network("ResNet", 1, device, flatten=True)
train_the_thing(model, "weight_regression", *datasets, criterion=nn.L1Loss())

# Access how good it is at guessing weights
def un_normalize_weights(outputs):
    guesses = torch.mul(outputs.data, WEIGHT_RANGE)
    guesses = torch.add(guesses, WEIGHT_MIN)
    return guesses

evaluate_weight_inference(model, datasets[1], un_normalize_weights)
