#!/usr/bin/env python3

from torch import nn
from utils.TrainValidate import *
from utils.WeightEstimate import *

# Currently our best setup for a single network
datasets = get_datasets("weight_win 5")
disp_labels = [ f"{5*i} - {5*i+5}" for i in range(50) ]
model = nn.Sequential(
    nn.Conv2d(3, 6, 5), nn.ReLU(),   # 3 * 128 * 128 ->  6 * 124 * 124
    nn.MaxPool2d(2, 2),              #               ->  6 *  62 *  62
    nn.Conv2d(6, 12, 7), nn.ReLU(),  #               -> 12 *  56 *  56
    nn.MaxPool2d(2, 2),              #               -> 12 *  28 *  28
    nn.Conv2d(12, 6, 5), nn.ReLU(),  #               ->  6 *  24 *  24
    nn.MaxPool2d(2, 2),              #               ->  6 *  12 *  12
    nn.Flatten(1),
    nn.Linear(6 * 12 * 12, 120), nn.ReLU(),
    nn.Linear(120, 84), nn.ReLU(),
    nn.Linear(84, len(disp_labels))
)
train_and_eval(model, *datasets, disp_labels)
