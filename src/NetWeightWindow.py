#!/usr/bin/env python3

import torch
from torch import nn
from utils.TrainValidate import *
from utils.WeightEstimate import *

# Get the datasets, and setup those windows
window_size  = 1
fi_to_window = lambda fi: int((fi.weight - WEIGHT_MIN) / window_size)
datasets     = get_datasets(fi_to_window)
disp_labels  = [ f"around {i*window_size + WEIGHT_MIN + (window_size/2)}g"
    for i in range(int(WEIGHT_RANGE / window_size) + 1) ]
print(f"Window size {window_size}, thus {len(disp_labels)} windows")

# Setup the model that we want to train
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

# Train the model, or get from cache
train_the_thing(model, f"weight_window_{len(disp_labels)}",
    *datasets, disp_labels, nn.CrossEntropyLoss())

# Access how good it is at guessing weights
def window_chances_to_weight(outputs):
    windows = torch.max(outputs.data, 1)[1]
    guesses = torch.mul(windows, window_size)
    guesses = torch.add(guesses, WEIGHT_MIN + window_size / 2)
    return guesses

evaluate_weight_inference(model, datasets[1], window_chances_to_weight)
