#!/usr/bin/env python3

import torch
from torch import nn
from utils.TrainValidate import *
from utils.WeightEstimate import *
from networks import *

device       = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Get the datasets, and setup those windows
window_size  = 2
fi_to_window = lambda fi: int((fi.weight - WEIGHT_MIN) / window_size)
datasets     = get_datasets(fi_to_window, 224, device)
disp_labels  = [ f"around {i*window_size + WEIGHT_MIN + (window_size/2)}g"
    for i in range(int(WEIGHT_RANGE / window_size) + 1) ]
print(f"Window size {window_size}, thus {len(disp_labels)} windows")

# Setup the model that we want to train
model = get_network("ResNet", len(disp_labels), device)
train_the_thing(model, f"weight_window_resnet_{window_size}",
    *datasets, disp_labels, nn.CrossEntropyLoss())

# Access how good it is at guessing weights
def window_chances_to_weight(outputs):
    windows = torch.max(outputs.data, 1)[1]
    guesses = torch.mul(windows, window_size)
    guesses = torch.add(guesses, WEIGHT_MIN + window_size / 2)
    return guesses

evaluate_weight_inference(model, datasets[1], window_chances_to_weight)
