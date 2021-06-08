#!/usr/bin/env python3

import torch
from torch import nn
from utils.TrainValidate import *
from utils.WeightEstimate import *
from utils.DataSet import *


# Get the datasets, classes and average weights
datasets    = get_datasets("to_class")
classes     = datasets[0].types
class_dataf = lambda c: datasets[0].summary_of_typei(c)
class_avg_w = [ class_dataf(c)['avg_weight'] for c in classes ]

# Setup the model that we want to train
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
    nn.Linear(84, len(classes))
)

# Train the model, or get from cache
train_the_thing(model, "fruit_classifier",
    *datasets, classes, nn.CrossEntropyLoss())

# Access how good it is at guessing weights
def out_to_weights(outputs):
    predictions = torch.max(outputs.data, 1)[1]
    weights = map(lambda c: class_avg_w[c], predictions)
    weights = torch.tensor(list(weights))
    return weights

evaluate_weight_inference(model, datasets[1], out_to_weights)
