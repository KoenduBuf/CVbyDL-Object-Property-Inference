#!/usr/bin/env python3

import torch
from torch import nn
from utils.TrainValidate import *
from utils.WeightEstimate import *
from utils.DataSet import *
from networks import *


# Get the datasets, classes and average weights
datasets    = get_datasets(lambda fi: fi.typei, 224)
classes     = datasets[0].types
class_dataf = lambda c: datasets[0].summary_of_typei(c)
class_avg_w = [ class_dataf(c)['avg_weight'] for c in classes ]

# Setup the model that we want to train
model = get_network("ResNet", len(classes))

# Train the model, or get from cache
train_the_thing(model, "fruit_classifier_resnet",
    *datasets, classes, nn.CrossEntropyLoss())

# Access how good it is at guessing weights
def out_to_weights(outputs):
    predictions = torch.max(outputs.data, 1)[1]
    weights = map(lambda c: class_avg_w[c], predictions)
    weights = torch.tensor(list(weights))
    return weights

evaluate_weight_inference(model, datasets[1], out_to_weights)
