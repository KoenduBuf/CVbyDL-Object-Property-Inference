#!/usr/bin/env python3

import torch
from torch import nn
from utils.TrainValidate import *
from utils.WeightEstimate import *
from utils.DataSet import *
from networks import *


# Get average weights and datasets etc
device      = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train, test = get_datasets(lambda fi: fi.typei, 224, device)
class_dataf = lambda c: train.summary_of_typei(c)
class_avg_w = [ class_dataf(c)['avg_weight'] for c in train.types ]


# Setup the model that we want to train
model       = get_network("ResNet", len(train.types), device)
train_the_thing(model, "fruit_classifier_resnet",
    train, test, train.types, nn.CrossEntropyLoss())

# Access how good it is at guessing weights
def out_to_weights(outputs):
    predictions = torch.max(outputs.data, 1)[1]
    weights = map(lambda c: class_avg_w[c], predictions)
    weights = torch.tensor(list(weights))
    return weights

evaluate_weight_inference(model, test, out_to_weights)
