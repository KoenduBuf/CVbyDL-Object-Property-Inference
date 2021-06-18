#!/usr/bin/env python3

import torch
from torch import nn
from utils.TrainValidate import *
from utils.WeightEstimate import *
from networks import *
import gc
import numpy as np
import matplotlib.pyplot as plt

gc.collect()
torch.cuda.empty_cache()


# Access how good it is at guessing weights
def un_normalize_weights(outputs):
    guesses = torch.mul(outputs.data, WEIGHT_RANGE)
    guesses = torch.add(guesses, WEIGHT_MIN)
    return guesses

# Get the datasets
device   = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Setup the windows
window_size  = 2


resolutions = [32, 64, 128, 192, 224]
diff_avg = []
diff_std = []
repeats = 5

for res in resolutions:
    diff_avg_temp = []
    for rep in range(repeats):
        datasets = get_datasets(lambda fi: (fi.weight - WEIGHT_MIN) / WEIGHT_RANGE, res, device)


        model = get_network("ResNet", 1, device, flatten=True)
        train_the_thing(model, "weight_regression", *datasets, criterion=nn.L1Loss(), epochs=30)

        diffs = evaluate_weight_inference(model, datasets[1], un_normalize_weights)

        diff_avg_temp.append(np.average(np.abs(diffs)))
    diff_avg.append(np.average(diff_avg_temp))
    diff_std.append(np.std(diff_avg_temp))

plt.figure()
#for i,res in enumerate(resolutions):
plt.errorbar(x=resolutions, y=diff_avg, yerr = diff_std)
#plt.legend()
plt.xlabel('image resolution')
plt.ylabel('average weight prediction error in grams')
plt.title(f"{repeats}-run Average prediction error for models trained for 30 epochs on different image resolutions")
