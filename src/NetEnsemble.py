#!/usr/bin/env python3

import torch
from torch import nn
from utils.TrainValidate import *
from utils.WeightEstimate import *
from utils.CustomSampler import *
from networks import *
import gc
import numpy as np
import matplotlib.pyplot as plt

#%%

gc.collect()
torch.cuda.empty_cache()


def window_chances_to_weight(outputs):
    windows = torch.max(outputs.data, 1)[1]
    max_conf = torch.max(outputs.data, 1).values
    positive_conf =torch.mul(windows, max_conf)
    return positive_conf


def ensemble_predict(models, target_weights, images, labels, output_transform):
    # Run our models, and return predictions for the most confident outcomes
    max_vals = [0]*len(images)
    predictions = [0]*len(images)
    for i, model in enumerate(models):
        outputs = model(images)
        outputs = outputs.cpu()
        outputs = output_transform(outputs)
        for j, val in enumerate(outputs.numpy()):
            if val > max_vals[j]:
                max_vals[j] = val
                predictions[j] = target_weights[i]
    return torch.FloatTensor(predictions)


def get_model_ensemble_results(models, target_weights, dataset, output_transform=lambda o:o,
    batch_size=4, criterion_for_loss=None):
    data_loader = torch.utils.data.DataLoader(dataset,
        batch_size=batch_size, shuffle=True, num_workers=0)
    # keep track of the data we guessed
    actual_values = torch.empty(0)
    model_values  = torch.empty(0)
    running_loss = 0
    # not training, no need to calculate the gradients
    with torch.no_grad():
        for images, labels in data_loader:

            outputs = ensemble_predict(models, target_weights, images, labels, output_transform)

            labels  = labels.cpu()
            actual_values = torch.cat( (actual_values, labels) )
            model_values  = torch.cat( (model_values, outputs) )
            if criterion_for_loss:
                loss = criterion_for_loss(outputs, labels)
                running_loss += loss.item()
    return actual_values, model_values, running_loss / len(data_loader)


def eval_ensemble(models, target_weights, dataset, model_output_to_weight):
    dataset.to_device()
    dataset.lbl_transform = lambda fi: fi.weight
    print("\nEVALUATING WEIGHT INFERENCE")
    actual_weights, guessed_weights, _ = get_model_ensemble_results(
        models, target_weights, dataset, model_output_to_weight)
    guessed_weights = torch.add(guessed_weights, window_size / 2)
    actual_weights = actual_weights.cpu()
    guessed_weights = guessed_weights.cpu()
    # Calculate some stats about the performance
    diffs_w  = torch.sub(actual_weights, guessed_weights).numpy()
    pc10, pc50, pc90 = np.percentile(np.abs(diffs_w), [ 10, 50, 90 ])
    # Print those, or graph them, idk
    print(f"d[ {round(pc10,1)} | {round(pc50,1)} | {round(pc90,1)} ]")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    plot_bins = [ (-95 + 10*i) for i in range(20) ]
    plt.hist(diffs_w, bins=plot_bins, edgecolor='black', linewidth=1,
        weights=np.ones(len(diffs_w)) / len(diffs_w))
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    if os.path.isdir("../docs/results/"):
        plt.savefig("../docs/results/last_run.png")
    plt.title("Grams off from actual weight")
    plt.show()
    # Return the differences to graph together maybe
    return diffs_w


#%%
# Get the datasets, and setup those windows
window_size  = 5
device       = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

target_weights = []
models = []


datasets = get_datasets(image_wh=224, device=device)
max_runs = int((WEIGHT_MAX-WEIGHT_MIN)/window_size)
disp_labels  = [ "0", "1" ]

for i in range(max_runs):
    fi_to_window = lambda fi: 1 if int((fi.weight - WEIGHT_MIN) / window_size) == i else 0
    target_weight = i * window_size + WEIGHT_MIN

    print(f'\n\nTraining Progress: {int(i/max_runs*100)}%')
    print(f"Now training target weight: {target_weight}")
    for dset in datasets:
        dset.lbl_transform = fi_to_window

    # Make sure to sample nicely, aka undersample the negative classes
    lbls = np.array(list(map(lambda fi: fi.lbl, datasets[0].fruit_images)))
    occurances = np.bincount(lbls) # occurances of a label
    if len(occurances) == 1:
        print(f"Skipping range {target_weight} because no instances found")
        continue
    weight_per_class = lbls.size / occurances
    weights = [ 0 ] * len(datasets[0].fruit_images)
    for idx, fi in enumerate(datasets[0].fruit_images):
        weights[idx] = weight_per_class[fi.lbl]
    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

    # Setup the model that we want to train
    model = get_network("ResNet", len(disp_labels), device)
    train_the_thing(model, f"weight_window_resnet_{window_size}_target_class_{target_weight}",
        *datasets, disp_labels, nn.CrossEntropyLoss(), epochs=10, sampler=sampler)

    target_weights.append(target_weight)
    models.append(model)


#%%

eval_ensemble(models, target_weights, datasets[1], window_chances_to_weight)
