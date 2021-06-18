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


# Access how good it is at guessing weights
def window_chances_to_weight(outputs):
    windows = torch.max(outputs.data, 1)[1]
    max_conf = torch.max(outputs.data, 1).values
    positive_conf =torch.mul(windows, max_conf)
    #guesses = torch.mul(windows, target_class)
    return positive_conf

def ensemble_predict(models, target_classes, images, labels, output_transform):
    # Run our network
    max_vals = [0]*len(images)
    predictions = [0]*len(images)
    for i, model in enumerate(models):
        outputs = model(images)
        outputs = outputs.cpu()
        outputs = output_transform(outputs)
        for j, val in enumerate(outputs.numpy()):
            if val > max_vals[j]:
                max_vals[j] = val
                predictions[j] = target_classes[i]
    return torch.FloatTensor(predictions)


def get_model_ensemble_results(models, target_classes, dataset, output_transform=lambda o:o,
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

            outputs = ensemble_predict(models, target_classes, images, labels, output_transform)

            labels  = labels.cpu()
            actual_values = torch.cat( (actual_values, labels) )
            model_values  = torch.cat( (model_values, outputs) )
            if criterion_for_loss:
                loss = criterion_for_loss(outputs, labels)
                running_loss += loss.item()
    return actual_values, model_values, running_loss / len(data_loader)


def eval_ensemble(models, target_classes, dataset, model_output_to_weight):
    dataset.to_device()
    dataset.lbl_transform = lambda fi: fi.weight
    print("\nEVALUATING WEIGHT INFERENCE")
    actual_weights, guessed_weights, _ = get_model_ensemble_results(
        models, target_classes, dataset, model_output_to_weight)
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


def getLabelsFromFruitImageDS(ds):
    labels = []
    for fruit in ds.fruit_images:
        labels.append(fruit.lbl)
    return labels

#%%
# Get the datasets, and setup those windows
window_size  = 5
device       = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

target_classes = []
models = []

short_run = False

max_runs = int((WEIGHT_MAX-WEIGHT_MIN)/2)

if short_run:
    iterations = 8
    current_iteration = 0
for i in range(int((WEIGHT_MAX-WEIGHT_MIN)/2)):
    fi_to_window = lambda fi: 1 if int((fi.weight - WEIGHT_MIN) / window_size) == i else 0
    target_classes.append(i*window_size + WEIGHT_MIN)


    print(f"target_class:{target_classes[i]}")

    datasets     = get_datasets(fi_to_window, 224, device)
    disp_labels  = [ "0", "1" ]


    y = torch.from_numpy(np.array(getLabelsFromFruitImageDS(datasets[0])))
    sampler = StratifiedSampler(class_vector=y, batch_size=2)

    # Setup the model that we want to train
    model = get_network("ResNet", len(disp_labels), device)
    train_the_thing(model, f"weight_window_resnet_{window_size}_target_class_{target_classes[i]}",
        *datasets, disp_labels, nn.CrossEntropyLoss(), epochs=5, sampler=sampler)



    #diffs = evaluate_weight_inference(model, datasets[1], window_chances_to_weight)



    models.append(model)

    if short_run:
        current_iteration += 1
        if iterations == current_iteration:
            break

    print(f'Training Progress: {i/max_runs}')

#%%

eval_ensemble(models, target_classes, datasets[1], window_chances_to_weight)



