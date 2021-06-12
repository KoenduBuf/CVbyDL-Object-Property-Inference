
import os
import numpy as np
import torch, torchvision
from utils.DataSet import *
from utils.TrainValidate import *


# The main benchmarking function, takes a model, a set and a transform
# The model should take in image from the set as input, and create some output
def evaluate_weight_inference(model, dataset, model_output_to_weight):
    dataset.to_device()
    dataset.lbl_transform = lambda fi: fi.weight
    print("\nEVALUATING WEIGHT INFERENCE")
    actual_weights, guessed_weights, _ = get_model_results(
        model, dataset, model_output_to_weight)
    dataset.to_device("cpu")
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


def train_the_thing(model, name, train_set, test_set,
    disp_labels=[], criterion=torch.nn.CrossEntropyLoss(),
    epochs=15):
    # First check if we already trained this model
    model_cache = f"./models/{name}.model"
    if os.path.isfile(model_cache):
        print("Using a cached, trained model")
        model.load_state_dict(torch.load(model_cache))
    else:
        train_set.to_device()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        if epochs == 0:
            cross_validate(model, criterion, optimizer, train_set, batch_size=4)
        else:
            train(model, criterion, optimizer, train_set, epochs, batch_size=4)
        train_set.to_device("cpu")
        if os.path.isdir(os.path.dirname(model_cache)):
            os.makedirs(os.path.dirname(model_cache), exist_ok=True)
            torch.save(model.state_dict(), model_cache)

    if isinstance(criterion, torch.nn.CrossEntropyLoss):
        test_set.to_device()
        validate_classifier(model, test_set, show_for_classes=disp_labels)
        test_set.to_device("cpu")
