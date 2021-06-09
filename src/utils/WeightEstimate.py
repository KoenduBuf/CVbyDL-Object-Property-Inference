
import numpy as np
import torch, torchvision
from utils.DataSet import *
from utils.TrainValidate import *


# The main benchmarking function, takes a model, a set and a transform
# The model should take in image from the set as input, and create some output
def evaluate_weight_inference(model, dataset, model_output_to_weight):
    dataset.lbl_transform = lambda fi: fi.weight
    print("\nEVALUATING WEIGHT INFERENCE")
    actual_weights, guessed_weights = get_model_results(
        model, dataset, model_output_to_weight)
    actual_weights, guessed_weights = actual_weights.cpu(), guessed_weights.cpu()
    # Calculate some stats about the performance
    diffs_w  = torch.sub(actual_weights, guessed_weights).numpy()
    avg_off  = np.sum(np.abs(diffs_w)) / len(diffs_w)
    pc10, pc50, pc90 = np.percentile(np.abs(diffs_w), [ 10, 50, 90 ])
    if avg_off == pc50:
        print("I knew it pc50 == avg")
    # Print those, or graph them, idk
    print(f"d[ {round(pc10,1)} | {round(avg_off,1)} | {round(pc90,1)} ]")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    plot_bins = [ (-95 + 10*i) for i in range(20) ]
    plt.hist(diffs_w, bins=plot_bins, edgecolor='black', linewidth=1,
        weights=np.ones(len(diffs_w)) / len(diffs_w))
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.title("Grams off from actual weight")
    plt.show()
    # Return the differences to graph together maybe
    return diffs_w


def train_the_thing(model, name, train_set, test_set,
    disp_labels=[], criterion=torch.nn.CrossEntropyLoss(),
    epochs=20):
    # First check if we already trained this model
    model_cache = f"./models/{name}_e{epochs}.model"
    if os.path.isfile(model_cache):
        print("Using a cached, trained model")
        model.load_state_dict(torch.load(model_cache))
        model.eval()
        return

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    train(model, criterion, optimizer, train_set, epochs = epochs)
    torch.save(model.state_dict(), model_cache)

    # validate_results = [ ]
    # for _ in range(5):
        # Train the thing for a bit
        # train(model, criterion, optimizer, train_set)
        # Check if we should still continue
        # validate_now = round(validate_classifier(model, test_set) * 100, 4)
        # validate_results.append(validate_now)
        # if len(validate_results) <= 5: continue
        # lasts = validate_results[-4:-1]
        # lasts_avg = sum(lasts) / len(lasts)
        # print(f"{validate_now}% < {lasts_avg}% ??")
        # if validate_now < lasts_avg:
        #     break

    if isinstance(criterion, torch.nn.CrossEntropyLoss):
        validate_classifier(model, test_set,
            show_for_classes=disp_labels)
