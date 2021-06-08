
import torch, torchvision
from utils.DataSet import *
from utils.TrainValidate import *


# The main benchmarking function, takes a model, a set and a transform
# The model should take in image from the set as input, and create some output
def evaluate_weight_inference(model, dataset, model_output_to_weight):
    dataset.lbl_transform = FruitImage.property_transforms["weight"]()
    data_loader = torch.utils.data.DataLoader(dataset,
        batch_size=4, shuffle=True, num_workers=2)
    guess_and_actual = [ ]
    # not training, no need to calculate the gradients
    with torch.no_grad():
        for data in data_loader:
            # Run our network
            images, labels = data
            outputs = model(images)
            print("labels", labels)
            model_output_to_weight(outputs)


def train_and_eval(model, train_set, test_set, disp_labels=[]):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # cross_validate(model, criterion, optimizer, train_set)
    validate_results = [ ]
    for _ in range(1):
        # Train the thing for a bit
        train(model, criterion, optimizer, train_set, epochs=5)
        # Check if we should still continue
        validate_now = round(validate_classifier(model, test_set) * 100, 4)
        validate_results.append(validate_now)
        if len(validate_results) <= 5: continue
        lasts = validate_results[-4:-1]
        lasts_avg = sum(lasts) / len(lasts)
        print(f"{validate_now}% < {lasts_avg}% ??")
        if validate_now < lasts_avg:
            break
    print("\n")
    validate_classifier(model, test_set, show_for_classes=disp_labels)
