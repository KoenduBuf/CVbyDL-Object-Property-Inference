
import torch, torchvision
from utils.DataSet import *
from utils.TrainValidate import *








def evaluate_classifier(model, on_set, show_per_class=False):
    validate


def train_and_eval(model, train_set, test_set, disp_labels=[]):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # cross_validate(model, criterion, optimizer, train_set)
    validate_results = [ ]
    while True:
        # Train the thing for a bit
        train(model, criterion, optimizer, train_set, epochs=5)
        # Check if we should still continue
        validate_now = round(validate(model, test_set) * 100, 4)
        validate_results.append(validate_now)
        if len(validate_results) <= 5: continue
        lasts = validate_results[-4:-1]
        lasts_avg = sum(lasts) / len(lasts)
        print(f"{validate_now}% < {lasts_avg}% ??")
        if validate_now < lasts_avg:
            break
    print("\n")
    validate(model, test_set, show_for_classes=disp_labels)
