
import torch

def train_epoch(model, criterion, optimizer, data_loader):
    running_loss = 0.0
    correct = total = 0
    for batch, data in enumerate(data_loader, 0):
        # data is a list of [inputs, labels]
        inputs, labels = data
        # forward + backward + optimize
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # get some statistics
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        running_loss += loss.item()
    return (running_loss / len(data_loader)), (correct / total)


def train(model, criterion, optimizer, dataset, epochs = 10, batch_size=4):
    train_loader = torch.utils.data.DataLoader(dataset,
        batch_size=batch_size, shuffle=True, num_workers=2)
    total_acc = 0
    for epoch in range(epochs):
        avgloss, acc = train_epoch(model, criterion, optimizer, train_loader)
        print(f"  [ epoch {str(epoch + 1):2s} ] acc: {acc:.1f} loss: {avgloss:.3f}")
        total_acc += acc
    return total_acc / epochs


def validate_classifier(model, dataset, batch_size=4, show_for_classes=[]):
    data_loader = torch.utils.data.DataLoader(dataset,
        batch_size=batch_size, shuffle=True, num_workers=2)
    correct = total = 0
    correct_pred = [ 0 ] * len(show_for_classes)
    total_pred = [ 0 ] * len(show_for_classes)
    # not training, no need to calculate the gradients
    with torch.no_grad():
        for data in data_loader:
            # Run our network
            images, labels = data
            outputs = model(images)
            _, predictions = torch.max(outputs.data, 1)
            # Get the statistics
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if len(show_for_classes) <= label: continue
                if label == prediction:
                    correct_pred[label] += 1
                total_pred[label] += 1
    acc = correct / total
    print(f"  VALIDATION: {acc * 100:.1f}% correctly classified")
    if len(show_for_classes) == 0: return acc
    accs = map(lambda tpl: round(tpl[0] / tpl[1] * 100, 2)\
        if tpl[1] != 0 else "-", zip(correct_pred, total_pred))
    class_justfify = max(map(len, show_for_classes)) + 2
    for clas, cacc in zip(show_for_classes, accs):
        print("Accuracy on class " + clas.rjust(class_justfify)
        + " is " + str(cacc).rjust(6))
    return acc

################################################################################
################################################### Stuff to do cross validation

def split_dataset(dataset, k_fold, fold):
    total_size  = len(dataset)
    seg_size    = int(total_size / k_fold)
    # We basically split in 3: [left train set][validation set][right train set]
    split1 = fold * seg_size
    split2 = fold * seg_size + seg_size
    # Get the indexes of the training, and validation sets:
    train_left_indices  = list(range( 0,      split1     ))
    train_right_indices = list(range( split2, total_size ))
    train_indices  = train_left_indices + train_right_indices
    val_indices    = list(range( split1, split2 ))
    # Create the training and test sets from the main set:
    train_set = torch.utils.data.dataset.Subset(dataset, train_indices)
    val_set   = torch.utils.data.dataset.Subset(dataset, val_indices)
    return train_set, val_set


# define a cross validation function https://stackoverflow.com/questions/60883696
def cross_validate(model, criterion, optimizer, dataset, k_fold=5, batch_size=5):
    train_score = [ -1 ] * k_fold
    val_score   = [ -1 ] * k_fold
    total_size  = len(dataset)
    seg_size    = int(total_size / k_fold)

    for fold in range(k_fold):
        # Create the training and test sets from the main set:
        train_set, val_set = split_dataset(dataset, k_fold, fold)
        print(f"Fold {fold}, train {len(train_set)} - test {len(val_set)}")
        # Train the model a bit, then validate to get our performance
        train_acc = train(model, criterion, optimizer, train_set, epochs=6)
        train_score[fold] = train_acc
        val_acc = validate_classifier(model, val_set)
        val_score[fold] = val_acc

    return train_score, val_score
