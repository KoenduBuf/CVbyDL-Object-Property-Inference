
import torch

################################################################################
##################################################### Stuff to do basic training

def train_epoch(model, criterion, optimizer, data_loader):
    running_loss = 0.0
    for inputs, labels in data_loader:
        # forward + backward + optimize
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(data_loader)

def train(model, criterion, optimizer, dataset, epochs=2, batch_size=4):
    train_loader = torch.utils.data.DataLoader(dataset,
        batch_size=batch_size, shuffle=True, num_workers=0)
    if epochs == 1:
        return train_epoch(model, criterion, optimizer, train_loader)
    avgloss = 0
    for epoch in range(epochs):
        avgloss += train_epoch(model, criterion, optimizer, train_loader)
        print(f"  [ epoch {str(epoch + 1):>2s} ] loss: {avgloss:.3f}")
    return avgloss / epochs

def train_until(model, criterion, optimizer, train_set,
    test_set, batch_size=4, patience=3):
    best_ever_loss = 1000; best_ever_model_state = None
    prev_losses = [ best_ever_loss ] * patience; at = 0
    for epoch in range(0, 1000):
        # Train for 1 epoch and validate
        train(model, criterion, optimizer, train_set,
            epochs=1, batch_size=batch_size)
        _, _, val_loss = get_model_results(model,
            test_set, criterion_for_loss=criterion)
        print(f"  Epoch {str(epoch):>2s}, val_loss: {str(round(val_loss,3)):<6s}")
        # Then keep track of our best performance
        if val_loss < best_ever_loss:
            best_ever_model_state = model.state_dict()
            best_ever_loss = val_loss
        prev_losses[epoch % patience] = val_loss
        # If it has been a while since our best, return
        if not best_ever_loss in prev_losses:
            print(f"  Trained! Best was {best_ever_loss}")
            model.load_state_dict(best_ever_model_state)
            return best_ever_loss

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

def cross_validate(model, criterion, optimizer, dataset, k_fold=5, batch_size=4):
    best_ever_loss = 1000; best_ever_model_state = None
    initial_model_state = model.state_dict()
    losses = [ -1 ] * k_fold

    for fold in range(k_fold):
        model.load_state_dict(initial_model_state)
        # Create the training and test sets from the main set:
        train_set, val_set = split_dataset(dataset, k_fold, fold)
        print(f"Fold {fold}, train {len(train_set)} - test {len(val_set)}")
        # Train the model a bit, then validate to get our performance
        losses[fold] = train_until(model, criterion, optimizer,
            train_set, val_set, batch_size)
        if losses[fold] < best_ever_loss:
            best_ever_model_state = model.state_dict()
            best_ever_loss = losses[fold]

    model.load_state_dict(best_ever_model_state)
    return losses

################################################################################
############################################### Stuff to easily get some results

def get_model_results(model, dataset, output_transform=lambda o:o,
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
            # Run our network
            outputs = output_transform(model(images))
            outputs, labels = outputs.cpu(), labels.cpu()
            actual_values = torch.cat( (actual_values, labels) )
            model_values  = torch.cat( (model_values, outputs) )
            if criterion_for_loss:
                loss = criterion_for_loss(outputs, labels)
                running_loss += loss.item()
    return actual_values, model_values, running_loss / len(data_loader)


def validate_classifier(model, dataset, batch_size=4, show_for_classes=[]):
    # Create the data loader and run the model to get some results
    actual, modelv, _ = get_model_results(model, dataset,
        lambda o: torch.max(o.data, 1)[1], batch_size)
    # Get the amount of correct
    accuracy = (actual == modelv).sum().item() / len(actual)
    # Get the amount of correct per label and such
    correct_pred = [ 0 ] * len(show_for_classes)
    total_pred   = [ 0 ] * len(show_for_classes)
    for real, pred in zip(actual.numpy(), modelv.numpy()):
        real, pred = int(real), int(pred)
        if len(show_for_classes) <= real: continue
        if real == pred: correct_pred[real] += 1
        total_pred[real] += 1
    # finally print out whatever needs printing
    print(f"  VALIDATION: {accuracy * 100:.1f}% correctly classified")
    if len(show_for_classes) == 0: return accuracy
    accs = map(lambda tpl: round(tpl[0] / tpl[1] * 100, 2)\
        if tpl[1] != 0 else "-", zip(correct_pred, total_pred))
    class_justfify = max(map(len, show_for_classes)) + 2
    for clas, cacc in zip(show_for_classes, accs):
        print("Accuracy on class " + clas.rjust(class_justfify)
        + " is " + str(cacc).rjust(6))
    return accuracy
