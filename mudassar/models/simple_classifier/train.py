from copy import deepcopy
from time import time

import torch

from sklearn.metrics import precision_recall_fscore_support


def to_device(x, device, non_blocking=True):
    if isinstance(x, torch.Tensor):
        return x.to(device, non_blocking=non_blocking)
    elif isinstance(x, (list, tuple)):
        return [to_device(a, device, non_blocking=non_blocking) for a in x]
    elif isinstance(x, dict):
        return {k: to_device(v, device, non_blocking=non_blocking) for k, v in x.items()}
    else:
        return x

def evaluate_model(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, device="cpu", ys=None):
    all_predicted, all_targets = [], []
    with torch.no_grad():
        for inputs, targets in data_loader:
            outputs = model(to_device(inputs, device))
            predicted = outputs.max(1).indices if outputs.shape[1] > 1 else (outputs > 0.5).long().squeeze(1)
            all_predicted.extend(predicted.cpu().numpy().tolist())
            all_targets.extend(targets.cpu().numpy().tolist())

    if isinstance(ys, dict):
        ys.setdefault("predicted", []).extend(all_predicted)
        ys.setdefault("targets", []).extend(all_targets)

    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_predicted, average='weighted', zero_division=0)
    return precision, recall, f1

def fit(
    model: torch.nn.Module,
    train_dataset: torch.utils.data.Dataset,
    valid_dataset: torch.utils.data.Dataset,
    batch_size=16,
    epochs=100,
    patience=5,
    lr=0.0001,
    verbose=True,
    collate_fn=None,
    pin_memory=False,
    num_workers=0,
    device="cpu",
):
    """
    Trains a PyTorch model with early stopping and checkpoints the best model
    based on a combined heuristic of validation loss and validation F1-score.
    """

    model = model.to(device)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  collate_fn=collate_fn, pin_memory=pin_memory, num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, pin_memory=pin_memory, num_workers=num_workers)

    def get_criterion(class_weights, n_outputs, device):
        if n_outputs == 1:
            if class_weights is not None:
                class_weights = (class_weights[1] / class_weights[0]).to(device)
            return torch.nn.BCEWithLogitsLoss(weight=class_weights)
        else:
            return torch.nn.CrossEntropyLoss(weight=class_weights).to(device)

    n_outputs = getattr(model, "output_dim", None)
    train_criterion = get_criterion(getattr(train_dataset, "class_weights", None), n_outputs, device)
    valid_criterion = get_criterion(getattr(valid_dataset, "class_weights", None), n_outputs, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    history = {"train_loss": [], "valid_loss": [], "checkpoint_epoch": None}

    best_loss = float("inf")
    best_model_wts = deepcopy(model.state_dict())
    patience_counter = 0

    if verbose:
        print(f"Training on device: {device}")
        print("-" * 50)
    try:
        for epoch in range(epochs):
            if verbose:
                print(f"Epoch {epoch+1:02d}/{epochs:02d} ->", end=" ")
            start_time = time()
            # --- TRAINING PHASE ---
            model.train()
            running_loss = 0.0

            for inputs, targets in train_loader:
                inputs, targets = to_device(inputs, device, non_blocking=pin_memory), targets.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                if outputs.shape[1] == 1:
                    targets = targets.to(outputs.dtype)
                loss = train_criterion(outputs.squeeze(-1), targets)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * len(inputs)

            epoch_train_loss = running_loss / (len(train_dataset) or 1)

            if verbose:
                print(f"Train Loss: {epoch_train_loss:.4f}", end=" ")

            # --- VALIDATION PHASE ---
            model.eval()
            running_val_loss = 0.0

            with torch.no_grad():
                for inputs, targets in valid_loader:
                    inputs, targets = to_device(inputs, device, non_blocking=pin_memory), targets.to(device)
                    outputs = model(inputs)
                    if outputs.shape[1] == 1:
                        targets = targets.to(outputs.dtype)
                    loss = valid_criterion(outputs.squeeze(-1), targets)

                    running_val_loss += loss.item() * len(inputs)
            epoch_val_loss = running_val_loss / (len(valid_dataset) or 1)

            history["train_loss"].append(epoch_train_loss)
            history["valid_loss"].append(epoch_val_loss)

            end_time = time()
            if verbose:
                print(f"Valid Loss: {epoch_val_loss:.4f} | time/epoch: {end_time-start_time:.3f}sec", end=" ")

            # --- EARLY STOPPING & CHECKPOINTING ---
            if epoch_val_loss < best_loss:
                best_loss = epoch_val_loss
                best_model_wts = deepcopy(model.state_dict())
                patience_counter = 0
                history["checkpoint_epoch"] = epoch
                if verbose:
                    print(f"--> Checkpoint saved!")
            else:
                patience_counter += 1
                if verbose:
                    print()
                if patience_counter >= patience:
                    if verbose:
                        print(f"\nEarly stopping triggered at epoch {epoch+1}!")
                    break
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")

    model.load_state_dict(best_model_wts)
    return history
