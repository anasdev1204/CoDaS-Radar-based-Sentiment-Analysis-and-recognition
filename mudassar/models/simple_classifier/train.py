from copy import deepcopy
from time import time
from typing import Literal

import torch

from .loss import get_criterion
from .infer import infer_model


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
    objective: Literal["classification", "contrastive", "cross_modal"] = "classification",
    train_sampler=None,
    valid_sampler=None,
):
    """
    Trains a PyTorch model with early stopping and checkpoints the best model
    based on a combined heuristic of validation loss and validation F1-score.
    """

    model = model.to(device)

    if objective == "classification":
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  collate_fn=collate_fn, pin_memory=pin_memory, num_workers=num_workers, batch_sampler=train_sampler)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, pin_memory=pin_memory, num_workers=num_workers, batch_sampler=valid_sampler)
    else:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_sampler, collate_fn=collate_fn, pin_memory=pin_memory, num_workers=num_workers)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_sampler=valid_sampler, collate_fn=collate_fn, pin_memory=pin_memory, num_workers=num_workers)

    n_outputs = getattr(model, "output_dim", None)
    train_criterion = get_criterion(objective=objective, class_weights=getattr(train_dataset, "class_weights", None), n_out=n_outputs, device=device)
    valid_criterion = get_criterion(objective=objective, class_weights=getattr(valid_dataset, "class_weights", None), n_out=n_outputs, device=device)
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

            for train_batch in train_loader:
                optimizer.zero_grad()
                loss, outputs = infer_model(objective, model, train_batch, train_criterion, device=device, non_blocking=pin_memory, frame_size=(train_sampler.frame_size if train_sampler is not None else None))
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * len((outputs[0] if isinstance(outputs, tuple) else outputs))

            epoch_train_loss = running_loss / (len(train_dataset) or 1)

            if verbose:
                print(f"Train Loss: {epoch_train_loss:.4f}", end=" ")

            # --- VALIDATION PHASE ---
            model.eval()
            running_val_loss = 0.0

            with torch.no_grad():
                for valid_batch in valid_loader:
                    loss, outputs = infer_model(objective, model, valid_batch, valid_criterion, device=device, non_blocking=pin_memory)

                    running_val_loss += loss.item() * len((outputs[0] if isinstance(outputs, tuple) else outputs))
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
                    print("--> Checkpoint saved!")
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
