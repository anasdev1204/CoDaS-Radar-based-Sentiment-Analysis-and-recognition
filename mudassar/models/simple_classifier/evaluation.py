import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from .utils import to_device


from collections import Counter

def evaluate_classification_model(
    model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, device="cpu", ys=None
):
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

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, all_predicted, average="weighted", zero_division=0
    )
    return precision, recall, f1


def evaluate_contrastive_model(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, device="cpu", ys=None):
    embeddings, all_labels = [], []
    with torch.no_grad():
        for inputs, targets in data_loader:
            outputs = model(to_device(inputs, device))
            embeddings.append(outputs.cpu())
            all_labels.append(targets.cpu())

    embeddings = torch.nn.functional.normalize(torch.cat(embeddings, dim=0), dim=-1)
    all_labels = torch.cat(all_labels, dim=0)

    # cross-fold
    unique_labels = sorted([label.item() for label in torch.unique(all_labels)])
    label_to_indexes = {label: torch.where(all_labels == label)[0] for label in unique_labels}
    cross_fold_predictions = []
    cross_fold_labels = []
    for i in range(min(len(label_to_indexes[label]) for label in unique_labels)):
        target_idx = [label_to_indexes[label][i].item() for label in unique_labels]
        mask = torch.zeros(len(all_labels), dtype=torch.bool)
        mask[target_idx] = True
        labels_rfrnc     = all_labels[ mask]
        embeddings_rfrnc = embeddings[ mask]
        embeddings_query = embeddings[~mask]
        sim = torch.matmul(embeddings_query, embeddings_rfrnc.T)
        predicted = labels_rfrnc[sim.argmax(dim=-1)].tolist()
        cross_fold_predictions.extend(predicted)
        lbls = all_labels[~mask].cpu().numpy().tolist()
        cross_fold_labels.extend(lbls)

    if isinstance(ys, dict):
        ys.setdefault("predicted", []).extend(cross_fold_predictions)
        ys.setdefault("targets", []).extend(cross_fold_labels)

    precision, recall, f1, _ = precision_recall_fscore_support(
        cross_fold_labels, cross_fold_predictions, average="weighted", zero_division=0
    )
    return precision, recall, f1
