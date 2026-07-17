from typing import Tuple

import torch

from .utils import to_device


def infer_model_for_classification(model: torch.nn.Module, inputs: torch.Tensor, targets: torch.Tensor, criterion: torch.nn.Module, device="cpu", non_blocking=False):
    inputs, targets = to_device(inputs, device, non_blocking=non_blocking), targets.to(device)

    outputs = model(inputs)
    if outputs.shape[1] == 1:
        targets = targets.to(outputs.dtype)
    loss = criterion(outputs.squeeze(-1), targets)
    return loss, outputs

def infer_model_for_contrastive(model: torch.nn.Module, batch, criterion: torch.nn.Module, device="cpu", non_blocking=False, frame_size=None):
    inputs, _ = batch
    inputs = to_device(inputs, device, non_blocking=non_blocking)

    og_outputs = model(inputs)

    if frame_size and frame_size < og_outputs.shape[0]:
        outputs = og_outputs.view(og_outputs.shape[0]//frame_size, frame_size, *og_outputs.shape[1:])
    else:
        outputs = og_outputs.unsqueeze(0)
    loss = criterion(outputs)
    return loss, og_outputs


def infer_model(objective: str, model: torch.nn.Module, batch, criterion: torch.nn.Module, device="cpu", non_blocking=False, frame_size=None):
    if objective == "classification":
        inputs, targets = batch
        return infer_model_for_classification(model, inputs, targets, criterion, device=device, non_blocking=non_blocking)
    if objective == "contrastive":
        return infer_model_for_contrastive(model, batch, criterion, device=device, non_blocking=non_blocking, frame_size=frame_size)

    raise ValueError(f"Unknown objective: {objective}")
