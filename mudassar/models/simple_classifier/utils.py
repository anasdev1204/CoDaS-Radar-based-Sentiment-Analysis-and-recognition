from typing import Optional

import torch
import numpy as np


def n_params(model: torch.nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def right_edge_padding(data_list, existing_attn_masks: Optional[list[Optional[torch.Tensor]]]=None)-> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """shape: [(B, T1, ...), (B, T2, ...), ...]"""
    if len(data_list) == 1:
        return data_list[0], (existing_attn_masks or [None])[0]

    max_size = max(d.shape[1] for d in data_list)

    if all(d.shape[1] == max_size for d in data_list):
        x = torch.cat([(torch.from_numpy(d) if isinstance(d, np.ndarray) else d) for d in data_list], dim=0)
        a = torch.cat([(m if m is not None else torch.ones(*d.shape[:2], device=d.device)) for m, d in zip(existing_attn_masks, data_list)], dim=0) if existing_attn_masks is not None else None
        return x, a

    items  = []
    attn_masks = []
    for i, d in enumerate(data_list):
        if isinstance(d, np.ndarray):
            d = torch.from_numpy(d)
        attn_masks.append(torch.cat([
            (torch.ones(d.shape[0], d.shape[1]) if existing_attn_masks is None or existing_attn_masks[i] is None else existing_attn_masks[i]).to(d.device),
            torch.zeros(d.shape[0], max_size - d.shape[1]).to(d.device)
        ], dim=1))

        if d.shape[1] < max_size:
            repeated = d[:, -1:, ...].expand(*(-1, max_size - d.shape[1], *([-1] * (d.ndim - 2))))
            # print(repeated.shape, d.shape)
            d = torch.cat([d, repeated], dim=1)
        items.append(d)
    # print("attn_masks", [tuple(m.shape) for m in attn_masks])
    items = torch.cat(items, dim=0)
    attn_masks = torch.cat(attn_masks, dim=0).to(items.device)
    # print("items", items.shape, "attns", attn_masks.shape)
    return items, attn_masks
