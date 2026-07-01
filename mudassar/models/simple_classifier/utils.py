from typing import Optional

import torch
import numpy as np


def n_params(model: torch.nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def right_edge_padding(data_list)-> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """shape: [(B, T1, ...), (B, T2, ...), ...]"""
    max_size = max(d.shape[1] for d in data_list)

    if all(d.shape[1] == max_size for d in data_list):
        return torch.cat(data_list, dim=0), None

    items  = []
    attn_masks = []
    for d in data_list:
        if isinstance(d, np.ndarray):
            d = torch.from_numpy(d)
        attn_masks.append(torch.cat([torch.ones(d.shape[1]), torch.zeros(max_size - d.shape[1])])[None, :].expand(d.shape[0], -1))

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
