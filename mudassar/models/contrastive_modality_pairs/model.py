from collections import namedtuple

import torch
import numpy as np

from . import layers as L

## for Radar, Infrared, IMU
class PointCloudEncoder(torch.nn.Module):
    def __init__(self, input_dim=4, initial_proj_dim=8, n_positional_encodings:int=None, dtype=torch.bfloat16, **kwargs):
        super().__init__()
        self.CLS = torch.nn.Parameter((torch.randn((1, 1, initial_proj_dim))))
        self.initial_proj = torch.nn.Linear(input_dim, initial_proj_dim)
        self.positional_encodings = None if n_positional_encodings is None else torch.nn.Parameter(torch.randn((1, n_positional_encodings, initial_proj_dim)))
        d = initial_proj_dim
        self.blocks = torch.nn.ModuleList([
            L.TransformerBlock(input_dim=d  , head_size=d//2, n_heads=4, queries_per_head=1, dtype=dtype, **kwargs),
            L.TransformerBlock(input_dim=d*2, head_size=d//2, n_heads=4, queries_per_head=2, dtype=dtype, **kwargs),
            L.TransformerBlock(input_dim=d*4, head_size=d   , n_heads=4, queries_per_head=2, dtype=dtype, pruning_n_preserved=1, **kwargs),
        ])
        # self.final_proj = torch.nn.Linear(self.blocks[-1].output_dim, out_dim, bias=False)
        self.type(dtype)

    def forward(self, x):
        # print("input shape:", x.shape)
        if x.ndim == 2:
            x = x.unsqueeze(0)
        x = self.initial_proj(x)
        # print("after initial_proj shape:", x.shape)
        if self.positional_encodings is not None:
            if x.shape[-2] > self.positional_encodings.shape[-2]:
                raise ValueError(f"Input has more points ({x.shape[-2]}) than positional encodings ({self.positional_encodings.shape[-2]}).")
            x = x + self.positional_encodings[..., :x.shape[-2], :]
            # print("after adding positional encodings shape:", x.shape)
        # print("CLS shape:", self.CLS.shape)
        # print("CLS .expand shape:", self.CLS.expand(*x.shape[:-2], -1, -1).shape)
        x = torch.cat([self.CLS.expand(*x.shape[:-2], -1, -1), x], dim=-2)
        # print("after CLS shape:", x.shape)
        for block in self.blocks:
            x, _ = block(x)
        # x = self.final_proj(x)
        return x

    @property
    def device(self):
        return self.CLS.device

    @property
    def dtype(self):
        return self.CLS.dtype

    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters())


TemporalDecoderOutput = namedtuple("TemporalDecoderOutput", ["embedding", "inputs", "last_hidden_states", "preserved_indexes"])

class TemporalDecoder(torch.nn.Module):
    def __init__(self, input_dim=64, out_dim=512, n_blocks=3, dropout_p=0.25, dtype=torch.bfloat16, **kwargs):
        super().__init__()
        # self.modality_tokens = torch.nn.Parameter(torch.randn((1, 5, input_dim)))
        self.EOS = torch.nn.Parameter((torch.randn((1, 1, input_dim))))

        d , blocks = input_dim, []
        for _ in range(n_blocks):
            blocks.append(L.TransformerBlock(input_dim=d, head_size=d//8, n_heads=8, queries_per_head=2, pruning_preserved_end=-1, pruning_frac_min_preserved=32, dropout_p=dropout_p, causal=True, rope=True, dtype=dtype, **kwargs))
            d = blocks[-1].output_dim

        self.blocks = torch.nn.ModuleList(blocks)
        self.embed_proj = torch.nn.Linear(self.blocks[-1].output_dim, out_dim)
        self.down_proj = torch.nn.Linear(self.blocks[-1].output_dim, input_dim)
        self.type(dtype)

    def forward(self, x, timestamps=None, return_last_hidden_states=False):
        # print("input shape:", x.shape)
        if x.ndim == 2:
            x = x.unsqueeze(0)
        inputs = x
            # print("unsqueezed shape:", x.shape)
        # print("EOS shape:", self.EOS.shape)
        # print("EOS .expand shape:", self.EOS.expand(x.shape[0], -1, -1).shape)
        x = torch.cat([x, self.EOS.expand(x.shape[0], -1, -1)], dim=-2)
        # print("after adding EOS shape:", x.shape)
        pos_ids = None
        if timestamps is not None:
            if isinstance(timestamps, np.ndarray):
                timestamps = torch.from_numpy(timestamps).to(x.device)
            pos_ids = self.time_to_pos_ids(timestamps)
            pos_ids = torch.cat([pos_ids, pos_ids[..., -1:]+1], dim=-1)  # maybe change to [pos_ids+1, pos_ids[:1]] if LM loss is used
            # print("position ids:", pos_ids)
        preserved_indexes = torch.arange(x.shape[-2], device=x.device)
        for block in self.blocks:
            x, mask = block(x, position_ids=pos_ids)
            preserved_indexes = preserved_indexes[mask]
            pos_ids = pos_ids[..., mask] if pos_ids is not None else None
            # print(pos_ids)
        embedding = self.embed_proj(x[..., -1, :])
        last_hidden_states = self.down_proj(x[..., :-1, :]) if return_last_hidden_states else None

        return TemporalDecoderOutput(
            embedding=embedding,
            inputs=inputs if return_last_hidden_states else None,
            last_hidden_states=last_hidden_states,
            preserved_indexes=preserved_indexes if return_last_hidden_states else None
        )

    def time_to_pos_ids(self, timestamps:torch.Tensor):
        return (timestamps/self.MAX_TIMESTAMP_SECONDS * (getattr(self.blocks[0].attn.rope, "max_seq_len", 2**12)-1)).long()

    MAX_TIMESTAMP_SECONDS = 40

    @property
    def device(self):
        return self.EOS.device

    @property
    def dtype(self):
        return self.EOS.dtype

    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters())

class ContrastiveModalityPairsModel(torch.nn.Module):
    def __init__(self, frame_kwargs={}, time_kwargs={}, dtype=torch.bfloat16):
        super().__init__()
        self.frame_model = PointCloudEncoder(**frame_kwargs)
        self.time_model = TemporalDecoder(**time_kwargs)
        self.type(dtype)

    def forward(self, x:torch.Tensor, timestamps=None, return_last_hidden_states=False):
        """
        Args:
            x (torch.Tensor): shape=(batch, time, points, features)
            timestamps (torch.Tensor, optional): shape=(batch, time). Defaults to None.
            return_last_hidden_states (bool, optional): _description_. Defaults to False.

        Returns:
            TemporalDecoderOutput: namedtuple.attributes=[embedding, inputs, last_hidden_states, preserved_indexes]
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if isinstance(x, torch.Tensor):
            # print(f"[uniform input {x.shape}]", end=" ")
            while x.ndim < 4:
                x = x.unsqueeze(0)
            # print(f"[unsqueezed input {x.shape}]", end=" ")
            x = self.frame_model(x.type(self.dtype).to(self.device))
            # print(f"[output {x.shape}]")

        elif isinstance(x, list):
            # print(f"jagged input with {len(x)} sequences", end=" | ")
            ## assuming only `num_points` dimension (dim=-2) is variable
            if (not isinstance(x[0], list)) and isinstance(x[0], (np.ndarray, torch.Tensor)) and x[0].ndim == 2:
                x = [x]  # batch dimension
                # print(f"added batch dimension", end=" | ")

            # print(f"frames in each sequence: {[len(seq) for seq in x]}", end=" | ")
            buckets = {}
            for b, seq in enumerate(x):
                for t, frame in enumerate(seq):
                    buckets.setdefault(len(frame), {"frames": [], "indexes": []})
                    buckets[len(frame)]["frames" ].append(frame)
                    buckets[len(frame)]["indexes"].append((b, t))
            x_new = [[None for _ in range(len(seq))] for seq in x]
            # print(f"created {len(buckets)} buckets based on frame lengths", end=" | ")
            # print(f"bucket sizes: {sorted([(n,len(bucket['frames'])) for n,bucket in buckets.items()], key=lambda x: x[-1])}", end=" | ")
            for bucket in buckets.values():
                bucket["frames"] = [(torch.from_numpy(f) if isinstance(f, np.ndarray) else torch.tensor(f)) for f in bucket["frames"]]
                bucket["frames"] = torch.stack(bucket["frames"]).type(self.dtype).to(self.device)
                bucket["frames"] = self.frame_model(bucket["frames"])
                for frame, (b, t) in zip(bucket["frames"], bucket["indexes"]):
                    x_new[b][t] = frame
            # print(x_new)
            x = torch.stack([torch.stack(seq) for seq in x_new]).contiguous()
            # print("after frame_model shape:", x.shape)

        x = self.time_model(x.squeeze(-2), timestamps=timestamps, return_last_hidden_states=return_last_hidden_states)
        return x

    @property
    def device(self):
        return self.frame_model.device

    @property
    def dtype(self):
        return self.frame_model.dtype

    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters())