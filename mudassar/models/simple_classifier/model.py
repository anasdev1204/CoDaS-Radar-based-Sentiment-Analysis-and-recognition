from typing import Literal

import torch
from transformers import BertConfig, BertModel, LlamaConfig, LlamaModel

from .utils import n_params, right_edge_padding


class BaseModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def n_params(self):
        return n_params(self)


class Identity(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, **kwargs):
        return kwargs.get("inputs_embeds", kwargs.get("input_ids", None))


class BertEncoder(BaseModule):
    def __init__(
        self,
        input_dim,
        embedding_size=64,
        num_layers=4,
        num_heads=8,
        intermediate_size=None,
        dropout=0.1,
        use_pos_emb=False,
        max_position_embeddings=7,
        pooling_strategy: Literal["mean", "cls"] = "mean",
    ):
        super().__init__()

        self.input_dim = input_dim
        self.input_proj = torch.nn.Identity() if input_dim == embedding_size else torch.nn.Linear(input_dim, embedding_size)

        config = BertConfig(
            hidden_size=embedding_size,
            num_hidden_layers=num_layers,
            num_attention_heads=min(num_heads, embedding_size // 4),
            intermediate_size=intermediate_size or embedding_size * 4,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
            max_position_embeddings=max_position_embeddings if use_pos_emb else 1, # unused
            vocab_size=1,  # unused
        )

        self.encoder = BertModel(config, add_pooling_layer=False)
        if not use_pos_emb:
            self.encoder.embeddings = Identity()

        self.pooling_strategy = pooling_strategy
        if self.pooling_strategy == "cls":
            self.cls_embedding = torch.nn.Parameter(torch.randn(1, 1, embedding_size))
            self.pooler = torch.nn.Linear(embedding_size, embedding_size)

        self.config = {
            "input_dim": input_dim,
            "embedding_size": embedding_size,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "intermediate_size": intermediate_size,
            "dropout": dropout,
            "use_pos_emb": use_pos_emb,
            "max_position_embeddings": max_position_embeddings,
            "pooling_strategy": pooling_strategy,
        }

    def forward(self, x, attention_mask=None):
        """
        x: (batch, seq_len, embedding_size)
        """
        x = self.input_proj(x.to(self.device))
        if self.pooling_strategy == "cls":
            B, S, E = x.shape
            cls_embedding = self.cls_embedding.expand(B, -1, -1)  # (B, 1, E)
            x = torch.cat([cls_embedding, x], dim=1)  # (B, S+1, E)
            if attention_mask is not None:
                attention_mask = torch.cat([torch.ones(B, 1, device=self.device), attention_mask], dim=1)  # (B, S+1)
        outputs = self.encoder(inputs_embeds=x, attention_mask=attention_mask)
        if self.pooling_strategy == "cls":
            pooled = self.pooler(outputs.last_hidden_state[:, 0, :])  # (B, E)
        else:
            hidden = outputs.last_hidden_state  # (B, S, E)
            if attention_mask is None:
                pooled = hidden.mean(dim=1)
            else:
                mask = attention_mask.unsqueeze(-1)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return pooled

    @property
    def output_dim(self):
        return self.encoder.config.hidden_size


class LlamaEncoder(BaseModule):
    def __init__(
        self,
        input_dim,
        embedding_size=512,
        num_layers=4,
        num_heads=8,
        intermediate_size=None,
        pooling_strategy: Literal["mean", "eos"] = "mean",
    ):
        super().__init__()

        self.input_dim = input_dim
        self.input_proj = torch.nn.Identity() if input_dim == embedding_size else torch.nn.Linear(input_dim, embedding_size)

        config = LlamaConfig(
            hidden_size=embedding_size,
            intermediate_size=intermediate_size or embedding_size * 4,
            num_hidden_layers=num_layers,
            num_attention_heads=min(num_heads, embedding_size // 4),
            num_key_value_heads=min(num_heads, embedding_size // 4),
            max_position_embeddings=4096,
            vocab_size=1,  # unused
        )

        self.encoder = LlamaModel(config)

        self.pooling_strategy = pooling_strategy
        if self.pooling_strategy == "eos":
            self.eos_embedding = torch.nn.Parameter(torch.randn(1, 1, embedding_size))
            self.pooler = torch.nn.Linear(embedding_size, embedding_size)

        self.config = {
            "input_dim": input_dim,
            "embedding_size": embedding_size,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "intermediate_size": intermediate_size,
            "pooling_strategy": pooling_strategy,
        }

    def forward(self, x, attention_mask=None):
        """
        x: (B, S, input_dim)
        attention_mask: (B, S)
        """
        x = self.input_proj(x.to(self.device))
        if self.pooling_strategy == "eos":
            B, S, E = x.shape
            eos_embedding = self.eos_embedding.expand(B, -1, -1)  # (B, 1, E)
            x = torch.cat([x, eos_embedding], dim=1)  # (B, S+1, E)
            if attention_mask is not None:
                attention_mask = torch.cat([attention_mask, torch.ones(B, 1, device=self.device)], dim=1)  # (B, S+1)

        outputs = self.encoder(inputs_embeds=x, attention_mask=attention_mask)
        if self.pooling_strategy == "eos":
            pooled = self.pooler(outputs.last_hidden_state[:, -1, :])  # (B, E)
        else:
            hidden = outputs.last_hidden_state
            if attention_mask is None:
                pooled = hidden.mean(dim=1)
            else:
                mask = attention_mask.unsqueeze(-1)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        return pooled

    @property
    def output_dim(self):
        return self.encoder.config.hidden_size


class RadarEncoder(BaseModule):
    def __init__(self, input_dim, output_dim=2, point_cloud_encoder_kwargs=None, temporal_encoder_kwargs=None):
        super().__init__()
        self.input_dim = input_dim
        self.point_cloud_encoder = BertEncoder(input_dim, **(point_cloud_encoder_kwargs or {}))
        self.temporal_encoder    = LlamaEncoder(self.point_cloud_encoder.output_dim, **(temporal_encoder_kwargs or {}))
        self.output_proj = torch.nn.Linear(self.temporal_encoder.output_dim, output_dim)
        self.config = {
            "input_dim": input_dim,
            "output_dim": output_dim,
            "point_cloud_encoder_kwargs": point_cloud_encoder_kwargs,
            "temporal_encoder_kwargs": temporal_encoder_kwargs,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, N, 4)
        """

        buckets = self.reshape_pce_input(x)  #   (B*T, N, 4)
        for k in buckets.keys():
            buckets[k]["tensors"] = self.point_cloud_encoder(torch.cat(buckets[k]["tensors"], dim=0))  # (B*T, 64)
        x_ = self.reshape_pce_output(buckets)  #  (B, T, 64)
        x, attn_mask = right_edge_padding(x_)
        # print("items" ,x.shape, "attns", attn_mask.shape if attn_mask is not None else None)
        pooled = self.temporal_encoder(x.contiguous(), attention_mask=attn_mask)  #   (B, 512)
        logits = self.output_proj(pooled)  #   (B, 2)

        return logits

    @property
    def output_dim(self):
        return self.output_proj.out_features

    def reshape_pce_input(self, x):
        if isinstance(x, torch.Tensor):
            x = [x]

        buckets = {}
        for i, t in enumerate(x):
            B, T, N, E = t.shape
            buckets.setdefault(N, {"tensors":[], "shapes": [], "indices": []})
            buckets[N]["tensors"].append(t.reshape(B * T, N, E))
            buckets[N]["shapes" ].append((B, T, N, E))
            buckets[N]["indices"].append(i)

        return buckets

    def reshape_pce_output(self, buckets):
        x = [None for _ in range(sum(len(bucket["shapes"]) for bucket in buckets.values()))]
        for bucket in buckets.values():
            split_sections = [B*T for B, T, N, E in bucket["shapes"]]
            split_tensors = torch.split(bucket["tensors"], split_sections, dim=0)
            for t, s, i in zip(split_tensors, bucket["shapes"], bucket["indices"]):
                B, T, N, E = s
                x[i] = t.reshape(B, T, self.point_cloud_encoder.output_dim)

        return x


class DownsampleBlock(BaseModule):
    def __init__(self, cin, cout, stride):
        super().__init__()

        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(cin, cout, 7, stride=stride, padding=3),
            torch.nn.BatchNorm1d(cout),
            torch.nn.GELU(),
            torch.nn.Conv1d(cout, cout, 3, padding=1),
            torch.nn.BatchNorm1d(cout),
        )

        self.skip = torch.nn.Conv1d(cin, cout, 1, stride=stride)

    def forward(self, x):
        return torch.nn.functional.gelu(self.conv(x) + self.skip(x))


class Downsampler(BaseModule):
    def __init__(self, input_dim, output_dim=64, hidden_dims=(32, 64, 64)):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim or input_dim

        if output_dim is None:
            self.blocks = torch.nn.Identity()
        else:
            blocks = []
            in_dim = input_dim
            for out_dim in list(hidden_dims)+[output_dim]:
                blocks.append(DownsampleBlock(in_dim, out_dim, stride=2))
                in_dim = out_dim
            self.blocks = torch.nn.Sequential(*blocks)

        self.config = {
            "input_dim": input_dim,
            "output_dim": output_dim,
            "hidden_dims": list(hidden_dims),
        }

    def forward(self, x):
        return self.blocks(x.to(self.device))

    def forward_attn_mask(self, attn_mask=None):
        if isinstance(self.blocks, torch.nn.Identity):
            return attn_mask

        if attn_mask is None:
            return None

        for _ in self.blocks:
            attn_mask = attn_mask[:, ::2]

        return attn_mask

    @property
    def n_blocks(self):
        return len(self.blocks) if isinstance(self.blocks, torch.nn.Sequential) else 0


class InfraredEncoder(BaseModule):
    def __init__(self, input_dim, output_dim=2, downsampler_kwargs=None, temporal_encoder_kwargs=None):
        super().__init__()
        self.input_dim = input_dim
        self.downsampler    = Downsampler(self.input_dim, **(downsampler_kwargs or {}))
        self.temporal_encoder = LlamaEncoder(self.downsampler.output_dim, **(     temporal_encoder_kwargs or {}))
        self.output_proj = torch.nn.Linear(self.temporal_encoder.output_dim, output_dim)
        self.config = {
            "input_dim": input_dim,
            "output_dim": output_dim,
            "downsampler_kwargs": downsampler_kwargs,
            "temporal_encoder_kwargs": temporal_encoder_kwargs,
        }


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, N * 7)
        """

        x, attn_mask = self.reshape_input(x)  # (B, N*7, T), (B, T)
        x = self.downsampler(x)  # (B, 64, t)
        attn_mask = self.downsampler.forward_attn_mask(attn_mask)  # (B, t)

        x = x.transpose(-2, -1)  # (B, t, 64)
        x = self.temporal_encoder(x, attn_mask)  # (B, 512)

        logits = self.output_proj(x)  # (B, 2)

        return logits

    @property
    def output_dim(self):
        return self.output_proj.out_features

    def reshape_input(self, x):
        if isinstance(x, torch.Tensor):
            x = [x]

        t, attn_mask = right_edge_padding(x)
        if t.dim() == 4:
            B, T, N, E = t.shape
            t = t.reshape(B, T, N * E)  # (B, T, N*7)
        t = t.transpose(-2, -1)  # (B, N*7, T)
        return t, attn_mask


class FancyInfraredEncoder(BaseModule):
    def __init__(self, input_dim, output_dim=2, downsampler_kwargs=None, point_cloud_encoder_kwargs=None, temporal_encoder_kwargs=None):
        super().__init__()
        self.input_dim = input_dim
        self.downsampler    = Downsampler(self.input_dim, **(downsampler_kwargs or {}))
        self.point_cloud_encoder = BertEncoder(self.downsampler.output_dim, **(point_cloud_encoder_kwargs or {}))
        self.temporal_encoder = LlamaEncoder(self.point_cloud_encoder.output_dim, **(     temporal_encoder_kwargs or {}))
        self.output_proj = torch.nn.Linear(self.temporal_encoder.output_dim, output_dim)
        self.config = {
            "input_dim": input_dim,
            "output_dim": output_dim,
            "downsampler_kwargs": downsampler_kwargs,
            "point_cloud_encoder_kwargs": point_cloud_encoder_kwargs,
            "temporal_encoder_kwargs": temporal_encoder_kwargs,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, N, E)
        """

        x, attn_mask = self.handle_jagged_input(x)  # (B, T, N, E), (B, T)

        x = x.transpose(-3, -2)  #         (B, N, T, E)
        x = x.transpose(-2, -1)  #         (B, N, E, T)
        B, N, E, T = x.shape
        x = x.reshape(B*N , E, T)  #       (B*N,  E, T)
        x = self.downsampler(x)  #         (B*N,  C, t)
        attn_mask = self.downsampler.forward_attn_mask(attn_mask)  # (B, t)

        x = x.view(B, N, *x.shape[-2:])  # (B, N, C, t)
        x = x.transpose(-2, -1)  #         (B, N, t, C)
        x = x.transpose(-3, -2)  #         (B, t, N, C)
        B, T, N, C = x.shape
        x = x.reshape(B * T, N, C)  #      (B*t, N, C)
        x = self.point_cloud_encoder(x)  # (B*t, C)

        x = x.reshape(B, T, self.point_cloud_encoder.output_dim)  # (B, t, C)
        x = self.temporal_encoder(x, attention_mask=attn_mask)  #    (B, e)

        logits = self.output_proj(x)  #    (B, output_dim)
        return logits

    @property
    def output_dim(self):
        return self.output_proj.out_features

    def handle_jagged_input(self, x):
        if isinstance(x, torch.Tensor):
            x = [x]
        t, attn_mask = right_edge_padding(x)
        return t, attn_mask

    # def handle_pce_input(self, x, attn_mask):
    #     """the padding frames are unnecessarily processed by the point cloud encoder.
    #     but we need to add padding again after PCE."""

def get_model(model_type, model_kwargs):
    if model_type == "radar":
        return RadarEncoder(**(model_kwargs or {}))
    elif model_type == "infrared":
        return InfraredEncoder(**(model_kwargs or {}))
    elif model_type == "fancy_infrared":
        return FancyInfraredEncoder(**(model_kwargs or {}))
    else:
        raise ValueError(f"Unknown model type: {model_type}")
