from typing import Optional

import torch
import numpy as np


class AttentionBlock(torch.nn.Module):
    def __init__(self, input_dim=8, head_size=4, n_heads=2, queries_per_head=1, dropout_p=0.1, causal=False, rope=False, dtype=torch.bfloat16):
        super().__init__()
        self.input_dim = input_dim
        self.head_size = head_size
        self.n_heads = n_heads
        self.queries_per_head = queries_per_head
        self.dropout_p = dropout_p
        self.causal = causal
        self.rope = RotaryPositionalEmbeddings(head_size//2) if rope else None

        self.layer_norm = torch.nn.LayerNorm(input_dim)
        self.qkv_proj = torch.nn.Linear(input_dim, head_size * (n_heads * (queries_per_head + 2)), bias=False)
        self.out_proj = torch.nn.Linear(head_size * n_heads * queries_per_head, head_size * n_heads * queries_per_head, bias=False)

        self.type(dtype)

    def forward(self, x, position_ids=None):
        # print("======= attention block forward =======")
        # print("input shape:", x.shape)
        if x.ndim == 2:
            x = x.unsqueeze(0)
        # print("unsqueezed shape:", x.shape)
        x = self.layer_norm(x)
        x = self.attend(x, position_ids=position_ids)
        x = self.out_proj(x)
        # print("output shape:", x.shape)
        # print("======= END attention block forward =======")
        return x

    def attend(self, x: torch.Tensor, position_ids=None):
        # print("pre attention shape:", x.shape)
        rest, S, E = x.shape[:-2], x.shape[-2], x.shape[-1]
        qkv = self.qkv_proj(x)
        # print("qkv shape:", qkv.shape)
        q, k, v = qkv.split([self.n_heads * self.queries_per_head * self.head_size, self.n_heads * self.head_size, self.n_heads * self.head_size], dim=-1)
        # print("q shape:", q.shape, "k shape:", k.shape, "v shape:", v.shape)
        # required shape: (N, ..., H, S, E)
        permute_order = list(range(len(rest))) + [-2,-3,-1]
        q = q.view(*rest, S, self.n_heads * self.queries_per_head, self.head_size)
        k = k.view(*rest, S, self.n_heads, self.head_size)
        v = v.view(*rest, S, self.n_heads, self.head_size)
        # print("q shape:", q.shape, "k shape:", k.shape, "v shape:", v.shape)
        # if isinstance(cache, dict) and not self.training and "kv_cache" in cache:
        if self.rope is not None:
            n_nope, n_pe = self.head_size - self.head_size//2, self.head_size//2
            q_nope, q_pe = q.split([n_nope, n_pe], dim=-1)
            k_nope, k_pe = k.split([n_nope, n_pe], dim=-1)
            # print("q_nope shape:", q_nope.shape, "q_pe shape:", q_pe.shape)
            # print("k_nope shape:", k_nope.shape, "k_pe shape:", k_pe.shape)
            q_pe = self.rope(q_pe, input_pos=position_ids)
            k_pe = self.rope(k_pe, input_pos=position_ids)
            # print("q_pe shape:", q_pe.shape)
            # print("k_pe shape:", k_pe.shape)
            q = torch.cat([q_nope, q_pe], dim=-1).contiguous()
            k = torch.cat([k_nope, k_pe], dim=-1).contiguous()
            # print("after rope q shape:", q.shape, "after rope k shape:", k.shape)

        q = q.permute(*permute_order).contiguous()
        k = k.permute(*permute_order).contiguous()
        v = v.permute(*permute_order).contiguous()
        # print("q shape:", q.shape, "k shape:", k.shape, "v shape:", v.shape)

        x = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout_p, is_causal=self.causal, enable_gqa=True)
        # print("attn output shape:", x.shape)
        x = x.permute(*permute_order).contiguous()
        # print("attn output permuted shape:", x.shape)
        x = x.view(*rest, S, self.n_heads * self.queries_per_head * self.head_size).contiguous()
        # print("attn output reshaped shape:", x.shape)

        return x

class TransformerBlock(torch.nn.Module):
    def __init__(self, input_dim=8, head_size=4, n_heads=2, queries_per_head=2, mlp_ratio=4, dropout_p=0.125, causal=False, rope=False, pruning_preserved_end=0, pruning_n_preserved=None, pruning_preserve_frac=0.5, pruning_frac_min_preserved=8, dtype=torch.bfloat16):
        super().__init__()
        self.input_dim = input_dim
        self.attn = AttentionBlock(input_dim, head_size, n_heads, queries_per_head, dropout_p, causal, rope, dtype)
        self.mlp = MLPBlock(head_size * n_heads * queries_per_head, mlp_ratio, dropout_p, dtype)
        self.pruning_preserved_end = pruning_preserved_end
        self.pruning_n_preserved = pruning_n_preserved
        self.pruning_preserve_frac = pruning_preserve_frac
        self.pruning_frac_min_preserved = pruning_frac_min_preserved
        self.output_dim = self.attn.out_proj.out_features
        self.type(dtype)

    def forward(self, x, position_ids=None):
        x = jagged_residual(x, self.attn(x, position_ids=position_ids))

        # prune the sequence length
        n_preserved = self.pruning_n_preserved if self.pruning_n_preserved is not None else max(self.pruning_frac_min_preserved, int(x.shape[-2] * self.pruning_preserve_frac))
        mask = make_pruning_mask(n_preserved, x.shape[-2], preserved_end=self.pruning_preserved_end, is_training=self.training)
        # print(mask.numpy().astype(int))
        # print(mask.shape, x.shape)
        x = x[..., mask, :]

        x = x + self.mlp(x)
        return x, mask

def jagged_residual(x: torch.Tensor, y: torch.Tensor):
    if x.shape == y.shape:
        return x + y
    shorter = min([x, y], key=lambda t: t.shape[-1])
    longer = x if shorter is y else y
    return torch.cat([shorter + longer[..., :shorter.shape[-1]], longer[..., shorter.shape[-1]:]], dim=-1).contiguous()

def evenly_spaced_mask(n_true, total_length):
    return np.diff(np.linspace(0, n_true, total_length+1).astype(int)).astype(bool)

def make_pruning_mask(n_preserved: int, total:int, preserved_end=-1, is_training=False):
    if n_preserved >= total:
        return torch.ones(total, dtype=torch.bool)
    mask = evenly_spaced_mask(n_preserved - int(preserved_end in (0,-1)), total - int(preserved_end in (0,-1)))
    if is_training:
        np.random.shuffle(mask)

    if preserved_end == 0:
        mask = np.concatenate([[True], mask])
    elif preserved_end == -1:
        mask = np.concatenate([mask, [True]])

    mask = torch.from_numpy(mask).type(torch.bool)
    return mask


class MLPBlock(torch.nn.Module):
    def __init__(self, input_dim=8, mlp_ratio=4, dropout_p=0.1, dtype=torch.bfloat16):
        super().__init__()
        self.input_dim = input_dim
        self.mlp_ratio = mlp_ratio

        self.layer_norm = torch.nn.LayerNorm(input_dim)
        self.fc1 = torch.nn.Linear(input_dim, input_dim * mlp_ratio)
        self.silu = torch.nn.SiLU()
        self.dropout = torch.nn.Dropout(dropout_p)
        self.fc2 = torch.nn.Linear(input_dim * mlp_ratio, input_dim, bias=False)

        self.type(dtype)

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.silu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class RotaryPositionalEmbeddings(torch.nn.Module):
    """
    Copied from https://meta-pytorch.org/torchtune/0.5/_modules/torchtune/modules/position_embeddings.html#RotaryPositionalEmbeddings
    Args:
        dim (int): Embedding dimension. This is usually set to the dim of each
            head in the attention module computed as ``embed_dim // num_heads``
        max_seq_len (int): Maximum expected sequence length for the
            model, if exceeded the cached freqs will be recomputed
        base (int): The base for the geometric progression used to compute
            the rotation angles
    """

    def __init__(self, dim: int, max_seq_len: int = 2**12, base: int = 10_000, dtype=torch.bfloat16) -> None:
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        self.rope_init()
        self.type(dtype)

    def rope_init(self):
        theta = 1.0 / (self.base ** (torch.arange(0, self.dim, 2)[: (self.dim // 2)].float() / self.dim))
        self.register_buffer("theta", theta, persistent=False)
        self.build_rope_cache(self.max_seq_len)

    def build_rope_cache(self, max_seq_len: int = 4096) -> None:
        # Create position indexes `[0, 1, ..., max_seq_len - 1]`
        seq_idx = torch.arange(max_seq_len, dtype=self.theta.dtype, device=self.theta.device)

        # Outer product of theta and position index; output tensor has
        # a shape of [max_seq_len, dim // 2]
        idx_theta = torch.einsum("i, j -> ij", seq_idx, self.theta).float()

        # cache includes both the cos and sin components and so the output shape is
        # [max_seq_len, dim // 2, 2]
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
        self.register_buffer("cache", cache, persistent=False)

    def forward(
        self, x: torch.Tensor, *, input_pos: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor with shape
                ``[b, s, n_h, h_d]``
            input_pos (Optional[torch.Tensor]): Optional tensor which contains the position ids
                of each token. During training, this is used to indicate the positions
                of each token relative to its sample when packed, shape [b, s].
                During inference, this indicates the position of the current token.
                If none, assume the index of the token is its position id. Default is None.

        Returns:
            torch.Tensor: output tensor with shape ``[b, s, n_h, h_d]``

        Notation used for tensor shapes:
            - b: batch size
            - s: sequence length
            - n_h: num heads
            - h_d: head dim
        """
        # input tensor has shape [b, s, n_h, h_d]
        seq_len = x.size(-3)

        # extract the values based on whether input_pos is set or not
        rope_cache = (self.cache[:seq_len] if input_pos is None else self.cache[input_pos])

        # reshape input; the last dimension is used for computing the output.
        # Cast to float to match the reference implementation
        # tensor has shape [b, s, n_h, h_d // 2, 2]
        xshaped = x.reshape(*x.shape[:-1], -1, 2)

        # reshape the cache for broadcasting
        # tensor has shape [b, s, 1, h_d // 2, 2] if packed samples,
        # otherwise has shape [1, s, 1, h_d // 2, 2]
        rope_cache = rope_cache.view(-1, xshaped.size(-4), 1, xshaped.size(-2), 2)

        # tensor has shape [b, s, n_h, h_d // 2, 2]
        x_out = torch.stack(
            [
                xshaped[..., 0] * rope_cache[..., 0]
                - xshaped[..., 1] * rope_cache[..., 1],
                xshaped[..., 1] * rope_cache[..., 0]
                + xshaped[..., 0] * rope_cache[..., 1],
            ],
            -1,
        )

        # tensor has shape [b, s, n_h, h_d]
        x_out = x_out.flatten(-2)
        return x_out.type_as(x)
