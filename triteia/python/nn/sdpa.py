import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional
from flash_attn_interface import flash_attn_func

def sdpa(
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor]=None, 
        dropout: float=0.0,
        is_causal: bool=True,
        scale: Optional[float]=None,
        impl: str="fa",
    ):
    # qkv should be of the same shape
    assert q.shape == k.shape == v.shape
    if attn_mask is not None:
        impl = "torch"
    # qkv are of shape (batch, seq_len, num_heads, head_dim)
    if impl == "torch":
        return F.scaled_dot_product_attention(
            torch.permute(q, [0,2,1,3]),
            torch.permute(k, [0,2,1,3]),
            torch.permute(v, [0,2,1,3]),
            attn_mask=attn_mask,
            dropout_p=dropout,
            is_causal=is_causal,
            scale=scale,
        )
    elif impl == "fa":
        return flash_attn_func(q,k,v, softmax_scale=scale, causal=is_causal)