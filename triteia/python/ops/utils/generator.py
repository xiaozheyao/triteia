import torch
import torch.nn as nn
from .sparsity import mask_creator


def gen_quant4_NT(m, k, groupsize=-1, device="cuda", prune_n=2, prune_m=4):
    from triteia.python.nn.linear import sparse_low_precision_linear

    maxq = 2**4 - 1
    w = torch.randn((m, k), dtype=torch.half, device=device)
    k_sp = k // 2

    w = w.t()
    if groupsize != -1:
        w = w.reshape((-1, groupsize, m))
        w = w.permute(1, 0, 2)
        w = w.reshape((groupsize, -1))
    s = torch.max(torch.abs(w), 0, keepdim=True)[0]
    s *= 2 / maxq
    w = torch.round(w / s).int()
    w += (maxq + 1) // 2
    w = torch.clamp(w, 0, maxq)
    ref = (w - (maxq + 1) // 2).half() * s
    if groupsize != -1:

        def reshape(w):
            w = w.reshape((groupsize, -1, m))
            w = w.permute(1, 0, 2)
            w = w.reshape((k, m)).contiguous()
            return w

        ref = reshape(ref)
        w = reshape(w)
    mask = mask_creator(w.T, n=prune_n, m=prune_m).cuda().bool()
    uncompress = (mask * ref.T).T

    s = s.reshape((-1, m)).contiguous()
    linear = nn.Linear(k, m)
    linear.weight.data = ref

    layer = sparse_low_precision_linear(256, 256, groupsize=groupsize)
    if groupsize == -1:
        groupsize = k
    layer.k = k
    layer.n = m
    layer.groupsize = groupsize
    layer.B = torch.empty((k_sp // 16, m * 16 // 8), dtype=torch.int, device=device)
    layer.meta = torch.empty((m, k // 16), dtype=torch.int16, device=device)
    layer.s = torch.empty(
        (k_sp // (groupsize // 2), m), dtype=torch.half, device=device
    )
    layer.pack(uncompress, s, True)
    q = layer.B
    s = layer.s
    meta = layer.meta

    return uncompress, q, s, meta
