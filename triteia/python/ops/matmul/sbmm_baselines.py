import torch
from triteia.python.ops import gen_batched_sparse_quant4_NT
from triteia.python.ops.utils.generator import generate_model_distribution

def sbmm_fp16_bmm(
        indices, y, x, weight
    ):
    if torch.all(indices == -1):
        return y
    mask = indices != -1
    valid_indices = indices[mask]
    x = x[mask, :].unsqueeze(1)
    valid_weights = weight.index_select(0, valid_indices)
    output = torch.bmm(x, valid_weights).squeeze(1)
    y[mask] += output
    return y

def sbmm_fp16_forloop(
    indices, y, x, weight
):
    if torch.all(indices == -1):
        return y
    mask = indices != -1
    valid_indices = indices[mask]
    unique_indices, counts = torch.unique(valid_indices, sorted=False, return_counts=True)
    for id, count in zip(unique_indices, counts):
        idx_mask = indices == id
        inp = x[idx_mask]
        output = torch.matmul(inp, weight[id])
        y[idx_mask] += output
    return y

if __name__=="__main__":
    distribution = 'uniform'
    nr = 20
    nm = 5
    indices = generate_model_distribution(distribution, nr, nm)
    indices = torch.sort(indices)[0]
    ref_weights = []
    groupsize = -1
    dev = "cuda"
    m,k = 1024, 1024
    weight_ref, qweight, scale, meta = gen_batched_sparse_quant4_NT(
        nm, m, k, groupsize=groupsize, device=dev
    )
    x = torch.randn((nr, k), dtype=torch.float16, device=dev)
    
    y_forloop = sbmm_fp16_forloop(
        indices, torch.zeros((nr, m), dtype=torch.float16, device=dev), x, weight_ref)
    
    y_bmm = sbmm_fp16_bmm(
        indices, torch.zeros((nr, m), dtype=torch.float16, device=dev), x, weight_ref)
    
    print(y_forloop)
    print(y_bmm)
    print(torch.allclose(y_forloop, y_bmm, atol=1e-1, rtol=1e-3))