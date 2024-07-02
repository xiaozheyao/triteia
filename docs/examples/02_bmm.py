import torch
from triteia.python.ops import bmm_4bit_2_4_forloop, gen_batched_sparse_quant4_NT

dev = "cuda"
b=16
n=1
m=256
p=512
groupsize = -1

x = torch.randn((b, 1, m), dtype=torch.float16, device=dev)
weight_ref, qweight, scale, meta = gen_batched_sparse_quant4_NT(
    b, m, p, groupsize=groupsize, device=dev
)
# weight_ref = weight_ref.permute(0, 2, 1)
fp16_output = torch.bmm(x, weight_ref)
qs_output = bmm_4bit_2_4_forloop(qweight, x, meta, scale)
print(f"weight_ref: {weight_ref.shape}, qweight: {qweight.shape}, scale: {scale.shape}, meta: {meta.shape}")
print(fp16_output)
print(qs_output)
torch.cuda.synchronize()
