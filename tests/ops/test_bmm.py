import torch
import unittest
from triteia.python.ops import bmm_4bit_2_4_forloop, gen_batched_sparse_quant4_NT
from triteia.python.configs.models.llama import llama_shapes

class TestMatmulOp(unittest.TestCase):
    def run_problem(self, b: int, m: int, n: int, k: int, groupsize=-1, dev="cuda"):
        try:
            print(f"Running bmm problem with b={b} m={m}, n={n}, k={k}")
            x = torch.randn((b, 1, k), dtype=torch.float16, device=dev)
            weight_ref, qweight, scale, meta = gen_batched_sparse_quant4_NT(
                b, m, k, groupsize=groupsize, device=dev
            )
            fp16_output = torch.matmul(x, weight_ref)
            qs_output = bmm_4bit_2_4_forloop(qweight, x, meta, scale)
            torch.cuda.synchronize()
            
            self.assertLess(
                torch.mean(torch.abs(qs_output - fp16_output))
                / torch.mean(torch.abs(fp16_output)),
                0.002,
            )
        except torch.cuda.OutOfMemoryError as e:
            print("Out of memory, skipping")

    def test_tiny(self):
        self.run_problem(16, 256, 16, 256, groupsize=-1)


if __name__ == "__main__":
    unittest.main()
