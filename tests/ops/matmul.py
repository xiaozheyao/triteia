import torch
import unittest
from triteia.python.ops import matmul_4bit_2_4, gen_quant4_NT


class TestMatmulOp(unittest.TestCase):
    def run_problem(self, m: int, n: int, k: int, groupsize=-1, dev="cuda"):
        print(f"Running problem with m={m}, n={n}, k={k}")
        x = torch.randn((n, k), dtype=torch.float16, device=dev)
        weight_ref, qweight, scale, meta = gen_quant4_NT(
            m, k, groupsize=groupsize, device=dev
        )
        fp16_output = torch.matmul(x, weight_ref)
        qs_output = matmul_4bit_2_4(qweight, x, meta, scale)
        torch.cuda.synchronize()
        self.assertLess(
            torch.mean(torch.abs(qs_output - fp16_output))
            / torch.mean(torch.abs(fp16_output)),
            0.002,
        )

    def test_tiny(self):
        self.run_problem(256, 16, 256, groupsize=-1)
        self.run_problem(21504, 16, 4096, groupsize=128)


if __name__ == "__main__":
    unittest.main()
