import torch
import unittest
from triteia.python.nn.sdpa import sdpa

class TestSDPAOp(unittest.TestCase):
    def run_problem(self, bsz, seq_len, num_heads, head_dim, is_causal, impl):
        q = torch.randn((bsz, seq_len, num_heads, head_dim), dtype=torch.float16)
        k = torch.randn((bsz, seq_len, num_heads, head_dim), dtype=torch.float16)
        v = torch.randn((bsz, seq_len, num_heads, head_dim), dtype=torch.float16)
        sdpa(q, k, v, is_causal=is_causal, impl=impl)
    
    def test_tiny(self):
        output = self.run_problem(1, 16, 4, 4, True, "fa")
        ref = self.run_problem(1, 16, 4, 4, True, "torch")
    
if __name__ == "__main__":
    unittest.main()
