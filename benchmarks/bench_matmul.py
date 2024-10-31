import torch
from triteia.python.ops import matmul_4bit_2_4, gen_sparse_quant4_NT
from triteia.python.utils import (
    timing_function,
    print_results_table,
    export_benchmark_results,
)

flops_func = lambda m, n, k: m * n * (2*k-1)

def benchmark(m, n, k, dev="cuda", groupsize=-1):
    repeats = 1
    x = torch.randn((n, k), dtype=torch.float16, device=dev)
    weight_ref, qweight, scale, meta = gen_sparse_quant4_NT(
        m, k, groupsize=groupsize, device=dev
    )
    print(x.shape, weight_ref.shape)
    def fp16_func(x, weight_ref):
        return torch.matmul(x, weight_ref)

    def w4_2_4_func(qweight, x, meta, scale):
        return matmul_4bit_2_4(qweight, x, meta, scale)
    w4_2_4_result = timing_function(
        w4_2_4_func,
        flops_func,
        kwargs={
            "m": m,
            "n": n,
            "k": k,
            "qweight": qweight,
            "x": x,
            "meta": meta,
            "scale": scale,
        },
        repeats=repeats,
    )
    fp16_result = timing_function(
        fp16_func,
        flops_func,
        kwargs={"m": m, "n": n, "k": k, "x": x, "weight_ref": weight_ref},
        repeats=repeats,
    )
    results = [fp16_result, w4_2_4_result]
    print_results_table(f"matmul m={m},n={n},k={k}", results)
    return results


if __name__ == "__main__":
    results = []
    results.append(benchmark(4096, 24, 4096))
    # export_benchmark_results(results, ".local/matmul_bench.json")
