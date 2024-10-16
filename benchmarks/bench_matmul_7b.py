import gc
import torch
from triteia.python.ops import (
    matmul_4bit_2_4,
    gen_sparse_quant4_NT,
    matmul_xbit_perf_only,
    bb_gen_weight,
)
from triteia.python.utils import (
    timing_function,
    print_results_table,
    export_benchmark_results,
)
flops_func = lambda m, n, k: m * n * (2*k-1)


def benchmark(m, n, k, dev="cuda", groupsize=-1):
    repeats = 5
    int1_op, int1_weight = bb_gen_weight(n, m, k, "int1")
    int2_op, int2_weight = bb_gen_weight(n, m, k, "int2")
    int4_op, int4_weight = bb_gen_weight(n, m, k, "int4")
    x = torch.randn((n, k), dtype=torch.float16, device=dev)
    weight_ref, qweight, scale, meta = gen_sparse_quant4_NT(
        m, k, groupsize=groupsize, device=dev
    )
    
    def fp16_func(x, weight_ref):
        return torch.matmul(x, weight_ref)

    def w4_2_4_func(qweight, x, meta, scale):
        return matmul_4bit_2_4(qweight, x, meta, scale)

    # just for naming...
    def bb_int1_func(op, qweight, x):
        return matmul_xbit_perf_only(op, qweight, x)
    def bb_int2_func(op, qweight, x):
        return matmul_xbit_perf_only(op, qweight, x)
    def bb_int4_func(op, qweight, x):
        return matmul_xbit_perf_only(op, qweight, x)

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
    bitblas_int1_result = timing_function(
        bb_int1_func,
        flops_func,
        kwargs={
            "m": m,
            "n": n,
            "k": k,
            "op": int1_op,
            "x": x,
            "qweight": int1_weight,
        },
        repeats=repeats,
    )
    bitblas_int2_result = timing_function(
        bb_int2_func,
        flops_func,
        kwargs={
            "m": m,
            "n": n,
            "k": k,
            "op": int2_op,
            "x": x,
            "qweight": int2_weight,
        },
        repeats=repeats,
    )
    bitblas_int4_result = timing_function(
        bb_int4_func,
        flops_func,
        kwargs={
            "m": m,
            "n": n,
            "k": k,
            "op": int4_op,
            "x": x,
            "qweight": int4_weight,
        },
        repeats=repeats,
    )
    results = [
        w4_2_4_result, 
        fp16_result,
        bitblas_int1_result,
        bitblas_int2_result,
        bitblas_int4_result
    ]
    print_results_table(f"matmul m={m},n={n},k={k}", results)
    return results


if __name__ == "__main__":
    results = []
    batchsizes=[4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    infeatures = 4096
    outfeatures = 4096
    for bsz in batchsizes:
        results.append(benchmark(infeatures, bsz, outfeatures))
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    export_benchmark_results(results, ".local/matmul_bench_13b_3090.json")
