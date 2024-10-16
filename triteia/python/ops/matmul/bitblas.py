import torch
try:
    import bitblas
except ImportError:
    print("bitblas not installed, some features will not work")

def bb_gen_weight(m, n, k, w_dtype: str):
    if w_dtype == "int4":
        matmul_config = bitblas.MatmulConfig(
            M=m,
            N=n,
            K=k,
            A_dtype="float16",
            W_dtype=w_dtype,
            # we use fp32 accumulator to be consistent with sparse marlin for benchmarking
            accum_dtype="float32",
            out_dtype="float16",
            layout="nt",
            with_bias=False,
            group_size=None,
            with_scaling=False,
            with_zeros=False,
            zeros_mode=None,
        )
        matmul = bitblas.Matmul(config=matmul_config)
        weight_tensor = torch.randint(0, 7, (n, k), dtype=torch.int8).cuda()
        # Transform weight tensor to int4 data type
        weight_tensor_int = matmul.transform_weight(weight_tensor)
        return matmul, weight_tensor_int
    elif w_dtype == "int2":
        matmul_config = bitblas.MatmulConfig(
            M=m,
            N=n,
            K=k,
            A_dtype="float16",
            W_dtype=w_dtype,
            # we use fp32 accumulator to be consistent with sparse marlin for benchmarking
            accum_dtype="float32",
            out_dtype="float16",
            layout="nt",
            with_bias=False,
            group_size=None,
            with_scaling=False,
            with_zeros=False,
            zeros_mode=None,
        )
        matmul = bitblas.Matmul(config=matmul_config)
        weight_tensor = torch.randint(0, 1, (n, k), dtype=torch.int8).cuda()
        # Transform weight tensor to int4 data type
        weight_tensor_int = matmul.transform_weight(weight_tensor)
        return matmul, weight_tensor_int
    elif w_dtype == "int1":
        matmul_config = bitblas.MatmulConfig(
            M=m,
            N=n,
            K=k,
            A_dtype="float16",
            W_dtype=w_dtype,
            # we use fp32 accumulator to be consistent with sparse marlin for benchmarking
            accum_dtype="float32",
            out_dtype="float16",
            layout="nt",
            with_bias=False,
            group_size=None,
            with_scaling=False,
            with_zeros=False,
            zeros_mode=None,
        )
        matmul = bitblas.Matmul(config=matmul_config)
        weight_tensor = torch.randint(0, 1, (n, k), dtype=torch.int8).cuda()
        # Transform weight tensor to int4 data type
        weight_tensor_int = matmul.transform_weight(weight_tensor)
        return matmul, weight_tensor_int
    else:
        raise ValueError(f"Unsupported weight data type {w_dtype}")
    
def matmul_xbit_perf_only(op, qweight, x):
    output_tensor = op(x, qweight)
    return output_tensor
