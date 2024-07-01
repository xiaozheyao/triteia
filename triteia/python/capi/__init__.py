import marlin_cuda


def marlin_mul_2_4(
    A, B, meta, C, s, workspace, thread_k=-1, thread_m=-1, sms=-1, max_par=16
):
    """Marlin FP16x(INT4+2:4 sparsity) multiply; can be used within `torch.compile`.
    ----
    Parameters:
        A: `torch.int` weight matrix of original shape `(m, k)` in Marlin format
        B: `torch.half` input matrix of shape `(n, k/2)` in column-major layout
        meta: `torch.int` metadata information for 2:4 sparsity
        C: `torch.half` out matrix of shape `(n, m)` in column-major layout
        s: `torch.half` scales of shape `(n / groupsize /2, m)`
        workspace: `torch.int` tensor with at least `m / 128 * max_par` entries that are all zero
        thread_k: `k` size of a thread_tile in `A` (can usually be left as auto -1)
        thread_m: `m` size of a thread_tile in `A` (can usually be left as auto -1)
        sms: number of SMs to use for the kernel (can usually be left as auto -1)
        max_par: maximum number of batch 64 problems to solve in parallel for large input sizes
    ----
    """
    marlin_cuda.mul_2_4(A, B, meta, C, s, workspace, thread_k, thread_m, sms, max_par)
