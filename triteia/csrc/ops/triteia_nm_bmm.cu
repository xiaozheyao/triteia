#ifndef TRITEIA_CUDA_KERNEL_CUH_
#define TRITEIA_CUDA_KERNEL_CUH_
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>

#include "marlin.cuh"
#include "triteia.cuh"

namespace triteia {

template <const int threads, const int thread_m_blocks,
          const int thread_n_blocks, const int thread_k_blocks,
          const int stages, const int group_blocks = -1>
__global__ void BMM_2_4(
    /**
     * A:    [n, k]: n: #reqs, k: in features
     * B:    [n, k/32, 2*m]: n: #reqs, k: in features, m: out features
     * C:    [n, m]: n: #reqs, m: out features
     * s:    [n, 1, m]: n: #reqs, m: out features
     * meta: [n, k, m/16]: n: #reqs, k: in features, m: out features
     */
    const int4 *__restrict__ A, const int4 *__restrict__ B,
    const int4 *__restrict__ meta, int4 *__restrict__ C,
    const int4 *__restrict__ s, cudaStream_t stream, int blocks, int prob_m,
    int prob_n, int prob_k, int *locks, int max_par) {
  // printf("prob_m: %d, prob_n: %d, prob_k: %d\n", prob_m, prob_n, prob_k);
  // 1 int4 pointer = 4 x 32 bit
  // B: 32 bit packed, 4

  // A: 16 bit, 8
  // C: 16 bit, 8
  // s: 16 bit, 8
  // meta: 16 bit, 8
  // locks: 32 bit
  printf(
      "thread_m_blocks: %d, thread_n_blocks: %d, thread_k_blocks: %d, "
      "group_blocks: %d\n",
      thread_m_blocks, thread_n_blocks, thread_k_blocks, group_blocks);
  for (int batch_idx = 0; batch_idx < prob_m; batch_idx++) {
    const int4 *__restrict__ A_ptr = A + batch_idx * prob_k / 8;
    const int4 *__restrict__ B_ptr = B + batch_idx * prob_k * prob_n / 16 / 4;
    const int4 *__restrict__ meta_ptr =
        meta + batch_idx * prob_k * prob_n / 16 / 8;

    const int4 *__restrict__ s_ptr = s + batch_idx * prob_n / 8;
    int4 *__restrict__ C_ptr = C + batch_idx * prob_n / 8;
    int *locks_ptr = locks + batch_idx * prob_k;
    const int possible_thread_m_blocks = 1;

    marlin::Marlin_2_4<threads, possible_thread_m_blocks, thread_n_blocks,
                       thread_k_blocks, stages, group_blocks>
        <<<blocks, THREADS, SHARED_MEM, stream>>>(
            A_ptr, B_ptr, meta_ptr, C_ptr, s_ptr, 1, prob_n, prob_k, locks_ptr);
  }
};

#define CALL_IF_BMM_2_4(THREAD_M_BLOCKS, THREAD_N_BLOCKS, THREAD_K_BLOCKS,    \
                        GROUP_BLOCKS)                                         \
  else if (thread_m_blocks == THREAD_M_BLOCKS &&                              \
           thread_n_blocks == THREAD_N_BLOCKS &&                              \
           thread_k_blocks == THREAD_K_BLOCKS &&                              \
           group_blocks == GROUP_BLOCKS) {                                    \
    cudaFuncSetAttribute(BMM_2_4<THREADS, THREAD_N_BLOCKS, THREAD_M_BLOCKS,   \
                                 THREAD_K_BLOCKS, STAGES, GROUP_BLOCKS>,      \
                         cudaFuncAttributeMaxDynamicSharedMemorySize,         \
                         SHARED_MEM);                                         \
    BMM_2_4<THREADS, THREAD_N_BLOCKS, THREAD_M_BLOCKS, THREAD_K_BLOCKS,       \
            STAGES, GROUP_BLOCKS><<<blocks, THREADS, SHARED_MEM, stream>>>(   \
        A_ptr, B_ptr, meta_ptr, C_ptr, s_ptr, stream, blocks, prob_n, prob_m, \
        prob_k, locks, max_par);                                              \
  }

#define Set_Max_SharedMemory(THREAD_M_BLOCKS, THREAD_N_BLOCKS,      \
                             THREAD_K_BLOCKS)                       \
  cudaFuncSetAttribute(                                             \
      marlin::Marlin_2_4<THREADS, THREAD_M_BLOCKS, THREAD_N_BLOCKS, \
                         THREAD_K_BLOCKS, STAGES, -1>,              \
      cudaFuncAttributeMaxDynamicSharedMemorySize, SHARED_MEM);

int triteia_cuda_bmm_2_4(const void *A, const void *B, const void *meta,
                         void *C, void *s, int prob_m, int prob_n, int prob_k,
                         void *workspace, int groupsize = -1, int dev = 0,
                         cudaStream_t stream = 0, int thread_k = -1,
                         int thread_m = -1, int sms = -1, int max_par = 16) {
  int tot_n = prob_n;
  int tot_n_blocks = marlin::ceildiv(tot_n, 16);
  int pad = 16 * tot_n_blocks - tot_n;
  if (sms == -1)
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  if (thread_k == -1 || thread_m == -1) {
    thread_m = 128;
    thread_k = 128;
  }
  int thread_k_blocks = thread_k / 32;  // 2:4 version with m16n8k32 instruction
  int thread_m_blocks = thread_m / 16;
  int group_blocks = (groupsize == -1) ? -1 : groupsize / 16;
  int blocks = sms;

  if (prob_m % thread_m != 0 || prob_k % thread_k != 0 ||
      (group_blocks != -1 && (prob_k / 2) % group_blocks != 0))

    return ERR_PROB_SHAPE;
  if (prob_m == 0 || prob_n == 0 || prob_k == 0) return 0;
  const int4 *A_ptr = (const int4 *)A;
  const int4 *B_ptr = (const int4 *)B;
  const int4 *meta_ptr = (const int4 *)meta;
  int4 *C_ptr = (int4 *)C;
  const int4 *s_ptr = (const int4 *)s;

  int cols = prob_m / thread_m;
  int *locks = (int *)workspace;
  int ret = 0;
  printf("prob_m: %d, prob_n: %d, prob_k: %d\n", prob_m, prob_n, prob_k);
  for (int i = 0; i < tot_n_blocks; i += 4) {
    int thread_n_blocks = tot_n_blocks - i;
    prob_n = tot_n - 16 * i;
    int par = 1;
    printf("thread_n_blocks: %d\n", thread_n_blocks);
    if (thread_n_blocks > 4) {
      // Note that parallel > 1 currently only works for inputs without any
      // padding
      par = (16 * thread_n_blocks - pad) / 64;
      if (par > max_par) par = max_par;
      prob_n = 64 * par;
      i += 4 * (par - 1);
      thread_n_blocks = 4;
    }
    // For compilation speed, we only define the kernel configurations that have
    // seemed useful (in terms of performance) in our testing, however many more
    // are, in principle, possible.
    if (false) {
    }  //         BMxBNxBK,   group
    CALL_IF_BMM_2_4(8, 1, 4, -1)  // e.g., 16x128x128
    CALL_IF_BMM_2_4(8, 2, 4, -1)
    CALL_IF_BMM_2_4(8, 4, 4, -1)   // e.g., 16x128x128
    CALL_IF_BMM_2_4(16, 1, 2, -1)  // e.g., 16x256x64
    CALL_IF_BMM_2_4(16, 2, 2, -1)  // e.g.. 32x256x64
    CALL_IF_BMM_2_4(16, 3, 2, -1)
    CALL_IF_BMM_2_4(16, 4, 2, -1)
    CALL_IF_BMM_2_4(32, 1, 1, -1)  // e.g., 16x256x64
    CALL_IF_BMM_2_4(32, 2, 1, -1)  // e.g.. 32x256x64
    CALL_IF_BMM_2_4(32, 3, 1, -1)
    CALL_IF_BMM_2_4(32, 4, 1, -1)
    else ret = ERR_KERN_SHAPE;

    A_ptr += 16 * thread_n_blocks * (prob_k / 8) * par;
    C_ptr += 16 * thread_n_blocks * (prob_m / 8) * par;
  }
  return ret;
}
#endif // TRITEIA_CUDA_KERNEL_CUH_
}  // namespace triteia
