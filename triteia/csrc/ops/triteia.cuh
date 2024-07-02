#pragma once
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
#include "marlin.cuh"

namespace triteia {
const int ERR_PROB_SHAPE = 1;
const int ERR_KERN_SHAPE = 2;
const int THREADS = 256;
const int STAGES = 4;  // 4 pipeline stages fit into shared memory
const int SHARED_MEM =
    96 * 1024;  // max shared memory on compute capability 8.6 (< 8.0)

#define gpuErrchk(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "Device Side Error: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort) exit(code);
  }
}
#define CALL_MM_2_4(THREAD_M_BLOCKS, THREAD_N_BLOCKS, THREAD_K_BLOCKS,         \
                    GROUP_BLOCKS)                                              \
  else if (thread_m_blocks == THREAD_M_BLOCKS &&                               \
           thread_n_blocks == THREAD_N_BLOCKS &&                               \
           thread_k_blocks == THREAD_K_BLOCKS &&                               \
           group_blocks == GROUP_BLOCKS) {                                     \
    marlin::Marlin_2_4<THREADS, THREAD_N_BLOCKS, THREAD_M_BLOCKS,        \
                             THREAD_K_BLOCKS, STAGES, GROUP_BLOCKS>            \
        <<<blocks, THREADS, SHARED_MEM, stream>>>(A_ptr, B_ptr, meta_ptr,      \
                                                  C_ptr, s_ptr, count, prob_n, \
                                                  prob_k, locks_ptr);          \
  }
}  // namespace triteia