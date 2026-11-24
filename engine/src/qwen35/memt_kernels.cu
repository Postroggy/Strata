// src/qwen35/memt_kernels.cu - MemT-1.0's Native Memory Block on CUDA.
//
// Four launches per memory layer per forward (gates, key_effective, read, update);
// the engine's three linear projections stay on its own MMVQ / FP32 GEMM paths,
// so the memory block uses the same weights pipeline as the rest of the model.
// The kernel bodies live in memt_kernels_impl.cuh, where memt_kernels_cpu_test
// compiles the same code for the CPU (one simulated thread per block).
//
// The arithmetic is memt_math.h's, in FP32.  memt_sglang runs the same read and
// update in the model's BF16 (`tl.dot` on the operand dtype); this engine
// computes them in FP32, the same recurrence at higher precision - the P6
// tolerance against memt_sglang is measured, not assumed bitwise.
#include "memt_kernels.cuh"

#include <cuda_runtime.h>

#include "memt_kernels_impl.cuh"

#include <cstdio>
#include <cstdlib>

namespace strata::qwen35::memt {
namespace {

constexpr int kThreads = 256;

#define MEMT_CK(call)                                                                          \
    do {                                                                                       \
        const cudaError_t memt_err_ = (call);                                                   \
        if (memt_err_ != cudaSuccess) {                                                         \
            std::fprintf(stderr, "strata-q36: memt kernel %s failed: %s\n", #call,              \
                         cudaGetErrorString(memt_err_));                                        \
            std::exit(1);                                                                       \
        }                                                                                       \
    } while (0)

inline cudaStream_t as_stream(void* stream) { return reinterpret_cast<cudaStream_t>(stream); }

}  // namespace

void gates(const float* fdown, const float* bias, const float* decay_scale_log, float* decay, int tokens, int heads,
           int dim, void* stream) {
    const int total = tokens * heads * dim;
    if (total <= 0) return;
    const int blocks = (total + kThreads - 1) / kThreads;
    device::k_gates<<<blocks, kThreads, 0, as_stream(stream)>>>(fdown, bias, decay_scale_log, decay, total, heads, dim);
    MEMT_CK(cudaGetLastError());
}

void latents(const float* proj, float* decay_latent, float* og_latent, int tokens, int64_t proj_stride,
             int64_t head_dim, int rank, void* stream) {
    const int total = tokens * rank;
    if (total <= 0) return;
    const int blocks = (total + kThreads - 1) / kThreads;
    device::k_latents<<<blocks, kThreads, 0, as_stream(stream)>>>(proj, decay_latent, og_latent, tokens, proj_stride,
                                                                  head_dim, rank);
    MEMT_CK(cudaGetLastError());
}

void key_effective(const float* proj, float* k_eff, int tokens, int heads, int dim, int64_t proj_stride,
                   void* stream) {
    if (tokens * heads <= 0) return;
    device::k_key<<<tokens * heads, kThreads, (size_t) kThreads * sizeof(float), as_stream(stream)>>>(proj, k_eff,
                                                                                                      heads, dim,
                                                                                                      proj_stride);
    MEMT_CK(cudaGetLastError());
}

void read(const float* proj, const float* og, const float* o_norm, const float* state, float* att, int tokens,
          int heads, int dim, int64_t proj_stride, float eps, float softcap, void* stream) {
    if (tokens * heads <= 0) return;
    const size_t shared = (size_t) (2 * dim + kThreads) * sizeof(float);
    device::k_read<<<tokens * heads, kThreads, shared, as_stream(stream)>>>(proj, og, o_norm, state, att, heads, dim,
                                                                           proj_stride, eps, softcap);
    MEMT_CK(cudaGetLastError());
}

void update(const float* proj, const float* k_eff, const float* decay, float* state, int tokens, int heads, int dim,
            int64_t proj_stride, void* stream) {
    if (tokens <= 0 || heads <= 0) return;
    const size_t shared = (size_t) (2 * dim) * sizeof(float);
    device::k_update<<<heads, kThreads, shared, as_stream(stream)>>>(proj, k_eff, decay, state, tokens, heads, dim,
                                                                    proj_stride);
    MEMT_CK(cudaGetLastError());
}

}  // namespace strata::qwen35::memt
