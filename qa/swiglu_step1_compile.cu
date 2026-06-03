// F2 route#2 · P1 Step-1 compile-verify: type config + two-accumulator TMEM fit (no full kernel).
// Compiles the SwiGluConfig CollectiveBuilder for SM100 + forces a partition_fragment_C accumulator.
// Build (in container):
//   nvcc -std=c++17 -arch=sm_100a --expt-relaxed-constexpr -I <gemm_dir> \
//        -I 3rdparty/cutlass/include -I 3rdparty/cutlass/tools/util/include -c swiglu_step1_compile.cu
#include "cute/tensor.hpp"
#include "cutlass_grouped_gemm_swiglu.cuh"

using namespace transformer_engine::grouped_gemm_swiglu;
using Cfg = SwiGluConfig<cutlass::bfloat16_t, cutlass::bfloat16_t>;

// G1 at compile time: two N=128 fp32 accumulators must fit the 512-col TMEM.
static_assert(static_cast<uint32_t>(SwiGluTmem::kEnd) <= 512u,
              "two accumulators exceed 512-col TMEM");

// Force device instantiation of the CollectiveMma type-provider + TiledMma + accumulator fragment.
__global__ void probe() {
  typename Cfg::TiledMma mma;
  auto acc = cute::partition_fragment_C(mma, cute::take<0, 2>(typename Cfg::TileShape{}));
  if (threadIdx.x == 99999) {
    (void)cute::size(acc);
  }
}

int main() { return 0; }
