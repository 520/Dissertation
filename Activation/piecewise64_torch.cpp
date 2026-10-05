#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <torch/library.h>
#include <algorithm>
#include <cmath>
#include <limits>
#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace {
inline float scalar(float x, const float* knots, const float* values, const float* slopes) {
    if (std::isnan(x)) return x;
    if (x <= knots[0]) return 0.0f;
    if (x >= knots[10]) return x;
    // Balanced binary search, at most four comparisons for nine boundaries.
    const int i = static_cast<int>(std::upper_bound(knots + 1, knots + 10, x) - (knots + 1));
    return std::fma(x - knots[i], slopes[i], values[i]);
}

#if defined(__aarch64__)
struct VectorTable {
    uint8x16x4_t blocks;
    explicit VectorTable(const float* table) {
        for (int j = 0; j < 4; ++j) blocks.val[j] = vreinterpretq_u8_f32(vld1q_f32(table + 4*j));
    }
};
inline float32x4_t gather16(const VectorTable& table, int32x4_t indices) {
    // NEON has no float gather. Four-register byte lookup gathers four floats.
    const uint8x16_t replicate = {0,0,0,0,4,4,4,4,8,8,8,8,12,12,12,12};
    const uint8x16_t offsets = {0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3};
    const uint8x16_t bytes = vaddq_u8(vshlq_n_u8(vqtbl1q_u8(vreinterpretq_u8_s32(indices), replicate), 2), offsets);
    return vreinterpretq_f32_u8(vqtbl4q_u8(table.blocks, bytes));
}

void simd_range(const float* input, float* output, int64_t begin, int64_t end,
                const float* knots, const float* values, const float* slopes,
                const float* boundaries, const float* intercepts) {
    const VectorTable tree_table(boundaries), slope_table(slopes), intercept_table(intercepts);
    int64_t j = begin;
    for (; j + 4 <= end; j += 4) {
        const float32x4_t x = vld1q_f32(input + j);
        // Eytzinger layout: one comparison per level, no lo/hi bookkeeping.
        // Pad the 10-segment search to 16 leaves with +infinity boundaries.
        int32x4_t node = vdupq_n_s32(0);
        for (int depth = 0; depth < 4; ++depth) {
            const int32x4_t right = vreinterpretq_s32_u32(vshrq_n_u32(vcgeq_f32(x, gather16(tree_table, node)), 31));
            node = vaddq_s32(vaddq_s32(vshlq_n_s32(node, 1), vdupq_n_s32(1)), right);
        }
        const int32x4_t index = vminq_s32(vsubq_s32(node, vdupq_n_s32(15)), vdupq_n_s32(9));
        float32x4_t y = vfmaq_f32(gather16(intercept_table, index), x, gather16(slope_table, index));
        y = vbslq_f32(vcleq_f32(x, vdupq_n_f32(knots[0])), vdupq_n_f32(0), y);
        y = vbslq_f32(vcgeq_f32(x, vdupq_n_f32(knots[10])), x, y);
        vst1q_f32(output + j, y);
    }
    for (; j < end; ++j) output[j] = scalar(input[j], knots, values, slopes);
}
#endif

at::Tensor forward(const at::Tensor& input, const at::Tensor& k, const at::Tensor& v, const at::Tensor& s) {
    TORCH_CHECK(input.device().is_cpu() && input.scalar_type() == at::kFloat, "Expected CPU FP32 input");
    for (const auto& t : {k, v, s}) {
        TORCH_CHECK(t.device().is_cpu() && t.scalar_type() == at::kFloat && t.is_contiguous(), "Expected contiguous CPU FP32 tables");
    }
    TORCH_CHECK(k.numel() == 11 && v.numel() == 11 && s.numel() == 10, "Invalid table sizes");
    const auto x = input.contiguous();
    auto y = at::empty_like(x);
    alignas(16) float knots[16] = {}, values[16] = {}, slopes[16] = {}, boundaries[16], intercepts[16] = {};
    std::copy_n(k.const_data_ptr<float>(), 11, knots);
    std::copy_n(v.const_data_ptr<float>(), 11, values);
    std::copy_n(s.const_data_ptr<float>(), 10, slopes);
    std::fill_n(boundaries, 16, std::numeric_limits<float>::infinity());
    constexpr int midpoints[15] = {8,4,12,2,6,10,14,1,3,5,7,9,11,13,15};
    for (int j = 0; j < 15; ++j)
        if (midpoints[j] <= 9) boundaries[j] = knots[midpoints[j]];
    for (int j = 0; j < 10; ++j) intercepts[j] = std::fma(-knots[j], slopes[j], values[j]);
    const float* in = x.const_data_ptr<float>();
    float* out = y.mutable_data_ptr<float>();
    at::parallel_for(0, x.numel(), 32768, [&](int64_t begin, int64_t end) {
#if defined(__aarch64__)
        simd_range(in, out, begin, end, knots, values, slopes, boundaries, intercepts);
#else
        for (int64_t j = begin; j < end; ++j) out[j] = scalar(in[j], knots, values, slopes);
#endif
    });
    return y;
}
}

TORCH_LIBRARY(dissertation_piecewise64, m) {
    m.def("forward(Tensor input, Tensor knots, Tensor values, Tensor slopes) -> Tensor");
}
TORCH_LIBRARY_IMPL(dissertation_piecewise64, CPU, m) {
    m.impl("forward", &forward);
}
