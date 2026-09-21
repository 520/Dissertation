#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <torch/library.h>

#include <algorithm>
#include <cmath>
#include <cstdint>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace {

struct LUT21Tables {
    float values[21];
    float deltas[20];

    LUT21Tables()
    {
        for (int i = 0; i < 21; ++i) {
            const float x = -5.0f + 0.5f * static_cast<float>(i);
            values[i] = x / (1.0f + std::exp(-x));
        }
        for (int i = 0; i < 20; ++i)
            deltas[i] = values[i + 1] - values[i];
    }
};

inline float lut21_scalar(float x, const LUT21Tables& table)
{
    const float clipped = std::min(std::max(x, -5.0f), 5.0f);
    const float position = (clipped + 5.0f) * 2.0f;
    const int index = std::min(static_cast<int>(position), 19);
    const float fraction = position - static_cast<float>(index);
    float interpolated = std::fma(
        fraction, table.deltas[index], table.values[index]
    );
    interpolated = x <= -5.0f ? 0.0f : interpolated;
    return x >= 5.0f ? x : interpolated;
}

#if defined(__aarch64__)
inline uint8x16_t lookup_20_f32_bytes(const float* table,
                                     uint8x16_t byte_indices)
{
    uint8x16x4_t first_16;
    first_16.val[0] = vreinterpretq_u8_f32(vld1q_f32(table));
    first_16.val[1] = vreinterpretq_u8_f32(vld1q_f32(table + 4));
    first_16.val[2] = vreinterpretq_u8_f32(vld1q_f32(table + 8));
    first_16.val[3] = vreinterpretq_u8_f32(vld1q_f32(table + 12));
    const uint8x16_t last_4 = vreinterpretq_u8_f32(vld1q_f32(table + 16));

    const uint8x16_t low = vqtbl4q_u8(first_16, byte_indices);
    const uint8x16_t high_indices = vsubq_u8(
        byte_indices, vdupq_n_u8(64)
    );
    const uint8x16_t high = vqtbl1q_u8(last_4, high_indices);
    return vorrq_u8(low, high);
}

inline void lut21_neon_range(const float* input, float* output,
                             int64_t begin, int64_t end,
                             const LUT21Tables& table)
{
    /* Expand four uint32 indices into 16 byte offsets for NEON TBL. */
    alignas(16) static constexpr uint8_t duplicate_lanes[16] = {
        0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12
    };
    alignas(16) static constexpr uint8_t byte_in_float[16] = {
        0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3
    };
    const uint8x16_t duplicate = vld1q_u8(duplicate_lanes);
    const uint8x16_t offsets = vld1q_u8(byte_in_float);
    const float32x4_t lower = vdupq_n_f32(-5.0f);
    const float32x4_t upper = vdupq_n_f32(5.0f);
    const float32x4_t five = vdupq_n_f32(5.0f);
    const float32x4_t two = vdupq_n_f32(2.0f);
    const int32x4_t maximum_index = vdupq_n_s32(19);
    const float32x4_t zero = vdupq_n_f32(0.0f);

    int64_t i = begin;
    for (; i + 4 <= end; i += 4) {
        const float32x4_t x = vld1q_f32(input + i);
        const float32x4_t clipped = vmaxq_f32(vminq_f32(x, upper), lower);
        const float32x4_t position = vmulq_f32(vaddq_f32(clipped, five), two);
        const int32x4_t index = vminq_s32(vcvtq_s32_f32(position), maximum_index);
        const float32x4_t fraction = vsubq_f32(
            position, vcvtq_f32_s32(index)
        );

        const uint8x16_t index_bytes = vreinterpretq_u8_s32(index);
        const uint8x16_t repeated = vqtbl1q_u8(index_bytes, duplicate);
        const uint8x16_t byte_indices = vaddq_u8(
            vshlq_n_u8(repeated, 2), offsets
        );
        const float32x4_t values = vreinterpretq_f32_u8(
            lookup_20_f32_bytes(table.values, byte_indices)
        );
        const float32x4_t deltas = vreinterpretq_f32_u8(
            lookup_20_f32_bytes(table.deltas, byte_indices)
        );
        float32x4_t result = vfmaq_f32(values, fraction, deltas);
        result = vbslq_f32(vcleq_f32(x, lower), zero, result);
        result = vbslq_f32(vcgeq_f32(x, upper), x, result);
        vst1q_f32(output + i, result);
    }
    for (; i < end; ++i)
        output[i] = lut21_scalar(input[i], table);
}
#endif

at::Tensor lut21_silu_cpu(const at::Tensor& input)
{
    TORCH_CHECK(input.device().is_cpu(), "LUT21 only supports CPU tensors");
    TORCH_CHECK(input.scalar_type() == at::kFloat,
                "LUT21 requires a float32 tensor");

    const at::Tensor contiguous = input.contiguous();
    at::Tensor output = at::empty_like(contiguous);
    const float* input_data = contiguous.const_data_ptr<float>();
    float* output_data = output.mutable_data_ptr<float>();
    const int64_t count = contiguous.numel();
    static const LUT21Tables table;

    /* Reuses PyTorch's intra-op thread pool; small tensors remain single-threaded. */
    constexpr int64_t grain_size = 32768;
    at::parallel_for(0, count, grain_size, [&](int64_t begin, int64_t end) {
#if defined(__aarch64__)
        lut21_neon_range(input_data, output_data, begin, end, table);
#else
        int64_t i = begin;
        for (; i + 4 <= end; i += 4) {
            output_data[i] = lut21_scalar(input_data[i], table);
            output_data[i + 1] = lut21_scalar(input_data[i + 1], table);
            output_data[i + 2] = lut21_scalar(input_data[i + 2], table);
            output_data[i + 3] = lut21_scalar(input_data[i + 3], table);
        }
        for (; i < end; ++i)
            output_data[i] = lut21_scalar(input_data[i], table);
#endif
    });
    return output;
}

}  // namespace

TORCH_LIBRARY(dissertation_lut21, library)
{
    library.def("forward(Tensor input) -> Tensor");
}

TORCH_LIBRARY_IMPL(dissertation_lut21, CPU, library)
{
    library.impl("forward", &lut21_silu_cpu);
}
