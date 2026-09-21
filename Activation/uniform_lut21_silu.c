#include <math.h>
#include <stddef.h>

#if defined(_WIN32)
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

static float lut_y[21];
static float lut_delta[20];

EXPORT void initialize_lut21(void)
{
    for (int i = 0; i < 21; ++i) {
        const float x = -5.0f + 0.5f * (float)i;
        lut_y[i] = x / (1.0f + expf(-x));
    }
    for (int i = 0; i < 20; ++i)
        lut_delta[i] = lut_y[i + 1] - lut_y[i];
}

static inline float lut21_scalar(float x)
{
    /* This branchless form is easier for the compiler to SIMD-vectorize. */
    const float clipped = fminf(fmaxf(x, -5.0f), 5.0f);
    const float position = (clipped + 5.0f) * 2.0f;
    int index = (int)position;
    if (index > 19)
        index = 19;
    const float fraction = position - (float)index;
    float interpolated = fmaf(fraction, lut_delta[index], lut_y[index]);
    interpolated = x <= -5.0f ? 0.0f : interpolated;
    return x >= 5.0f ? x : interpolated;
}

EXPORT void lut21_silu_f32(const float *restrict input,
                           float *restrict output,
                           size_t count)
{
    size_t i = 0;

    /* Independent lookups help overlap conversion and table-load latency. */
    for (; i + 4 <= count; i += 4) {
        output[i] = lut21_scalar(input[i]);
        output[i + 1] = lut21_scalar(input[i + 1]);
        output[i + 2] = lut21_scalar(input[i + 2]);
        output[i + 3] = lut21_scalar(input[i + 3]);
    }
    for (; i < count; ++i)
        output[i] = lut21_scalar(input[i]);
}
