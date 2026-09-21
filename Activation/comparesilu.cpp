#include <algorithm>
#include <chrono>
#include <cmath>
#include <cerrno>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#if defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#if defined(_WIN32)
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif


using activation_fn = float (*)(float);
using array_kernel = void (*)(const float *, float *, size_t);

static inline float silu_exact(float x);

struct Method {
    const char *key;
    const char *label;
    activation_fn function;
    array_kernel kernel;
};

struct Timing {
    double median_ms;
    double best_ms;
    double ns_per_element;
    double million_elements_per_second;
    float checksum;
};

struct GridLUT {
    int points;
    float minimum;
    float maximum;
    float scale;
    std::vector<float> values;
    std::vector<float> deltas;

    GridLUT(int point_count, float lower, float upper)
        : points(point_count), minimum(lower), maximum(upper),
          scale(static_cast<float>(point_count - 1) / (upper - lower)),
          values(static_cast<size_t>(point_count)),
          deltas(static_cast<size_t>(point_count - 1))
    {
        for (int i = 0; i < points; ++i) {
            float x = minimum
                + (maximum - minimum) * static_cast<float>(i)
                    / static_cast<float>(points - 1);
            values[static_cast<size_t>(i)] = silu_exact(x);
        }
        for (int i = 0; i + 1 < points; ++i)
            deltas[static_cast<size_t>(i)] =
                values[static_cast<size_t>(i + 1)]
                - values[static_cast<size_t>(i)];
    }
};

static inline float silu_exact(float x)
{
    return x / (1.0f + expf(-x));
}

static inline float exp_taylor3(float x)
{
    return 1.0f + x + 0.5f * x * x + (x * x * x) / 6.0f;
}

static inline float silu_taylor3(float x)
{
    /* The third-order expansion of exp(-x) becomes unstable for positive x.
     * At x=1 the Taylor approximation is exactly 0.75. Connect that point
     * linearly to (5, 5), then use SiLU's asymptotic identity tail. */
    if (x >= 5.0f)
        return x;
    if (x > 1.0f)
        return 1.0625f * x - 0.3125f;
    return x / (1.0f + exp_taylor3(-x));
}

static inline float exp_range_poly(float x)
{
    const float inv_ln2 = 1.44269504089f;
    const float ln2 = 0.69314718056f;
    int n = (int)nearbyintf(x * inv_ln2);
    float r = x - (float)n * ln2;
    float r2 = r * r;
    float polynomial = 1.0f + r + 0.5f * r2 + (r2 * r) / 6.0f;
    return ldexpf(polynomial, n);
}

static inline float silu_range_poly(float x)
{
    return x / (1.0f + exp_range_poly(-x));
}

static inline float sigmoid_fast(float x)
{
    return 0.5f * (x / (1.0f + fabsf(x))) + 0.5f;
}

static inline float silu_fast_sigmoid(float x)
{
    return x * sigmoid_fast(x);
}

static inline float hard_silu(float x)
{
    float clipped = fminf(fmaxf(x + 3.0f, 0.0f), 6.0f);
    return x * clipped * (1.0f / 6.0f);
}

static const float pwl_x[] = {
    -5.0f, -3.0f, -2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f, 5.0f
};
static float pwl_y[sizeof(pwl_x) / sizeof(pwl_x[0])];
static float lut_y[21];
/* Interleaved [intercept, slope] allows one paired load per segment. */
static float lut_segments[20][2];

/* Preserve single-rounding FMA via a compiler builtin. On the tested
 * AArch64 target this emits fmadd directly, with no libm call. Targets
 * without hardware FMA may still need the compiler's library fallback.
 */
static inline float lut_multiply_add(float a, float b, float c)
{
    return __builtin_fmaf(a, b, c);
}

static void initialize_tables(void)
{
    size_t count = sizeof(pwl_x) / sizeof(pwl_x[0]);
    for (size_t i = 0; i < count; ++i)
        pwl_y[i] = silu_exact(pwl_x[i]);
    for (size_t i = 0; i < 21; ++i)
        lut_y[i] = silu_exact(-5.0f + 0.5f * (float)i);
    /* Same 20 linear intervals, evaluated as fma(x, slope, intercept). */
    for (size_t i = 0; i < 20; ++i) {
        lut_segments[i][1] = 2.0f * (lut_y[i + 1] - lut_y[i]);
        lut_segments[i][0] = lut_multiply_add(5.0f - 0.5f * (float)i,
                                        lut_segments[i][1], lut_y[i]);
    }
}

static inline float interpolate(float x, float x0, float x1, float y0, float y1)
{
    return y0 + (x - x0) * (y1 - y0) / (x1 - x0);
}

static inline float silu_piecewise_linear(float x)
{
    if (x <= -5.0f)
        return 0.0f;
    if (x >= 5.0f)
        return x;

    size_t count = sizeof(pwl_x) / sizeof(pwl_x[0]);
    for (size_t i = 0; i + 1 < count; ++i) {
        if (x < pwl_x[i + 1])
            return interpolate(x, pwl_x[i], pwl_x[i + 1], pwl_y[i], pwl_y[i + 1]);
    }
    return x;
}

static inline float silu_lut(float x)
{
    if (x != x)
        return lut_y[0];
    /* Outer branches do not need index conversion or table loads. */
    if (x <= -5.0f)
        return 0.0f;
    if (x >= 5.0f)
        return x;

    float position = (x + 5.0f) * 2.0f;
    int index = (int)position;
    /* Rounding can produce position == 20 just below x == 5. */
    if (index > 19)
        index = 19;
    return x * lut_segments[index][1] + lut_segments[index][0];
}

static const GridLUT *active_grid_lut = nullptr;

static inline float grid_lut_value(float x, const GridLUT &lut)
{
    if (x != x)
        return lut.values[0];
    if (x <= lut.minimum)
        return 0.0f;
    if (x >= lut.maximum)
        return x;

    float position = (x - lut.minimum) * lut.scale;
    int index = static_cast<int>(position);
    if (index > lut.points - 2)
        index = lut.points - 2;
    float fraction = position - static_cast<float>(index);
    size_t offset = static_cast<size_t>(index);
    return lut.values[offset] + fraction * lut.deltas[offset];
}

static void kernel_configured_grid_lut(const float *__restrict input,
                                       float *__restrict output, size_t count,
                                       const GridLUT &lut)
{
    for (size_t i = 0; i < count; ++i)
        output[i] = grid_lut_value(input[i], lut);
}

static void kernel_grid_lut(const float *__restrict input,
                            float *__restrict output, size_t count)
{
    kernel_configured_grid_lut(input, output, count, *active_grid_lut);
}

static inline float relu(float x)
{
    return fmaxf(x, 0.0f);
}

#define DEFINE_KERNEL(name, function)                                      \
    static void name(const float *__restrict input, float *__restrict output, \
                     size_t count)                                         \
    {                                                                       \
        for (size_t i = 0; i < count; ++i)                                 \
            output[i] = function(input[i]);                                \
    }

DEFINE_KERNEL(kernel_exact, silu_exact)
DEFINE_KERNEL(kernel_taylor3, silu_taylor3)
DEFINE_KERNEL(kernel_range_poly, silu_range_poly)
DEFINE_KERNEL(kernel_fast_sigmoid, silu_fast_sigmoid)
DEFINE_KERNEL(kernel_hard_silu, hard_silu)
DEFINE_KERNEL(kernel_piecewise, silu_piecewise_linear)
DEFINE_KERNEL(kernel_relu, relu)

static void kernel_lut(const float *__restrict input, float *__restrict output,
                       size_t count)
{
    size_t i = 0;

#if defined(__aarch64__) && defined(__ARM_NEON)
    #if defined(__clang__)
    #pragma clang loop unroll_count(4)
    #endif
    for (; count - i >= 4; i += 4) {
        float32x4_t x = vld1q_f32(input + i);
        float32x4_t shifted = vaddq_f32(x, vdupq_n_f32(5.0f));
        /* Fixed-point conversion includes the factor of two. Unsigned
         * saturation also clamps negatives to zero; cap the upper index. */
        uint32x4_t index = vminq_u32(vcvtq_n_u32_f32(shifted, 1),
                                     vdupq_n_u32(19));
        float32x2_t a = vld1_f32(lut_segments[vgetq_lane_u32(index, 0)]);
        float32x2_t b = vld1_f32(lut_segments[vgetq_lane_u32(index, 1)]);
        float32x2_t c = vld1_f32(lut_segments[vgetq_lane_u32(index, 2)]);
        float32x2_t d = vld1_f32(lut_segments[vgetq_lane_u32(index, 3)]);
        float32x4_t ab = vcombine_f32(a, b);
        float32x4_t cd = vcombine_f32(c, d);
        float32x4_t intercept = vuzp1q_f32(ab, cd);
        float32x4_t slope = vuzp2q_f32(ab, cd);
        float32x4_t result = vfmaq_f32(intercept, x, slope);
        /* One mask selects the interpolation or the outer 0/x branches. */
        uint32x4_t inside = vcltq_f32(vabsq_f32(x), vdupq_n_f32(5.0f));
        float32x4_t outside = vmaxnmq_f32(x, vdupq_n_f32(0.0f));
        result = vbslq_f32(inside, result, outside);
        /* Match the original numeric min/max treatment of NaN. */
        result = vbslq_f32(vceqq_f32(x, x), result, vdupq_n_f32(lut_y[0]));
        vst1q_f32(output + i, result);
    }
#else
    for (; count - i >= 4; i += 4) {
        output[i] = silu_lut(input[i]);
        output[i + 1] = silu_lut(input[i + 1]);
        output[i + 2] = silu_lut(input[i + 2]);
        output[i + 3] = silu_lut(input[i + 3]);
    }
#endif
    for (; i < count; ++i)
        output[i] = silu_lut(input[i]);
}

static const Method methods[] = {
    {"exact_silu", "Exact SiLU", silu_exact, kernel_exact},
    {"taylor3", "Taylor-3 + linear bridge", silu_taylor3, kernel_taylor3},
    {"range_poly", "Range reduction + polynomial", silu_range_poly, kernel_range_poly},
    {"fast_sigmoid", "Fast sigmoid", silu_fast_sigmoid, kernel_fast_sigmoid},
    {"hard_silu", "Hard-SiLU", hard_silu, kernel_hard_silu},
    {"piecewise_linear", "Piecewise linear", silu_piecewise_linear, kernel_piecewise},
    {"lut_21", "LUT (21 values)", silu_lut, kernel_lut},
    {"relu", "ReLU", relu, kernel_relu},
};

extern "C" EXPORT void initialize_activations()
{
    initialize_tables();
}

#define EXPORT_KERNEL(symbol, kernel)                                      \
    extern "C" EXPORT void symbol(const float *input, float *output,       \
                                  size_t count)                            \
    {                                                                      \
        kernel(input, output, count);                                      \
    }

EXPORT_KERNEL(exact_silu_f32, kernel_exact)
EXPORT_KERNEL(taylor3_silu_f32, kernel_taylor3)
EXPORT_KERNEL(range_poly_silu_f32, kernel_range_poly)
EXPORT_KERNEL(fast_sigmoid_silu_f32, kernel_fast_sigmoid)
EXPORT_KERNEL(hard_silu_f32, kernel_hard_silu)
EXPORT_KERNEL(piecewise_linear_silu_f32, kernel_piecewise)
EXPORT_KERNEL(lut21_silu_f32, kernel_lut)
EXPORT_KERNEL(relu_f32, kernel_relu)

#define EXPORT_GRID_LUT(symbol, point_count, half_range)                   \
    extern "C" EXPORT void symbol(const float *input, float *output,       \
                                  size_t count)                            \
    {                                                                      \
        static const GridLUT lut(point_count, -(half_range), half_range);  \
        kernel_configured_grid_lut(input, output, count, lut);             \
    }

EXPORT_GRID_LUT(lut9_range2_silu_f32, 9, 2.0f)
EXPORT_GRID_LUT(lut9_range3_silu_f32, 9, 3.0f)
EXPORT_GRID_LUT(lut9_range4_silu_f32, 9, 4.0f)
EXPORT_GRID_LUT(lut9_range5_silu_f32, 9, 5.0f)
EXPORT_GRID_LUT(lut13_range2_silu_f32, 13, 2.0f)
EXPORT_GRID_LUT(lut13_range3_silu_f32, 13, 3.0f)
EXPORT_GRID_LUT(lut13_range4_silu_f32, 13, 4.0f)
EXPORT_GRID_LUT(lut13_range5_silu_f32, 13, 5.0f)
EXPORT_GRID_LUT(lut17_range2_silu_f32, 17, 2.0f)
EXPORT_GRID_LUT(lut17_range3_silu_f32, 17, 3.0f)
EXPORT_GRID_LUT(lut17_range4_silu_f32, 17, 4.0f)
EXPORT_GRID_LUT(lut17_range5_silu_f32, 17, 5.0f)
EXPORT_GRID_LUT(lut21_range2_silu_f32, 21, 2.0f)
EXPORT_GRID_LUT(lut21_range3_silu_f32, 21, 3.0f)
EXPORT_GRID_LUT(lut21_range4_silu_f32, 21, 4.0f)
EXPORT_GRID_LUT(lut21_range5_silu_f32, 21, 5.0f)

static double monotonic_seconds()
{
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

static Timing benchmark(array_kernel kernel, const float *input, float *output,
                        size_t count, int warmup, int repeats)
{
    for (int i = 0; i < warmup; ++i)
        kernel(input, output, count);

    std::vector<double> samples(static_cast<size_t>(repeats));

    double best_ms = HUGE_VAL;
    for (int i = 0; i < repeats; ++i) {
        double start = monotonic_seconds();
        kernel(input, output, count);
        samples[i] = (monotonic_seconds() - start) * 1000.0;
        if (samples[i] < best_ms)
            best_ms = samples[i];
    }

    float checksum = 0.0f;
    size_t stride = count / 16 + 1;
    for (size_t i = 0; i < count; i += stride)
        checksum += output[i];

    std::sort(samples.begin(), samples.end());
    double median_ms = repeats % 2 == 0
        ? 0.5 * (samples[repeats / 2 - 1] + samples[repeats / 2])
        : samples[repeats / 2];
    Timing result = {
        median_ms,
        best_ms,
        median_ms * 1e6 / (double)count,
        (double)count / (median_ms * 1000.0),
        checksum,
    };
    return result;
}

static uint32_t random_state = 0x12345678u;

static float random_uniform(float minimum, float maximum)
{
    random_state ^= random_state << 13;
    random_state ^= random_state >> 17;
    random_state ^= random_state << 5;
    float unit = (float)(random_state >> 8) * (1.0f / 16777216.0f);
    return minimum + (maximum - minimum) * unit;
}

static FILE *open_output(const char *path)
{
    FILE *file = fopen(path, "w");
    if (file == NULL) {
        fprintf(stderr, "Cannot open %s: %s\n", path, strerror(errno));
        exit(EXIT_FAILURE);
    }
    return file;
}

static void write_values_csv(const char *path, float minimum, float maximum, size_t samples)
{
    FILE *file = open_output(path);
    size_t method_count = sizeof(methods) / sizeof(methods[0]);
    fprintf(file, "x");
    for (size_t method = 0; method < method_count; ++method)
        fprintf(file, ",%s", methods[method].key);
    fputc('\n', file);

    for (size_t i = 0; i < samples; ++i) {
        float x = minimum + (maximum - minimum) * (float)i / (float)(samples - 1);
        fprintf(file, "%.9g", x);
        for (size_t method = 0; method < method_count; ++method)
            fprintf(file, ",%.9g", methods[method].function(x));
        fputc('\n', file);
    }
    fclose(file);
}

static void print_error_statistics(float minimum, float maximum, size_t samples)
{
    size_t method_count = sizeof(methods) / sizeof(methods[0]);
    printf("Error range: [%.3g, %.3g], samples: %zu\n", minimum, maximum, samples);
    printf("%-34s %12s %12s %12s\n", "Method", "MAE", "RMSE", "Max error");
    printf("-------------------------------------------------------------------------\n");

    for (size_t method = 1; method < method_count; ++method) {
        double absolute_sum = 0.0;
        double squared_sum = 0.0;
        double maximum_error = 0.0;
        size_t finite_count = 0;
        for (size_t i = 0; i < samples; ++i) {
            float x = minimum + (maximum - minimum) * (float)i / (float)(samples - 1);
            float approximation = methods[method].function(x);
            if (!isfinite(approximation))
                continue;
            double error = fabs((double)approximation - (double)silu_exact(x));
            absolute_sum += error;
            squared_sum += error * error;
            if (error > maximum_error)
                maximum_error = error;
            ++finite_count;
        }
        printf("%-34s %12.6g %12.6g %12.6g\n",
               methods[method].label,
               absolute_sum / (double)finite_count,
               sqrt(squared_sum / (double)finite_count),
               maximum_error);
    }
}

static void run_lut_grid_search(const char *path, float error_minimum,
                                float error_maximum, size_t samples,
                                const float *input, float *output,
                                size_t element_count, int warmup, int repeats)
{
    static const int point_counts[] = {9, 13, 17, 21};
    static const float half_ranges[] = {2.0f, 3.0f, 4.0f, 5.0f};

    FILE *file = open_output(path);
    fprintf(file,
            "points,minimum,maximum,mae,rmse,max_absolute_error,"
            "median_ms,best_ms,ns_per_element,million_elements_per_second,checksum\n");

    printf("\nUniform LUT grid search: %zu combinations\n",
           sizeof(point_counts) / sizeof(point_counts[0])
               * sizeof(half_ranges) / sizeof(half_ranges[0]));
    printf("%6s %13s %12s %12s %12s %12s\n",
           "Points", "Range", "MAE", "RMSE", "Max error", "Median ms");
    printf("----------------------------------------------------------------------------\n");

    for (int points : point_counts) {
        for (float half_range : half_ranges) {
            GridLUT lut(points, -half_range, half_range);
            double absolute_sum = 0.0;
            double squared_sum = 0.0;
            double maximum_error = 0.0;

            for (size_t i = 0; i < samples; ++i) {
                float x = error_minimum
                    + (error_maximum - error_minimum) * static_cast<float>(i)
                        / static_cast<float>(samples - 1);
                double error = fabs(static_cast<double>(grid_lut_value(x, lut))
                                    - static_cast<double>(silu_exact(x)));
                absolute_sum += error;
                squared_sum += error * error;
                if (error > maximum_error)
                    maximum_error = error;
            }

            double mae = absolute_sum / static_cast<double>(samples);
            double rmse = sqrt(squared_sum / static_cast<double>(samples));
            active_grid_lut = &lut;
            Timing timing = benchmark(kernel_grid_lut, input, output,
                                      element_count, warmup, repeats);

            printf("%6d %6.0f to %-5.0f %12.6g %12.6g %12.6g %12.4f\n",
                   points, -half_range, half_range, mae, rmse,
                   maximum_error, timing.median_ms);
            fprintf(file,
                    "%d,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g\n",
                    points, -half_range, half_range, mae, rmse,
                    maximum_error, timing.median_ms, timing.best_ms,
                    timing.ns_per_element, timing.million_elements_per_second,
                    timing.checksum);
        }
    }

    active_grid_lut = nullptr;
    fclose(file);
    printf("Grid-search CSV: %s\n", path);
}

int main(int argc, char **argv)
{
    const char *values_path = argc > 1 ? argv[1] : "silu_comparison.csv";
    const char *timings_path = argc > 2 ? argv[2] : "silu_timings.csv";
    float minimum = argc > 3 ? strtof(argv[3], NULL) : -8.0f;
    float maximum = argc > 4 ? strtof(argv[4], NULL) : 8.0f;
    size_t samples = argc > 5 ? (size_t)strtoull(argv[5], NULL, 10) : 4001;
    size_t element_count = argc > 6 ? (size_t)strtoull(argv[6], NULL, 10) : 1000000;
    int warmup = argc > 7 ? atoi(argv[7]) : 5;
    int repeats = argc > 8 ? atoi(argv[8]) : 20;
    const char *grid_path = argc > 9 ? argv[9] : "lut_grid_search.csv";

    if (!(maximum > minimum) || samples < 2 || element_count < 1 || warmup < 0 || repeats < 1) {
        fprintf(stderr, "Invalid arguments.\n");
        return EXIT_FAILURE;
    }

    initialize_tables();
    write_values_csv(values_path, minimum, maximum, samples);
    print_error_statistics(minimum, maximum, samples);

    std::vector<float> input(element_count);
    std::vector<float> output(element_count);
    for (size_t i = 0; i < element_count; ++i)
        input[i] = random_uniform(minimum, maximum);

    FILE *timings_file = open_output(timings_path);
    fprintf(timings_file, "key,label,median_ms,best_ms,ns_per_element,million_elements_per_second,checksum\n");
    printf("\nOptimized C++ timing: %zu float elements, %d repeats after %d warm-ups\n",
           element_count, repeats, warmup);
    printf("%-34s %12s %12s %12s %12s\n",
           "Method", "Median ms", "Best ms", "ns/element", "M elem/s");
    printf("---------------------------------------------------------------------------------------\n");

    size_t method_count = sizeof(methods) / sizeof(methods[0]);
    for (size_t method = 0; method < method_count; ++method) {
        Timing timing = benchmark(methods[method].kernel, input.data(), output.data(), element_count, warmup, repeats);
        printf("%-34s %12.4f %12.4f %12.4f %12.2f\n",
               methods[method].label, timing.median_ms, timing.best_ms,
               timing.ns_per_element, timing.million_elements_per_second);
        fprintf(timings_file, "%s,%s,%.9g,%.9g,%.9g,%.9g,%.9g\n",
                methods[method].key, methods[method].label, timing.median_ms,
                timing.best_ms, timing.ns_per_element,
                timing.million_elements_per_second, timing.checksum);
    }

    fclose(timings_file);
    run_lut_grid_search(grid_path, minimum, maximum, samples,
                        input.data(), output.data(), element_count,
                        warmup, repeats);
    printf("\nValues CSV:  %s\nTimings CSV: %s\n", values_path, timings_path);
    return EXIT_SUCCESS;
}
