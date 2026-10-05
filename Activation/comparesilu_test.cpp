#include <cmath>
#include <cstdio>
#include <limits>

static float lut_y[21];
static float lut_segments[20][2];

static inline float silu_exact(float x)
{
    return x / (1.0f + std::exp(-x));
}

static void initialize_tables()
{
    for (int i = 0; i < 21; ++i)
        lut_y[i] = silu_exact(-5.0f + 0.5f * static_cast<float>(i));

    for (int i = 0; i < 20; ++i) {
        const float slope = 2.0f * (lut_y[i + 1] - lut_y[i]);
        lut_segments[i][1] = slope;
        lut_segments[i][0] =
            std::fma(5.0f - 0.5f * static_cast<float>(i), slope, lut_y[i]);
    }
}

static float silu_lut(float x)
{
    std::printf("Input: %.9g\n", static_cast<double>(x));

    if (x != x) {
        std::printf("Input is NaN: % .9g\n", static_cast<double>(x));
        return lut_y[0];
    }
    if (x <= -5.0f) {
        std::printf("Lower tail: return 0\n");
        return 0.0f;
    }
    if (x >= 4.99f) {
        std::printf("Upper tail: return x\n");
        return x;
    }

    const float position = (x + 5.0f) * 2.0f;
    int index = static_cast<int>(position);
    std::printf("position: %.9g, index before clamp: %d\n",
                static_cast<double>(position), index);

    if (index > 19) {
        std::printf("index > 19: clamp %d to 19\n", index);
        index = 19;
    }

    return x * lut_segments[index][1] + lut_segments[index][0];
}

static void test_value(const char *name, float x)
{
    const float result = silu_lut(x);
    std::printf("%-10s input=% .9g, silu_lut=% .9g, is_nan=%s\n",
                name,
                static_cast<double>(x),
                static_cast<double>(result),
                std::isnan(result) ? "true" : "false");
    std::fflush(stdout);
}

int main()
{
    initialize_tables();

    test_value(
        "below_5",
        std::nextafter(
            5.0f,
            -std::numeric_limits<float>::infinity()
        )
    );
    // test_value("nan", std::numeric_limits<float>::quiet_NaN());

    return 0;
}
