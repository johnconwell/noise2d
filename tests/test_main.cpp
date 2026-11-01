#include "noise2d.h"
#include "test_helpers.h"
#include <cmath> // std::abs, std::log, std::sqrt
#include <complex> // std::complex
#include <cstdint> // std::uint16_t, std::uint64_t
#include <cstdio> // std::printf
#include <limits> // std::numeric_limits
#include <stdexcept> // std::invalid_argument, std::out_of_range
#include <type_traits> // std::void_t, std::true_type, std::false_type
#include <utility> // std::declval
#include <vector> // std::vector

using noise2d::Noise2D;

static int checks_failed = 0;
static int checks_run = 0;

#define CHECK(expr) \
    do \
    { \
        checks_run++; \
        if(!(expr)) \
        { \
            checks_failed++; \
            std::printf("FAILED %s:%d: %s\n", __FILE__, __LINE__, #expr); \
        } \
    } while(false)

#define CHECK_THROWS(expr, exception) \
    do \
    { \
        checks_run++; \
        bool caught = false; \
        try \
        { \
            (void)(expr); \
        } \
        catch(const exception&) \
        { \
            caught = true; \
        } \
        if(!caught) \
        { \
            checks_failed++; \
            std::printf("FAILED %s:%d: %s did not throw %s\n", __FILE__, __LINE__, #expr, #exception); \
        } \
    } while(false)

// transforms each row, then each column, with a naive DFT
static std::vector<std::complex<double>> naive_dft_2d(const std::vector<double>& input, std::size_t width, std::size_t height)
{
    const double pi = noise2d::detail::PI;
    std::vector<std::complex<double>> rows(width * height);
    std::vector<std::complex<double>> output(width * height);

    for(std::size_t y = 0; y < height; y++)
    {
        for(std::size_t k = 0; k < width; k++)
        {
            for(std::size_t x = 0; x < width; x++)
            {
                rows[y * width + k] += input[y * width + x] * std::polar(1.0, -2.0 * pi * double(k * x) / double(width));
            }
        }
    }

    for(std::size_t x = 0; x < width; x++)
    {
        for(std::size_t k = 0; k < height; k++)
        {
            for(std::size_t y = 0; y < height; y++)
            {
                output[k * width + x] += rows[y * width + x] * std::polar(1.0, -2.0 * pi * double(k * y) / double(height));
            }
        }
    }

    return output;
}

// radially averaged power spectrum of a square matrix, indexed by integer frequency, with the mean removed
static std::vector<double> radial_power(const std::vector<double>& input, std::size_t size)
{
    double mean = 0.0;

    for(double value : input)
    {
        mean += value / double(input.size());
    }

    std::vector<double> centered(input);

    for(double& value : centered)
    {
        value -= mean;
    }

    const std::vector<std::complex<double>> spectrum = naive_dft_2d(centered, size, size);
    std::vector<double> power(size / 2 + 1, 0.0);
    std::vector<double> count(size / 2 + 1, 0.0);

    for(std::size_t y = 0; y < size; y++)
    {
        for(std::size_t x = 0; x < size; x++)
        {
            const double fx = double(std::min(x, size - x));
            const double fy = double(std::min(y, size - y));
            const std::size_t bin = std::size_t(std::sqrt(fx * fx + fy * fy) + 0.5);

            if(bin > 0 && bin < power.size())
            {
                power[bin] += std::norm(spectrum[y * size + x]);
                count[bin] += 1.0;
            }
        }
    }

    for(std::size_t bin = 1; bin < power.size(); bin++)
    {
        power[bin] /= count[bin];
    }

    return power;
}

// least-squares slope of log(power) against log(frequency)
static double log_log_slope(const std::vector<double>& power)
{
    double sum_x = 0.0;
    double sum_y = 0.0;
    double sum_xx = 0.0;
    double sum_xy = 0.0;
    double n = 0.0;

    for(std::size_t bin = 1; bin < power.size(); bin++)
    {
        const double x = std::log(double(bin));
        const double y = std::log(power[bin]);
        sum_x += x;
        sum_y += y;
        sum_xx += x * x;
        sum_xy += x * y;
        n += 1.0;
    }

    return (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
}

template <typename T>
static std::vector<double> as_doubles(const Noise2D<T>& noise)
{
    return std::vector<double>(noise.values().begin(), noise.values().end());
}

static void test_link()
{
    const int a = link_test_a();
    const int b = link_test_b();
    CHECK(a >= 0 && a < 256);
    CHECK(b >= 0 && b < 256);
}

static void test_dimensions()
{
    CHECK_THROWS(Noise2D<int>(0, 8), std::invalid_argument);
    CHECK_THROWS(Noise2D<int>(8, 0), std::invalid_argument);
    CHECK_THROWS(Noise2D<int>(0, 0), std::invalid_argument);
    CHECK_THROWS(Noise2D<int>(std::numeric_limits<std::size_t>::max(), 2), std::invalid_argument);

    Noise2D<int> noise(5, 3, 256, 1);
    CHECK(noise.width() == 5);
    CHECK(noise.height() == 3);
    CHECK(noise.values().size() == 15);
    CHECK(noise.data() == noise.values().data());
    noise.generate_white_noise();
    CHECK(noise(4, 2) == noise.get_noise_at(4, 2));
    CHECK(noise.get_noise_at(4, 2) == noise.data()[2 * 5 + 4]);
    CHECK_THROWS(noise.get_noise_at(5, 0), std::out_of_range);
    CHECK_THROWS(noise.get_noise_at(0, 3), std::out_of_range);
}

static void test_blue_noise_permutation()
{
    const std::size_t sizes[][2] = {{1, 1}, {2, 1}, {1, 2}, {2, 2}, {3, 3}, {1, 7}, {7, 1}, {5, 7}, {16, 8}, {8, 16}, {16, 16}};

    for(const auto& size : sizes)
    {
        const std::size_t num_pixels = size[0] * size[1];

        // with one output level per cell, the integer output is the rank itself
        Noise2D<int> noise(size[0], size[1], std::max<std::size_t>(num_pixels, 2), 7);
        noise.generate_blue_noise();
        CHECK(distinct_values(noise) == num_pixels);

        Noise2D<double> noise_float(size[0], size[1], 256, 7);
        noise_float.generate_blue_noise();
        CHECK(distinct_values(noise_float) == num_pixels);
        CHECK(min_max(noise_float).first == 0.0);

        if(num_pixels > 1)
        {
            CHECK(min_max(noise_float).second == 1.0);
        }
    }

    // the sigma is wide relative to the matrix, so the kernel covers every cell
    Noise2D<int> noise_wide(9, 4, 36, 3);
    noise_wide.generate_blue_noise(20.0);
    CHECK(distinct_values(noise_wide) == 36);

    Noise2D<int> noise(4, 4);
    CHECK_THROWS(noise.generate_blue_noise(0.0), std::invalid_argument);
    CHECK_THROWS(noise.generate_blue_noise(-1.0), std::invalid_argument);
    CHECK_THROWS(noise.generate_blue_noise(std::numeric_limits<double>::quiet_NaN()), std::invalid_argument);
    CHECK_THROWS(noise.generate_blue_noise(std::numeric_limits<double>::infinity()), std::invalid_argument);
}

static void test_blue_noise_repeated_calls()
{
    Noise2D<int> noise(16, 16, 256);
    noise.generate_blue_noise();
    CHECK(distinct_values(noise) == 256);
    noise.generate_blue_noise();
    CHECK(distinct_values(noise) == 256);
}

static void test_blue_noise_spectrum()
{
    Noise2D<double> noise(64, 64, 256, 11);
    noise.generate_blue_noise();
    const std::vector<double> power = radial_power(as_doubles(noise), 64);
    double low = 0.0;
    double high = 0.0;

    for(std::size_t bin = 1; bin <= 8; bin++)
    {
        low += power[bin] / 8.0;
    }
    for(std::size_t bin = 16; bin <= 32; bin++)
    {
        high += power[bin] / 17.0;
    }

    CHECK(low * 5.0 < high);
}

static void test_blue_noise_sparse_ranks()
{
    const std::size_t size = 64;
    const std::size_t num_pixels = size * size;

    for(double sigma : {1.9, 0.8})
    {
        Noise2D<int> noise(size, size, num_pixels, 21);
        noise.generate_blue_noise(sigma);

        // the lowest and highest ranks are too sparse for truncated Gaussians to overlap, but must still be evenly spread rather than following scan order
        for(bool lowest : {true, false})
        {
            std::vector<std::pair<int, int>> points;

            for(std::size_t y = 0; y < size; y++)
            {
                for(std::size_t x = 0; x < size; x++)
                {
                    const std::size_t rank = std::size_t(noise(x, y));

                    if(lowest ? rank < 16 : rank >= num_pixels - 16)
                    {
                        points.push_back({int(x), int(y)});
                    }
                }
            }

            double mean_y = 0.0;
            double nearest_min = 1e9;

            for(const std::pair<int, int>& a : points)
            {
                mean_y += double(a.second) / double(points.size());

                for(const std::pair<int, int>& b : points)
                {
                    if(&a != &b)
                    {
                        const int dx = std::min(std::abs(a.first - b.first), int(size) - std::abs(a.first - b.first));
                        const int dy = std::min(std::abs(a.second - b.second), int(size) - std::abs(a.second - b.second));
                        nearest_min = std::min(nearest_min, std::sqrt(double(dx * dx + dy * dy)));
                    }
                }
            }

            CHECK(points.size() == 16);
            CHECK(nearest_min >= 6.0);
            CHECK(mean_y > 16.0 && mean_y < 48.0);
        }
    }
}

template <typename T>
static void check_range(const Noise2D<T>& noise)
{
    const std::pair<T, T> extremes = min_max(noise);
    CHECK(extremes.first >= T(0));

    if constexpr(std::is_integral_v<T>)
    {
        CHECK(extremes.second <= T(255));
    }
    else
    {
        CHECK(extremes.second <= T(1));
    }
}

template <typename T>
static void test_ranges()
{
    for(std::uint64_t seed = 0; seed < 5; seed++)
    {
        Noise2D<T> noise(33, 17, 256, seed);
        noise.generate_white_noise();
        check_range(noise);
        noise.generate_brown_noise();
        check_range(noise);
        noise.generate_blue_noise();
        check_range(noise);
        noise.generate_perlin_noise(0.1);
        check_range(noise);
        noise.generate_fractal_noise(0.05, 8);
        check_range(noise);
    }

    // brown and blue noise use the full range
    Noise2D<T> noise(32, 32, 256, 3);
    noise.generate_brown_noise();
    CHECK(min_max(noise).first == T(0));
    CHECK(min_max(noise).second == (std::is_integral_v<T> ? T(255) : T(1)));
    noise.generate_blue_noise();
    CHECK(min_max(noise).first == T(0));
    CHECK(min_max(noise).second == (std::is_integral_v<T> ? T(255) : T(1)));
}

static void test_single_pixel()
{
    Noise2D<int> noise(1, 1);
    noise.generate_brown_noise();
    CHECK(noise.get_noise_at(0, 0) == 128);
    noise.generate_blue_noise();
    CHECK(noise.get_noise_at(0, 0) == 0);
    noise.generate_white_noise();
    noise.generate_perlin_noise(0.1);
    noise.generate_fractal_noise(0.1, 2);

    Noise2D<double> noise_float(1, 1);
    noise_float.generate_brown_noise();
    CHECK(noise_float.get_noise_at(0, 0) == 0.5);
    noise_float.generate_blue_noise();
    CHECK(noise_float.get_noise_at(0, 0) == 0.0);
}

static void test_perlin()
{
    // every whole-number frequency produces varied output, including those that double the half-pixel sampling offset onto a whole number
    for(double frequency : {1.0, 2.0, 3.0, 4.0, 8.0})
    {
        Noise2D<int> noise(32, 32, 256, 5);
        noise.generate_perlin_noise(frequency);
        CHECK(distinct_values(noise) > 1);
    }

    // pixel (0, 0) varies across seeds
    std::set<int> first_pixels;

    for(std::uint64_t seed = 0; seed < 16; seed++)
    {
        Noise2D<int> noise(8, 8, 256, seed);
        noise.generate_perlin_noise(0.2);
        first_pixels.insert(noise.get_noise_at(0, 0));
    }

    CHECK(first_pixels.size() > 1);

    Noise2D<double> noise_float(256, 256, 256, 9);
    noise_float.generate_perlin_noise(0.05);
    CHECK(min_max(noise_float).first < 0.25);
    CHECK(min_max(noise_float).second > 0.75);

    // the lattice does not repeat at 256 cells
    Noise2D<int> noise_large(600, 1, 256, 2);
    noise_large.generate_perlin_noise(1.0);
    bool repeats = true;

    for(std::size_t x = 0; x < 256; x++)
    {
        repeats = repeats && noise_large.get_noise_at(x, 0) == noise_large.get_noise_at(x + 256, 0);
    }

    CHECK(!repeats);

    Noise2D<int> noise(4, 4);
    CHECK_THROWS(noise.generate_perlin_noise(0.0), std::invalid_argument);
    CHECK_THROWS(noise.generate_perlin_noise(-1.0), std::invalid_argument);
    CHECK_THROWS(noise.generate_perlin_noise(std::numeric_limits<double>::quiet_NaN()), std::invalid_argument);
    CHECK_THROWS(noise.generate_perlin_noise(std::numeric_limits<double>::infinity()), std::invalid_argument);
    CHECK_THROWS(noise.generate_perlin_noise(1e300), std::invalid_argument);
}

static void test_fractal()
{
    for(std::uint64_t seed = 0; seed < 30; seed++)
    {
        Noise2D<int> noise(64, 64, 256, seed);
        noise.generate_fractal_noise(0.05, 8);
        CHECK(min_max(noise).first >= 0);
        CHECK(min_max(noise).second <= 255);

        Noise2D<double> noise_float(64, 64, 256, seed);
        noise_float.generate_fractal_noise(0.05, 8);
        CHECK(min_max(noise_float).first >= 0.0);
        CHECK(min_max(noise_float).second <= 1.0);
    }

    // octaves at whole-number frequencies contribute to the output
    Noise2D<int> noise_a(32, 32, 65536 / 2, 5);
    Noise2D<int> noise_b(32, 32, 65536 / 2, 5);
    noise_a.generate_fractal_noise(1.0, 1);
    noise_b.generate_fractal_noise(1.0, 2);
    CHECK(!equal(noise_a, noise_b));

    Noise2D<int> noise(4, 4);
    CHECK_THROWS(noise.generate_fractal_noise(0.0, 4), std::invalid_argument);
    CHECK_THROWS(noise.generate_fractal_noise(-1.0, 4), std::invalid_argument);
    CHECK_THROWS(noise.generate_fractal_noise(0.1, 0), std::invalid_argument);
    CHECK_THROWS(noise.generate_fractal_noise(0.1, 5000), std::invalid_argument);
}

static void test_fft()
{
    for(std::size_t size : {1, 2, 3, 4, 5, 7, 8, 12, 17, 64, 100})
    {
        std::vector<std::complex<double>> input(size);
        std::mt19937_64 rng(size);
        std::normal_distribution<double> gaussian(0.0, 1.0);

        for(std::complex<double>& value : input)
        {
            const double real = gaussian(rng);
            const double imaginary = gaussian(rng);
            value = std::complex<double>(real, imaginary);
        }

        std::vector<std::complex<double>> transformed(input);
        noise2d::detail::fft(transformed, false);

        double error_forward = 0.0;

        for(std::size_t k = 0; k < size; k++)
        {
            std::complex<double> expected(0.0, 0.0);

            for(std::size_t n = 0; n < size; n++)
            {
                expected += input[n] * std::polar(1.0, -2.0 * noise2d::detail::PI * double(k * n) / double(size));
            }

            error_forward = std::max(error_forward, std::abs(transformed[k] - expected));
        }

        CHECK(error_forward < 1e-9);

        noise2d::detail::fft(transformed, true);

        double error_inverse = 0.0;

        for(std::size_t n = 0; n < size; n++)
        {
            error_inverse = std::max(error_inverse, std::abs(transformed[n] - input[n]));
        }

        CHECK(error_inverse < 1e-9);
    }
}

static void test_brown_noise_spectrum()
{
    const std::size_t size = 128;
    std::vector<double> power_brown(size / 2 + 1, 0.0);
    std::vector<double> power_white(size / 2 + 1, 0.0);

    // average several seeds to reduce the variance of the estimate
    for(std::uint64_t seed = 0; seed < 8; seed++)
    {
        Noise2D<double> noise(size, size, 256, seed);
        noise.generate_brown_noise();
        const std::vector<double> power_seed_brown = radial_power(as_doubles(noise), size);
        noise.generate_white_noise();
        const std::vector<double> power_seed_white = radial_power(as_doubles(noise), size);

        for(std::size_t bin = 1; bin < power_brown.size(); bin++)
        {
            power_brown[bin] += power_seed_brown[bin];
            power_white[bin] += power_seed_white[bin];
        }
    }

    const double slope_brown = log_log_slope(power_brown);
    const double slope_white = log_log_slope(power_white);
    std::printf("spectral slope: brown %.3f, white %.3f\n", slope_brown, slope_white);
    CHECK(slope_brown > -2.2 && slope_brown < -1.8);
    CHECK(std::abs(slope_white) < 0.2);

    // non-square and non-power-of-two sizes
    Noise2D<int> noise_non_square(33, 10, 256, 4);
    noise_non_square.generate_brown_noise();
    CHECK(min_max(noise_non_square).first == 0);
    CHECK(min_max(noise_non_square).second == 255);
    CHECK(distinct_values(noise_non_square) > 50);
}

template <typename N, typename = void>
struct brown_accepts_double : std::false_type {};
template <typename N>
struct brown_accepts_double<N, std::void_t<decltype(std::declval<N&>().generate_brown_noise(0.999))>> : std::true_type {};

template <typename N, typename = void>
struct perlin_accepts_leading_seed : std::false_type {};
template <typename N>
struct perlin_accepts_leading_seed<N, std::void_t<decltype(std::declval<N&>().generate_perlin_noise(1, 0.1))>> : std::true_type {};

static_assert(!brown_accepts_double<Noise2D<int>>::value, "a leaky integrator must not convert to a seed");
static_assert(!perlin_accepts_leading_seed<Noise2D<int>>::value, "a frequency must not convert to a seed");

template <typename F>
static void check_seeding(F generate)
{
    Noise2D<double> a(24, 16, 256, 42);
    Noise2D<double> b(24, 16, 256, 42);
    Noise2D<double> c(24, 16, 256, 43);
    Noise2D<double> d(24, 16, 256);

    // the constructor seed
    generate(a, std::nullopt);
    generate(b, std::nullopt);
    generate(c, std::nullopt);
    CHECK(equal(a, b));
    CHECK(!equal(a, c));

    // the seed parameter gives the same result whatever the state of the object
    generate(a, 7);
    generate(c, 7);
    generate(d, 7);
    CHECK(equal(a, c));
    CHECK(equal(a, d));
    generate(b, 8);
    CHECK(!equal(a, b));

    // successive calls without a seed continue the sequence
    generate(a, 9);
    const std::vector<double> first = a.values();
    generate(a, std::nullopt);
    CHECK(first != a.values());

    // seed() is equivalent to the seed parameter
    b.seed(9);
    generate(b, std::nullopt);
    CHECK(first == b.values());
}

static void test_seeding()
{
    using Seed = std::optional<std::uint64_t>;
    check_seeding([](Noise2D<double>& noise, Seed seed) { noise.generate_white_noise(seed); });
    check_seeding([](Noise2D<double>& noise, Seed seed) { noise.generate_brown_noise(seed); });
    check_seeding([](Noise2D<double>& noise, Seed seed) { noise.generate_blue_noise(1.9, seed); });
    check_seeding([](Noise2D<double>& noise, Seed seed) { noise.generate_perlin_noise(0.2, seed); });
    check_seeding([](Noise2D<double>& noise, Seed seed) { noise.generate_fractal_noise(0.1, 4, seed); });
}

static void test_type_constraints()
{
    CHECK_THROWS(Noise2D<signed char>(8, 8, 256), std::invalid_argument);
    CHECK_THROWS(Noise2D<unsigned char>(8, 8, 257), std::invalid_argument);
    CHECK_THROWS(Noise2D<int>(8, 8, 0), std::invalid_argument);
    CHECK_THROWS(Noise2D<int>(8, 8, 1), std::invalid_argument);

    Noise2D<signed char> noise(8, 8, 128);
    noise.generate_white_noise();
    CHECK(min_max(noise).first >= 0);
    noise.generate_blue_noise();
    CHECK(min_max(noise).first >= 0);
    CHECK(min_max(noise).second == 127);

    // output_levels is ignored by floating-point types
    Noise2D<float> noise_float(8, 8, 0);
    noise_float.generate_white_noise();
    CHECK(min_max(noise_float).second <= 1.0f);

    // levels above INT_MAX
    Noise2D<std::uint64_t> noise_wide(8, 8, std::numeric_limits<std::size_t>::max());
    noise_wide.generate_white_noise();
    CHECK(min_max(noise_wide).second > std::uint64_t(std::numeric_limits<int>::max()));
    noise_wide.generate_blue_noise();
    CHECK(min_max(noise_wide).second == std::numeric_limits<std::size_t>::max() - 1);
}

int main()
{
    test_link();
    test_dimensions();
    test_blue_noise_permutation();
    test_blue_noise_repeated_calls();
    test_blue_noise_spectrum();
    test_blue_noise_sparse_ranks();
    test_ranges<unsigned char>();
    test_ranges<std::uint16_t>();
    test_ranges<int>();
    test_ranges<float>();
    test_ranges<double>();
    test_single_pixel();
    test_perlin();
    test_fractal();
    test_fft();
    test_brown_noise_spectrum();
    test_seeding();
    test_type_constraints();

    std::printf("%d checks, %d failed\n", checks_run, checks_failed);
    return checks_failed == 0 ? 0 : 1;
}
