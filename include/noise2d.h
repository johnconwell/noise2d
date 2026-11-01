// MIT License
//
// Copyright (c) 2025 John Conwell
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef NOISE2D_H
#define NOISE2D_H

#include <algorithm> // std::clamp, std::copy, std::count, std::fill, std::max, std::min, std::minmax_element, std::shuffle, std::swap
#include <array> // std::array
#include <cmath> // std::ceil, std::cos, std::exp, std::floor, std::isfinite, std::sin, std::sqrt
#include <complex> // std::complex
#include <cstddef> // std::size_t
#include <cstdint> // std::int64_t, std::uint64_t, std::uintmax_t
#include <limits> // std::numeric_limits
#include <numeric> // std::iota
#include <optional> // std::optional
#include <random> // std::mt19937_64, std::normal_distribution, std::random_device, std::uniform_int_distribution, std::uniform_real_distribution
#include <stdexcept> // std::invalid_argument, std::logic_error, std::out_of_range
#include <type_traits> // std::enable_if_t, std::is_arithmetic_v, std::is_floating_point_v, std::is_integral_v, std::is_same_v
#include <utility> // std::pair
#include <vector> // std::vector

namespace noise2d
{

/**
 * A 2D matrix of noise values with methods that fill it.
 *
 * Integral T: generated values are in [0, output_levels - 1].
 * Floating-point T: generated values are in [0, 1].
 *
 * @note Type T must be an arithmetic type other than bool.
 */
template <typename T>
class Noise2D final
{
    static_assert(std::is_arithmetic_v<T> && !std::is_same_v<T, bool>, "Noise2D requires an integral or floating-point type other than bool");

public:
    /**
     * Do not use default constructor; width and height must be specified.
     */
    Noise2D() = delete;

    /**
     * Constructor for Noise2D objects.
     *
     * @param width The width of the generated noise matrix. Must be greater than 0.
     * @param height The height of the generated noise matrix. Must be greater than 0.
     * @param output_levels The number of output levels in the generated noise matrix.
     * If noise is a floating-point type, this value is ignored.
     * If noise is an integral type, then the values in the generated noise matrix range from [0, output_levels - 1].
     * @param seed The seed for the pseudo-random number generator, specified to allow repeatable results.
     * If not specified, the generator is seeded from std::random_device.
     * @throw std::invalid_argument if width or height is 0 or width * height overflows.
     * @throw std::invalid_argument if noise is an integral type and output_levels is less than 2 or output_levels - 1 does not fit in T.
     */
    Noise2D(std::size_t width, std::size_t height, std::size_t output_levels = 256, std::optional<std::uint64_t> seed = std::nullopt);

    /**
     * Method to reseed the pseudo-random number generator.
     *
     * @param value The new seed.
     */
    void seed(std::uint64_t value);

    /**
     * Method to get the noise from the noise matrix at an (x, y) coordinate.
     *
     * @param x The x coordinate of the desired noise data.
     * @param y The y coordinate of the desired noise data.
     * @return The desired noise data.
     * @throw std::out_of_range if x >= width or y >= height.
     */
    [[nodiscard]] T get_noise_at(std::size_t x, std::size_t y) const;

    /**
     * Method to get the noise from the noise matrix at an (x, y) coordinate without bounds checking.
     *
     * @param x The x coordinate of the desired noise data. Must be less than width.
     * @param y The y coordinate of the desired noise data. Must be less than height.
     * @return The desired noise data.
     */
    [[nodiscard]] T operator()(std::size_t x, std::size_t y) const noexcept;

    /**
     * Method to get a pointer to the noise matrix, stored contiguously in row-major order (index = y * width + x).
     *
     * @return A pointer to the first element of the noise matrix.
     */
    [[nodiscard]] const T* data() const noexcept;

    /**
     * Method to get the noise matrix, stored contiguously in row-major order (index = y * width + x).
     *
     * @return A reference to the noise matrix.
     */
    [[nodiscard]] const std::vector<T>& values() const noexcept;

    /**
     * Method to get the width of the noise matrix.
     *
     * @return The width of the noise matrix.
     */
    [[nodiscard]] std::size_t width() const noexcept;

    /**
     * Method to get the height of the noise matrix.
     *
     * @return The height of the noise matrix.
     */
    [[nodiscard]] std::size_t height() const noexcept;

    /**
     * Method to generate blue noise using Ulichney's Void and Cluster algorithm.
     *
     * @param sigma The standard deviation of the Gaussian function used to generate blue noise. Must be greater than 0.
     * Low values of sigma result in ordered patterns, while high values of sigma result in stronger low-frequency components.
     * If not specified, a default value of 1.9 is used.
     * @param seed If specified, reseeds the object before generating, specified to allow repeatable results.
     * @throw std::invalid_argument if sigma is not a positive finite value.
     */
    void generate_blue_noise(double sigma = 1.9, std::optional<std::uint64_t> seed = std::nullopt);

    /**
     * Method to generate brown noise by shaping Gaussian white noise in the frequency domain to a 1/f^2 power spectrum.
     * The result tiles seamlessly.
     *
     * @param seed If specified, reseeds the object before generating, specified to allow repeatable results.
     */
    void generate_brown_noise(std::optional<std::uint64_t> seed = std::nullopt);

    /**
     * Method to generate white noise.
     *
     * @param seed If specified, reseeds the object before generating, specified to allow repeatable results.
     */
    void generate_white_noise(std::optional<std::uint64_t> seed = std::nullopt);

    /**
     * Method to generate Perlin Noise.
     *
     * @param frequency The frequency of noise in cycles per pixel. Must be greater than 0. Values of 0.5 or greater alias (Nyquist) and increasingly resemble white noise.
     * @param seed If specified, reseeds the object before generating, specified to allow repeatable results.
     * @throw std::invalid_argument if frequency is not a positive finite value or is too large to sample accurately.
     */
    void generate_perlin_noise(double frequency, std::optional<std::uint64_t> seed = std::nullopt);

    /**
     * Method to generate Fractal Noise ("stacked Perlin noise").
     *
     * @param starting_frequency The beginning frequency for the Perlin noise in cycles per pixel, which will contribute the bulk of the noise value at each pixel.
     * @param octaves The number of octaves in the noise. Each octave contributes more noise at twice the frequency and half the amplitude.
     * @param seed If specified, reseeds the object before generating, specified to allow repeatable results.
     * @throw std::invalid_argument if starting_frequency is not a positive finite value or the highest octave's frequency is too large to sample accurately.
     * @throw std::invalid_argument if octaves is less than 1.
     */
    void generate_fractal_noise(double starting_frequency, std::size_t octaves, std::optional<std::uint64_t> seed = std::nullopt);

    // These overloads reject calls written for the parameter lists that took a leaky integrator or a leading seed,
    // which would otherwise convert a double to the optional seed and compile with a different meaning.
    template <typename S, typename = std::enable_if_t<std::is_floating_point_v<S>>>
    void generate_brown_noise(S) = delete;
    template <typename S, typename = std::enable_if_t<std::is_floating_point_v<S>>>
    void generate_perlin_noise(double, S) = delete;

private:
    void apply_seed(const std::optional<std::uint64_t>& value); // reseeds the generator if a seed was specified
    [[nodiscard]] T quantize(double value) const; // maps a value in [0, 1] to T, clamping values outside that range
    void validate_frequency(double frequency, std::size_t octaves) const; // throws if the frequency of any octave cannot be sampled

    std::vector<T> data_; // matrix the output of noise generation, stored in row-major order
    std::size_t width_; // width of the noise matrix
    std::size_t height_; // height of the noise matrix
    std::size_t output_levels_; // number of output levels of the noise matrix, not used if matrix type is floating-point
    std::mt19937_64 rng_; // pseudo-random number generator shared by every generation method
};

namespace detail
{

inline constexpr double PI = 3.14159265358979323846;
inline constexpr double PERLIN_AMPLITUDE_MAX = 0.70710678118654752440; // largest magnitude of 2D Perlin noise with unit gradients
inline constexpr double COVERAGE = 0.1; // fraction of cells that are ones in the blue noise initial binary pattern
inline constexpr double LATTICE_COORDINATE_MAX = 4503599627370496.0; // 2^52, beyond which a double has no fractional bits left to sample between lattice points

/**
 * Method to perform quintic (5th order) interpolation between two values.
 *
 * @param v0 The first value.
 * @param v1 The second value.
 * @param t The weight of interpolation between v0 and v1, in the range [0, 1].
 * @return An interpolated value between v0 and v1.
 */
inline double interpolate_quintic(double v0, double v1, double t)
{
    return v0 + (t * t * t * (t * (t * 6.0 - 15.0) + 10.0)) * (v1 - v0);
}

/**
 * Method to scramble the bits of a value (the finalizer of SplitMix64).
 *
 * @param value The value to scramble.
 * @return The scrambled value.
 */
inline std::uint64_t mix_bits(std::uint64_t value)
{
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

/**
 * Method to get the unit vector assigned to a lattice point.
 *
 * @param seed The seed of the lattice.
 * @param x The x coordinate of the lattice point.
 * @param y The y coordinate of the lattice point.
 * @return A unit vector pointing in a pseudo-random 2d direction that is repeatable for the same arguments.
 */
inline const std::pair<double, double>& lattice_gradient(std::uint64_t seed, std::int64_t x, std::int64_t y)
{
    static constexpr std::size_t NUM_GRADIENTS = 1024;
    static constexpr std::uint64_t GOLDEN_RATIO = 0x9e3779b97f4a7c15ULL;
    static const std::array<std::pair<double, double>, NUM_GRADIENTS> gradients = []()
    {
        std::array<std::pair<double, double>, NUM_GRADIENTS> table;

        for(std::size_t index = 0; index < NUM_GRADIENTS; index++)
        {
            const double angle = 2.0 * PI * static_cast<double>(index) / static_cast<double>(NUM_GRADIENTS);
            table[index] = std::pair<double, double>(std::sin(angle), std::cos(angle));
        }

        return table;
    }();

    const std::uint64_t hash_x = mix_bits(seed + static_cast<std::uint64_t>(x) * GOLDEN_RATIO);
    const std::uint64_t hash = mix_bits(hash_x + static_cast<std::uint64_t>(y) * GOLDEN_RATIO);
    return gradients[hash & (NUM_GRADIENTS - 1)];
}

/**
 * Method to sample Perlin noise at a point.
 *
 * @param seed The seed of the lattice.
 * @param x The x coordinate of the sample.
 * @param y The y coordinate of the sample.
 * @return The noise value at the sample, in the range [-PERLIN_AMPLITUDE_MAX, PERLIN_AMPLITUDE_MAX].
 */
inline double sample_perlin_point(std::uint64_t seed, double x, double y)
{
    // find the top left of the four surrounding lattice points
    const double x_floor = std::floor(x);
    const double y_floor = std::floor(y);
    const std::int64_t x0 = static_cast<std::int64_t>(x_floor);
    const std::int64_t y0 = static_cast<std::int64_t>(y_floor);

    // determine the x and y offset of the sample coordinate from the top left corner
    const double dx = x - x_floor;
    const double dy = y - y_floor;

    // calculate the dot product between the gradient at each surrounding lattice point and the vector from that point to the sample
    const auto dot_product = [seed](std::int64_t lattice_x, std::int64_t lattice_y, double offset_x, double offset_y)
    {
        const std::pair<double, double>& gradient = lattice_gradient(seed, lattice_x, lattice_y);
        return gradient.first * offset_x + gradient.second * offset_y;
    };

    const double dot_product_00 = dot_product(x0, y0, dx, dy);
    const double dot_product_01 = dot_product(x0, y0 + 1, dx, dy - 1.0);
    const double dot_product_10 = dot_product(x0 + 1, y0, dx - 1.0, dy);
    const double dot_product_11 = dot_product(x0 + 1, y0 + 1, dx - 1.0, dy - 1.0);

    // interpolate between the horizontal components
    const double interpolation_top = interpolate_quintic(dot_product_00, dot_product_10, dx);
    const double interpolation_bottom = interpolate_quintic(dot_product_01, dot_product_11, dx);

    // interpolate between the interpolated components
    return interpolate_quintic(interpolation_top, interpolation_bottom, dy);
}

/**
 * Method to generate an offset added to Perlin sample coordinates.
 * The fractional part is kept away from 0 so that samples do not land on lattice points, where the noise is always 0, at whole-number frequencies.
 *
 * @param rng The pseudo-random number generator.
 * @return The offset, in the range [0, 256).
 */
inline double lattice_offset(std::mt19937_64& rng)
{
    std::uniform_int_distribution<int> whole(0, 255);
    std::uniform_real_distribution<double> fraction(0.25, 0.75);
    const int offset_whole = whole(rng);
    return static_cast<double>(offset_whole) + fraction(rng);
}

/**
 * Method to compute the discrete Fourier transform of a sequence of any length in place.
 * Lengths that are not a power of two are computed with Bluestein's algorithm.
 *
 * @param values The sequence to transform.
 * @param inverse If true, computes the inverse transform, which is divided by the length of the sequence.
 */
inline void fft(std::vector<std::complex<double>>& values, bool inverse)
{
    const std::size_t size = values.size();
    const double direction = inverse ? 1.0 : -1.0;

    if(size <= 1)
    {
        return;
    }

    if((size & (size - 1)) == 0)
    {
        // reorder the sequence so that each element is at its bit-reversed index
        for(std::size_t index = 1, reversed = 0; index < size; index++)
        {
            std::size_t bit = size >> 1;

            for(; reversed & bit; bit >>= 1)
            {
                reversed ^= bit;
            }

            reversed ^= bit;

            if(index < reversed)
            {
                std::swap(values[index], values[reversed]);
            }
        }

        std::vector<std::complex<double>> twiddles(size / 2);

        for(std::size_t index = 0; index < size / 2; index++)
        {
            const double angle = direction * 2.0 * PI * static_cast<double>(index) / static_cast<double>(size);
            twiddles[index] = std::complex<double>(std::cos(angle), std::sin(angle));
        }

        // combine transforms of length/2 into transforms of length
        for(std::size_t length = 2; length <= size; length <<= 1)
        {
            const std::size_t twiddle_step = size / length;

            for(std::size_t start = 0; start < size; start += length)
            {
                for(std::size_t index = 0; index < length / 2; index++)
                {
                    const std::complex<double> even = values[start + index];
                    const std::complex<double> odd = values[start + index + length / 2] * twiddles[index * twiddle_step];
                    values[start + index] = even + odd;
                    values[start + index + length / 2] = even - odd;
                }
            }
        }
    }
    else
    {
        std::size_t size_padded = 1;

        while(size_padded < 2 * size - 1)
        {
            size_padded <<= 1;
        }

        // w[k] = exp(direction * i * pi * k^2 / size), with k^2 reduced modulo 2 * size to keep the angle accurate
        std::vector<std::complex<double>> chirp(size);

        for(std::size_t index = 0; index < size; index++)
        {
            const std::size_t index_squared = (index * index) % (2 * size);
            const double angle = direction * PI * static_cast<double>(index_squared) / static_cast<double>(size);
            chirp[index] = std::complex<double>(std::cos(angle), std::sin(angle));
        }

        std::vector<std::complex<double>> sequence(size_padded, std::complex<double>(0.0, 0.0));
        std::vector<std::complex<double>> filter(size_padded, std::complex<double>(0.0, 0.0));

        for(std::size_t index = 0; index < size; index++)
        {
            sequence[index] = values[index] * chirp[index];
        }

        filter[0] = std::conj(chirp[0]);

        for(std::size_t index = 1; index < size; index++)
        {
            filter[index] = std::conj(chirp[index]);
            filter[size_padded - index] = std::conj(chirp[index]);
        }

        // circular convolution of the chirped sequence with the filter
        fft(sequence, false);
        fft(filter, false);

        for(std::size_t index = 0; index < size_padded; index++)
        {
            sequence[index] *= filter[index];
        }

        fft(sequence, true);

        for(std::size_t index = 0; index < size; index++)
        {
            values[index] = sequence[index] * chirp[index];
        }
    }

    if(inverse)
    {
        for(std::size_t index = 0; index < size; index++)
        {
            values[index] /= static_cast<double>(size);
        }
    }

    return;
}

/**
 * Method to generate a real field with a 1/f^2 power spectrum, which has an amplitude spectrum of 1/f.
 * The field has a mean of zero and is periodic in both axes.
 *
 * @param width The width of the field.
 * @param height The height of the field.
 * @param rng The pseudo-random number generator.
 * @return The field, stored in row-major order.
 */
inline std::vector<double> generate_brown_field(std::size_t width, std::size_t height, std::mt19937_64& rng)
{
    std::vector<std::complex<double>> spectrum(width * height);
    std::vector<std::complex<double>> line;
    std::normal_distribution<double> gaussian(0.0, 1.0);

    // signed frequency of an index along an axis, in cycles per pixel
    const auto frequency = [](std::size_t index, std::size_t size)
    {
        return static_cast<double>(std::min(index, size - index)) / static_cast<double>(size);
    };

    for(std::size_t y = 0; y < height; y++)
    {
        const double frequency_y = frequency(y, height);

        for(std::size_t x = 0; x < width; x++)
        {
            const double frequency_x = frequency(x, width);
            const double frequency_magnitude = std::sqrt(frequency_x * frequency_x + frequency_y * frequency_y);
            const double real = gaussian(rng);
            const double imaginary = gaussian(rng);

            // the DC component stays zero since 1/f is undefined at f = 0
            if(frequency_magnitude > 0.0)
            {
                spectrum[y * width + x] = std::complex<double>(real, imaginary) / frequency_magnitude;
            }
        }
    }

    // inverse transform along each row, then along each column
    line.resize(width);

    for(std::size_t y = 0; y < height; y++)
    {
        std::copy(spectrum.begin() + static_cast<std::ptrdiff_t>(y * width), spectrum.begin() + static_cast<std::ptrdiff_t>((y + 1) * width), line.begin());
        fft(line, true);
        std::copy(line.begin(), line.end(), spectrum.begin() + static_cast<std::ptrdiff_t>(y * width));
    }

    line.resize(height);

    for(std::size_t x = 0; x < width; x++)
    {
        for(std::size_t y = 0; y < height; y++)
        {
            line[y] = spectrum[y * width + x];
        }

        fft(line, true);

        for(std::size_t y = 0; y < height; y++)
        {
            spectrum[y * width + x] = line[y];
        }
    }

    // the real part of the result is a real field with the same power spectrum
    std::vector<double> field(width * height);

    for(std::size_t index = 0; index < field.size(); index++)
    {
        field[index] = spectrum[index].real();
    }

    return field;
}

/**
 * A class that ranks every cell in a matrix using Ulichney's Void and Cluster algorithm.
 * See https://cv.ulichney.com/papers/1993-void-cluster.pdf for information on each algorithm step.
 *
 * The Gaussian "energy" of each cell is maintained in an LUT. The Gaussian is truncated at 3 sigma so that changing one cell only touches the cells around it.
 * While the ones in the binary pattern are too sparse for the truncated Gaussians to overlap, the full Gaussian is used instead, since cells that cannot see each other would all tie on energy.
 */
class VoidAndCluster
{
public:
    /**
     * Do not use default constructor; width and height must be specified.
     */
    VoidAndCluster() = delete;

    /**
     * Constructor for VoidAndCluster objects.
     *
     * @param width The width of the matrix.
     * @param height The height of the matrix.
     * @param sigma The standard deviation of the Gaussian function used to calculate the energy at each cell.
     */
    VoidAndCluster(std::size_t width, std::size_t height, double sigma) : width(width), height(height), sigma(sigma), pattern(width * height, 0), energy(width * height, 0.0)
    {
        kernel_truncated = build_kernel(std::ceil(3.0 * sigma));
        return;
    }

    /**
     * Method to rank every cell in the matrix.
     *
     * @param rng The pseudo-random number generator.
     * @return The rank of each cell, stored in row-major order. The ranks are a permutation of [0, width * height - 1].
     */
    std::vector<std::size_t> generate_ranks(std::mt19937_64& rng)
    {
        const std::size_t num_pixels = width * height;
        const std::size_t num_minority_pixels = std::max<std::size_t>(1, static_cast<std::size_t>(static_cast<double>(num_pixels) * COVERAGE)); // must have at least 1 minority pixel
        std::vector<std::size_t> ranks(num_pixels, 0);

        const std::vector<unsigned char> pattern_initial = generate_initial_binary_pattern(num_minority_pixels, rng);

        // phase 1: fill in ranks from one less than the number of ones in the initial binary pattern down to zero
        set_pattern(pattern_initial);

        for(std::size_t rank = num_minority_pixels; rank > 0; rank--)
        {
            const std::size_t index = find_tightest_cluster();
            toggle(index);
            ranks[index] = rank - 1;
            select_kernel();
        }

        // phase 2: fill in ranks from the number of ones in the initial binary pattern up to half the number of cells
        set_pattern(pattern_initial);

        for(std::size_t rank = num_minority_pixels; rank < num_pixels / 2; rank++)
        {
            const std::size_t index = find_largest_void();
            toggle(index);
            ranks[index] = rank;
            select_kernel();
        }

        // phase 3: invert the pattern so that zeros are the minority and fill in the remaining ranks
        std::vector<unsigned char> pattern_inverted = pattern;

        for(unsigned char& cell : pattern_inverted)
        {
            cell = cell ? 0 : 1;
        }

        set_pattern(pattern_inverted);

        for(std::size_t rank = std::max(num_minority_pixels, num_pixels / 2); rank < num_pixels; rank++)
        {
            const std::size_t index = find_tightest_cluster();
            toggle(index);
            ranks[index] = rank;
            select_kernel();
        }

        return ranks;
    }

private:
    struct KernelEntry
    {
        std::size_t offset_x;
        std::size_t offset_y;
        double weight;
    };

    // the wrapped Gaussian weights within a radius, by offset from the center cell
    [[nodiscard]] std::vector<KernelEntry> build_kernel(double radius) const
    {
        const double two_sigma_squared = 2.0 * sigma * sigma;
        std::vector<KernelEntry> kernel;

        // the kernel wraps around the edges of the matrix, so each offset is measured by its toroidal distance
        for(std::size_t offset_y = 0; offset_y < height; offset_y++)
        {
            const double distance_y = static_cast<double>(std::min(offset_y, height - offset_y));

            if(distance_y > radius)
            {
                continue;
            }

            for(std::size_t offset_x = 0; offset_x < width; offset_x++)
            {
                const double distance_x = static_cast<double>(std::min(offset_x, width - offset_x));

                if(distance_x <= radius)
                {
                    kernel.push_back({offset_x, offset_y, std::exp(-(distance_x * distance_x + distance_y * distance_y) / two_sigma_squared)});
                }
            }
        }

        return kernel;
    }

    // true if the truncated kernels of this number of ones are too sparse to overlap reliably
    [[nodiscard]] bool pattern_is_sparse(std::size_t num_ones_pattern) const
    {
        return num_ones_pattern * kernel_truncated.size() < SPARSE_OVERLAP * pattern.size();
    }

    // switches to the kernel suited to the current number of ones, recalculating the energy of every cell if that changes the kernel
    void select_kernel()
    {
        if(pattern_is_sparse(num_ones) != use_full_kernel)
        {
            set_pattern(pattern);
        }

        return;
    }

    // generates the initial binary pattern (matrix of zeros and ones, with the ones as evenly spaced as possible)
    std::vector<unsigned char> generate_initial_binary_pattern(std::size_t num_minority_pixels, std::mt19937_64& rng)
    {
        const std::size_t num_pixels = width * height;
        std::vector<std::size_t> indices(num_pixels);
        std::vector<unsigned char> pattern_random(num_pixels, 0);

        std::iota(indices.begin(), indices.end(), std::size_t(0));
        std::shuffle(indices.begin(), indices.end(), rng);

        for(std::size_t index = 0; index < num_minority_pixels; index++)
        {
            pattern_random[indices[index]] = 1;
        }

        set_pattern(pattern_random);

        // limiting iterations ensures that this will not get stuck forever if the swaps enter a cycle
        for(std::size_t count = 0; count < num_pixels; count++)
        {
            // find the tightest cluster and remove it
            const std::size_t index_tightest_cluster = find_tightest_cluster();
            toggle(index_tightest_cluster);

            // find the largest void and fill it
            const std::size_t index_largest_void = find_largest_void();
            toggle(index_largest_void);

            // exit condition when removing the tightest cluster creates the largest void
            if(index_tightest_cluster == index_largest_void)
            {
                break;
            }
        }

        return pattern;
    }

    // replaces the binary pattern and recalculates the energy of every cell
    void set_pattern(std::vector<unsigned char> pattern_new)
    {
        use_full_kernel = pattern_is_sparse(static_cast<std::size_t>(std::count(pattern_new.begin(), pattern_new.end(), static_cast<unsigned char>(1))));

        if(use_full_kernel && kernel_full.empty())
        {
            kernel_full = build_kernel(std::numeric_limits<double>::infinity());
        }

        num_ones = 0;
        std::fill(pattern.begin(), pattern.end(), static_cast<unsigned char>(0));
        std::fill(energy.begin(), energy.end(), 0.0);

        for(std::size_t index = 0; index < pattern_new.size(); index++)
        {
            if(pattern_new[index])
            {
                toggle(index);
            }
        }

        return;
    }

    // changes a cell from a zero to a one or vice versa and updates the energy of the surrounding cells
    void toggle(std::size_t index)
    {
        const std::size_t x = index % width;
        const std::size_t y = index / width;
        const double sign = pattern[index] ? -1.0 : 1.0;

        num_ones = pattern[index] ? num_ones - 1 : num_ones + 1;
        pattern[index] = pattern[index] ? 0 : 1;

        for(const KernelEntry& entry : (use_full_kernel ? kernel_full : kernel_truncated))
        {
            const std::size_t i = x + entry.offset_x >= width ? x + entry.offset_x - width : x + entry.offset_x;
            const std::size_t j = y + entry.offset_y >= height ? y + entry.offset_y - height : y + entry.offset_y;
            energy[j * width + i] += sign * entry.weight;
        }

        return;
    }

    // only ones in the binary pattern can be considered the tightest cluster
    [[nodiscard]] std::size_t find_tightest_cluster() const
    {
        double value_highest_energy = std::numeric_limits<double>::lowest();
        std::size_t index_highest_energy = pattern.size();

        for(std::size_t index = 0; index < pattern.size(); index++)
        {
            if(pattern[index] == 1 && energy[index] > value_highest_energy)
            {
                value_highest_energy = energy[index];
                index_highest_energy = index;
            }
        }

        if(index_highest_energy == pattern.size())
        {
            throw std::logic_error("VoidAndCluster binary pattern has no ones");
        }

        return index_highest_energy;
    }

    // only zeros in the binary pattern can be considered the largest void
    [[nodiscard]] std::size_t find_largest_void() const
    {
        double value_lowest_energy = std::numeric_limits<double>::max();
        std::size_t index_lowest_energy = pattern.size();

        for(std::size_t index = 0; index < pattern.size(); index++)
        {
            if(pattern[index] == 0 && energy[index] < value_lowest_energy)
            {
                value_lowest_energy = energy[index];
                index_lowest_energy = index;
            }
        }

        if(index_lowest_energy == pattern.size())
        {
            throw std::logic_error("VoidAndCluster binary pattern has no zeros");
        }

        return index_lowest_energy;
    }

    static constexpr std::size_t SPARSE_OVERLAP = 4; // average number of truncated kernels covering each cell below which the pattern is considered sparse

    std::size_t width; // the width of the matrix
    std::size_t height; // the height of the matrix
    double sigma; // the standard deviation of the Gaussian function
    std::vector<unsigned char> pattern; // the binary pattern, stored in row-major order
    std::size_t num_ones = 0; // the number of ones in the binary pattern
    std::vector<double> energy; // the LUT that stores the Gaussian energy of each cell, stored in row-major order
    std::vector<KernelEntry> kernel_truncated; // the weights of the Gaussian within 3 sigma, by offset from the center cell
    std::vector<KernelEntry> kernel_full; // the weights of the Gaussian at every offset, built when first needed
    bool use_full_kernel = false; // whether the energy LUT is maintained with the full Gaussian
};

} // namespace detail

template <typename T>
Noise2D<T>::Noise2D(std::size_t width, std::size_t height, std::size_t output_levels, std::optional<std::uint64_t> seed)
{
    if(width == 0 || height == 0)
    {
        throw std::invalid_argument("Noise2D invalid dimensions, width and height must be greater than 0");
    }
    if(height > std::numeric_limits<std::size_t>::max() / width)
    {
        throw std::invalid_argument("Noise2D invalid dimensions, width * height is too large");
    }

    if constexpr(std::is_integral_v<T>)
    {
        if(output_levels < 2)
        {
            throw std::invalid_argument("Noise2D invalid output_levels, must be greater than or equal to 2");
        }
        if(static_cast<std::uintmax_t>(output_levels - 1) > static_cast<std::uintmax_t>(std::numeric_limits<T>::max()))
        {
            throw std::invalid_argument("Noise2D invalid output_levels, output_levels - 1 must fit in the noise type");
        }
    }

    width_ = width;
    height_ = height;
    output_levels_ = output_levels;
    data_ = std::vector<T>(width * height, static_cast<T>(0));

    if(seed)
    {
        rng_.seed(*seed);
    }
    else
    {
        std::random_device rd;
        const std::uint64_t seed_high = rd();
        const std::uint64_t seed_low = rd();
        rng_.seed((seed_high << 32) | seed_low);
    }

    return;
}

template <typename T>
void Noise2D<T>::seed(std::uint64_t value)
{
    rng_.seed(value);
    return;
}

template <typename T>
T Noise2D<T>::get_noise_at(std::size_t x, std::size_t y) const
{
    if(x >= width_ || y >= height_)
    {
        throw std::out_of_range("get_noise_at argument(s) too large");
    }

    return data_[y * width_ + x];
}

template <typename T>
T Noise2D<T>::operator()(std::size_t x, std::size_t y) const noexcept
{
    return data_[y * width_ + x];
}

template <typename T>
const T* Noise2D<T>::data() const noexcept
{
    return data_.data();
}

template <typename T>
const std::vector<T>& Noise2D<T>::values() const noexcept
{
    return data_;
}

template <typename T>
std::size_t Noise2D<T>::width() const noexcept
{
    return width_;
}

template <typename T>
std::size_t Noise2D<T>::height() const noexcept
{
    return height_;
}

template <typename T>
void Noise2D<T>::generate_blue_noise(double sigma, std::optional<std::uint64_t> seed)
{
    const double two_sigma_squared = 2.0 * sigma * sigma;

    if(!(sigma > 0.0) || !(two_sigma_squared > 0.0) || !std::isfinite(two_sigma_squared))
    {
        throw std::invalid_argument("generate_blue_noise invalid sigma, must be a positive finite value");
    }

    apply_seed(seed);

    detail::VoidAndCluster void_and_cluster(width_, height_, sigma);
    const std::vector<std::size_t> ranks = void_and_cluster.generate_ranks(rng_);

    // the max value in ranks is one less than the number of cells, so this normalizes to the 0-1 range
    const double rank_max = static_cast<double>(std::max<std::size_t>(ranks.size() - 1, 1));

    for(std::size_t index = 0; index < ranks.size(); index++)
    {
        data_[index] = quantize(static_cast<double>(ranks[index]) / rank_max);
    }

    return;
}

template <typename T>
void Noise2D<T>::generate_brown_noise(std::optional<std::uint64_t> seed)
{
    apply_seed(seed);

    const std::vector<double> field = detail::generate_brown_field(width_, height_, rng_);
    const std::pair<std::vector<double>::const_iterator, std::vector<double>::const_iterator> field_extremes = std::minmax_element(field.begin(), field.end());
    const double field_min = *field_extremes.first;
    const double field_range = *field_extremes.second - field_min;

    // normalize to the 0-1 range, a field with no range (e.g. 1x1) has only its mid-value
    for(std::size_t index = 0; index < field.size(); index++)
    {
        data_[index] = quantize(field_range > 0.0 ? (field[index] - field_min) / field_range : 0.5);
    }

    return;
}

template <typename T>
void Noise2D<T>::generate_white_noise(std::optional<std::uint64_t> seed)
{
    apply_seed(seed);

    if constexpr(std::is_integral_v<T>)
    {
        std::uniform_int_distribution<std::uint64_t> dis(0, static_cast<std::uint64_t>(output_levels_ - 1));

        for(T& value : data_)
        {
            value = static_cast<T>(dis(rng_));
        }
    }
    else
    {
        std::uniform_real_distribution<T> dis(static_cast<T>(0), static_cast<T>(1));

        for(T& value : data_)
        {
            value = dis(rng_);
        }
    }

    return;
}

template <typename T>
void Noise2D<T>::generate_perlin_noise(double frequency, std::optional<std::uint64_t> seed)
{
    generate_fractal_noise(frequency, 1, seed);
    return;
}

template <typename T>
void Noise2D<T>::generate_fractal_noise(double starting_frequency, std::size_t octaves, std::optional<std::uint64_t> seed)
{
    if(octaves < 1)
    {
        throw std::invalid_argument("generate_fractal_noise invalid number of octaves, must be greater than or equal to 1");
    }

    validate_frequency(starting_frequency, octaves);
    apply_seed(seed);

    struct Octave
    {
        double frequency;
        double amplitude;
        double offset_x;
        double offset_y;
        std::uint64_t lattice_seed;
    };

    // each octave has its own lattice and offset so that octaves do not share lattice alignment
    std::vector<Octave> octave_list(octaves);
    double frequency = starting_frequency;
    double amplitude = 1.0;
    double amplitude_total = 0.0;

    for(Octave& octave : octave_list)
    {
        octave.frequency = frequency;
        octave.amplitude = amplitude;
        octave.offset_x = detail::lattice_offset(rng_);
        octave.offset_y = detail::lattice_offset(rng_);
        octave.lattice_seed = rng_();
        amplitude_total += amplitude;
        frequency *= 2.0;
        amplitude /= 2.0;
    }

    for(std::size_t y = 0; y < height_; y++)
    {
        for(std::size_t x = 0; x < width_; x++)
        {
            double noise = 0.0;

            for(const Octave& octave : octave_list)
            {
                // convert each pixel coordinate to a floating point sample coordinate scaled to the desired frequency
                const double xf = (static_cast<double>(x) + 0.5) * octave.frequency + octave.offset_x;
                const double yf = (static_cast<double>(y) + 0.5) * octave.frequency + octave.offset_y;

                noise += octave.amplitude * detail::sample_perlin_point(octave.lattice_seed, xf, yf);
            }

            // scale to the [-1, 1] range, then normalize to the 0-1 range
            const double noise_scaled = noise / (amplitude_total * detail::PERLIN_AMPLITUDE_MAX);
            data_[y * width_ + x] = quantize((noise_scaled + 1.0) / 2.0);
        }
    }

    return;
}

template <typename T>
void Noise2D<T>::apply_seed(const std::optional<std::uint64_t>& value)
{
    if(value)
    {
        rng_.seed(*value);
    }

    return;
}

template <typename T>
T Noise2D<T>::quantize(double value) const
{
    const double value_clamped = std::clamp(value, 0.0, 1.0);

    if constexpr(std::is_integral_v<T>)
    {
        // each output level covers an equal share of the 0-1 range, with 1.0 belonging to the last level
        const double output_levels = static_cast<double>(output_levels_);
        const double level = value_clamped * output_levels;
        return level >= output_levels ? static_cast<T>(output_levels_ - 1) : std::min(static_cast<T>(level), static_cast<T>(output_levels_ - 1));
    }
    else
    {
        return static_cast<T>(value_clamped);
    }
}

template <typename T>
void Noise2D<T>::validate_frequency(double frequency, std::size_t octaves) const
{
    if(!(frequency > 0.0) || !std::isfinite(frequency))
    {
        throw std::invalid_argument("invalid frequency, must be a positive finite value");
    }

    const double coordinate_max = (static_cast<double>(std::max(width_, height_)) + 0.5);
    double frequency_max = frequency;

    for(std::size_t octave = 1; octave < octaves && frequency_max * coordinate_max < detail::LATTICE_COORDINATE_MAX; octave++)
    {
        frequency_max *= 2.0;
    }

    if(frequency_max * coordinate_max >= detail::LATTICE_COORDINATE_MAX)
    {
        throw std::invalid_argument("invalid frequency, too large to sample accurately");
    }

    return;
}

} // namespace noise2d

// Opt-in alias for code written against the class in the global namespace.
#ifdef NOISE2D_GLOBAL_ALIAS
template <typename T>
using Noise2D = noise2d::Noise2D<T>;
#endif

#endif
