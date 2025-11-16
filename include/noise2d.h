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

#ifndef __NOISE2D_H
#define __NOISE2D_H

#include <algorithm> // std::max
#include <cfloat> // DBL_MIN, DBL_MAX
#include <cmath> // acos
#include <random> // std::random_device, std::mt19337, std::uniform_int_distribution
#include <stdexcept> // std::out_of_range
#include <type_traits> // std::is_floating_point
#include <vector> // std::vector

/**
 * A class containing 2D noise data and methods to generate noise.
 */
template <typename T>
class Noise2D final
{
public:
    /**
     * Do not use default constructor; width and height must be specified.
     */
    Noise2D() = delete;

    /**
     * Constructor for Noise2D objects.
     * 
     * @param width The width of the generated noise matrix.
     * @param height The height of the generated noise matrix.
     * @param output_levels The number of output levels in the generated noise matrix.
     * If noise is a floating-point type, this value is ignored.
     * If noise is an integral type, then the values in the generated noise matrix range from [0, output_levels)
     */
    Noise2D(std::size_t width, std::size_t height, std::size_t output_levels = 256);

    /**
     * Method to get the noise from the noise matrix at an (x, y) coordinate.
     * 
     * @param x The x coordinate of the desired noise data.
     * @param y The y coordinate of the desired noise data.
     * @return The desired noise data.
     * @throw std::out_of_range thrown if x exceeds width or y exceeds height.
     */
    [[nodiscard]] T get_noise_at(std::size_t x, std::size_t y) const;

    /**
     * Method to generate blue noise using Ulichney's Void and Cluster algorithm.
     * 
     * @param sigma The standard deviation of the Gaussian function used to generate blue noise.
     * Low values of sigma result in ordered patterns, while high values of sigma result in stronger low-frequency components.
     * If not specified, a default value of 1.9 is used.
     */
    void generate_blue_noise(double sigma = 1.9);

    /**
     * Method to generate brown noise.
     * 
     * @param leaky_integrator A scalar that is multiplied by the integral sum each step to ensure the noise does not deviate far from zero in large images.
     * If not specified, a default value of 0.999 is used.
     * @param kernel_size The size of the Gaussian kernel used to convolve the white noise into brown noise.
     * A size of 3 corresponds to a 3x3 kernel.
     * If not specified, a default value of 3 is used.
     * @param sigma The standard deviation of the Gaussian kernel used to convolve the white noise into brown noise.
     * A larger value of sigma results in blurrier noise and vice versa.
     */
    void generate_brown_noise(double leaky_integrator = 0.999, std::size_t kernel_size = 3, double sigma = 1.0);

    /**
     * Method to generate white noise.
     */
    void generate_white_noise();

    /**
     * Method to generate Perlin Noise.
     * 
     * @param seed The seed for the psuedo-random number generator, specified to allow repeatable results.
     * @param frequency The frequency of noise. Must be greater than 0. Values greater than 1 increasingly resemble white noise.
     * @throw std::out_of_range if frequency is less than 0.0.
     */
    void generate_perlin_noise(std::size_t seed, double frequency);

    /**
     * Method to generate Fractal Noise ("stacked Perlin noise").
     * 
     * @param starting_frequency The beginning frequency for the Perlin noise, which will contribute the bulk of the noise value at each pixel.
     * @param octaves The number of octaves in the noise. Each octave contributes more noise at twice the frequency and half the amplitude.
     * @throw std::out_of_range if starting frequency is less than 0.0.
     * @throw std::out_of_range if octaves is less than 1.
     */
    void generate_fractal_noise(double starting_frequency, std::size_t octaves);

private:
    class EnergyLUT; // EnergyLUT class declaration
    static inline const std::size_t OUTPUT_LEVELS_MIN = 2; // default value of output_levels, 2 output levels corresponds to a 1 bit image
    static inline const double COVERAGE = 0.1; // default value of coverage

    std::vector<std::vector<T>> data; // matrix the output of noise generation
    std::size_t width; // width of the noise matrix
    std::size_t height; // height of the noise matrix
    std::size_t output_levels; // number of output levels of the noise matrix, not used if matrix type is floating-point

    // blue noise specific members
    // see https://cv.ulichney.com/papers/1993-void-cluster.pdf for information on each algorithm step
    void generate_blue_noise_initial_binary_pattern(double sigma); // generates the blue noise initial binary pattern
    void generate_blue_noise_rank_data_phase_1(double sigma); // fill in noise matrix with ranks from zero to one less than the number of 1's in the initial binary pattern
    void generate_blue_noise_rank_data_phase_2(double sigma); // fill in noise matrix with ranks from the number of 1's in the initial binary pattern to half the number of cells in the matrix
    void generate_blue_noise_rank_data_phase_3(double sigma); // fill in noise matrix with ranks from one more than half the number of cells in the matrix to the number of cells in the matrix
    void normalize_blue_noise_rank_data(); // normalize every value in the matrix to be between [0, output_levels) if data type is integral or [0, 1] if data type is floating-point

    void binary_pattern_copy(std::vector<std::vector<int>>& binary_pattern_source, std::vector<std::vector<int>>& binary_pattern_destination); // copy the values from one binary pattern into another
    void binary_pattern_invert(std::vector<std::vector<int>>& binary_pattern); // change all zeros to ones and vice versa in a binary pattern

    std::vector<std::vector<int>> blue_noise_rank_data; // temporary matrix storing blue noise rank data pre-normalization
    std::vector<std::vector<int>> binary_pattern_initial; // initial state of the blue noise binary pattern (array of zeros and ones, with the ones as evenly spaced as possible)
    std::vector<std::vector<int>> binary_pattern_prototype; // prototype binary pattern used in the blue noise generation algorithm
    EnergyLUT energy_lut; // LUT that maintains the energy values of each cell to speed up blue noise generation
    double coverage; // arbitrary value that determines the starting state of blue noise

    // perlin noise specific members
    double sample_perlin_point(std::size_t seed, double x, double y);
};

/**
 * A class containing an LUT that stores the Gaussian "energy" of each cell in the noise matrix during blue noise generation.
 */
template <typename T>
class Noise2D<T>::EnergyLUT
{
public:
    /**
     * Do not use default constructor; width and height must be specified.
     */
    EnergyLUT() = delete;

    /**
     * Constructor for EnergyLUT objects.
     * 
     * @param width The width of the LUT.
     * @param height The height of the LUT.
     */
    EnergyLUT(std::size_t width, std::size_t height);

    /**
     * Method that fills the LUT with the calculated Gaussian "energy" value at each cell.
     * 
     * @param binary_pattern The binary pattern (matrix of zeros and ones) used to calculate the energy values.
     * @param sigma The standard deviation of the Gaussian function used to calculate the energy at each cell.
     */
    void create(const std::vector<std::vector<int>>& binary_pattern, double sigma);

    /**
     * Method that updates the LUT when a cell is changed.
     * 
     * @param binary_pattern The binary pattern to update.
     * @param x The x coordinate of the cell that has changed from a zero to a one or vice versa.
     * @param y The y coordinate of the cell that has changed from a zero to a one or vice versa.
     * @param sigma The standard deviation of the Gaussian function used to calculate the energy at each cell.
     */
    void update(const std::vector<std::vector<int>>& binary_pattern, std::size_t x, std::size_t y, double sigma);

    std::vector<std::vector<double>> LUT; // the LUT that stores the Gaussian energy of each cell
    std::size_t height; // the height of the LUT
    std::size_t width; // the width of the LUT
    double value_lowest_energy; // the lowest value currently present in the LUT
    double value_highest_energy; // the highest value currently present in the LUT
    std::pair<int, int> coordinate_lowest_energy; // the coordinate of the lowest value currently present in the LUT
    std::pair<int, int> coordinate_highest_energy; // the coordinate of the highest value currently present in the LUT
};

namespace noise2d
{

const double PI = acos(-1.0);

/**
 * Method to convolve a matrix against a kernel.
 * 
 * @param matrix The 2d matrix to convolve.
 * @param kernel The kernel used to perform the convolution.
 * @return The convolved matrix.
 * @note Type T may be an integral or floating-point type, type K must be floating-point type.
 */
template <typename T, typename K>
std::vector<std::vector<T>> convolve(const std::vector<std::vector<T>>& matrix, const std::vector<std::vector<K>>& kernel, double leaky_integrator);

/**
 * Method to generate a Gaussian kernel.
 * 
 * @param size The size of the generated kernel.
 * A size of 3 corresponds to a 3x3 kernel.
 * @param sigma The standard deviation of the Gaussian kernel.
 * A larger value of sigma gives more weight to cells further from the center cell and vice versa.
 * @return The generated kernel.
 * @note Type T must be a floating-point type.
 */
template <typename T>
std::vector<std::vector<T>> gaussian_kernel(std::size_t size, double sigma);

/**
 * Method to compute the dot product between two vectors.
 * 
 * @param v0 The first vector.
 * @param v1 The second vector.
 * @return The dot product of the two vectors.
 */
template <typename T>
double dot_product(const std::pair<T, T>& v0, const std::pair<T, T>& v1);

/**
 * Method to clamp a value between a minimum value and a maximum value.
 * 
 * @param value The value to clamp.
 * @param min The minimum value.
 * @param max The maximum value.
 * @return If value <= min, returns min.
 * If value >= max, returns max.
 * If value > min and value < max, returns value.
 */
template <typename T>
T clamp(T value, T min, T max);

/**
 * Method to perform linear (1st order) interpolation between two values.
 * 
 * @param v0 The first value.
 * @param v1 The second value.
 * @param t The weight of interpolation between v0 and v1.
 * t must be in the range [0, 1].
 * @return If t <= 0, returns v0.
 * If t >= 1, return v1.
 * If t > 0 and t < 1, returns an interpolated value between v0 and v1.
 * @note Type T may be an integral or floating-point type, type K must be floating-point type.
 */
template <typename T>
T interpolate_linear(T v0, T v1, double t);

/**
 * Method to perform cubic (3rd order) interpolation between two values.
 * 
 * @param v0 The first value.
 * @param v1 The second value.
 * @param t The weight of interpolation between v0 and v1.
 * t must be in the range [0, 1].
 * @return If t <= 0, returns v0.
 * If t >= 1, return v1.
 * If t > 0 and t < 1, returns an interpolated value between v0 and v1.
 * @note Type T may be an integral or floating-point type, type K must be floating-point type.
 */
template <typename T>
T interpolate_cubic(T v0, T v1, double t);

/**
 * Method to perform quintic (5th order) interpolation between two values.
 * 
 * @param v0 The first value.
 * @param v1 The second value.
 * @param t The weight of interpolation between v0 and v1.
 * t must be in the range [0, 1].
 * @return If t <= 0, returns v0.
 * If t >= 1, return v1.
 * If t > 0 and t < 1, returns an interpolated value between v0 and v1.
 * @note Type T may be an integral or floating-point type, type K must be floating-point type.
 */
template <typename T>
T interpolate_quintic(T v0, T v1, double t);

/**
 * Method to determine the vector between two points.
 * 
 * @param x0 The x coordinate of the first point.
 * @param y0 The y coordinate of the first point.
 * @param x1 The x coordinate of the second point.
 * @param y1 The y coordinate of the second point.
 * @return A vector pointing from (x0, y0) to (x1, y1)
 */
std::pair<double, double> distance_vector(double x0, double y0, double x1, double y1);

/**
 * Method to generate a unit vector pointing in a random 2d direction.
 * 
 * @param seed The seed for the psuedo-random number generator.
 * @param x An x coordinate that also seeds the psuedo-random number generator for repeatable results on the same coordinate.
 * @param y A y coordinate that also seeds the psuedo-random number generator for repeatable results on the same coordinate.
 * @return A unit vector pointing in a random 2d direction.
 */
std::pair<double, double> random_gradient_vector(std::size_t seed, std::size_t x, std::size_t y);

} // namespace noise2d

template <typename T>
Noise2D<T>::Noise2D(std::size_t width, std::size_t height, std::size_t output_levels) : energy_lut(width, height)
{
    this->data = std::vector<std::vector<T>>(height, std::vector<T>(width, static_cast<T>(0)));
    this->width = width;
    this->height = height;
    this->output_levels = std::max(OUTPUT_LEVELS_MIN, output_levels);

    // blue noise specific members
    this->blue_noise_rank_data = std::vector<std::vector<int>>(height, std::vector<int>(width, 0));
    this->binary_pattern_initial = std::vector<std::vector<int>>(height, std::vector<int>(width, 0));
    this->binary_pattern_prototype = std::vector<std::vector<int>>(height, std::vector<int>(width, 0));

    return;
}

template <typename T>
[[nodiscard]] T Noise2D<T>::get_noise_at(std::size_t x, std::size_t y) const
{
    if(x >= width || y >= height)
    {
        throw std::out_of_range("get_noise_at argument(s) too large");
    }

    return data[y][x];
}

template <typename T>
void Noise2D<T>::generate_blue_noise(double sigma)
{
    generate_blue_noise_initial_binary_pattern(sigma);
    generate_blue_noise_rank_data_phase_1(sigma);
    generate_blue_noise_rank_data_phase_2(sigma);
    generate_blue_noise_rank_data_phase_3(sigma);
    normalize_blue_noise_rank_data();
    return;
}

template <typename T>
void Noise2D<T>::generate_brown_noise(double leaky_integrator, std::size_t kernel_size, double sigma)
{
    std::vector<std::vector<double>> data_white_noise = std::vector<std::vector<double>>(height, std::vector<double>(width, 0.0));
    std::vector<std::vector<double>> kernel = noise2d::gaussian_kernel<double>(kernel_size, sigma);
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<> dis(0, 1);
    double integrated_min = DBL_MAX;
    double integrated_max = DBL_MIN;

    // generate initial white noise matrix to be integrated into brown noise
    for(std::size_t y = 0; y < height; y++)
    {
        for(std::size_t x = 0; x < width; x++)
        {
            data_white_noise[y][x] = dis(mt);
        }
    }

    // convolve the white noise matrix
    std::vector<std::vector<double>> data_temporary = noise2d::convolve<double, double>(data_white_noise, kernel, leaky_integrator);
     
    for(std::size_t y = 0; y < height; y++)
    {
        for(std::size_t x = 0; x < width; x++)
        {
            if(data_temporary[y][x] < integrated_min)
            {
                integrated_min = data_temporary[y][x];
            }

            if(data_temporary[y][x] > integrated_max)
            {
                integrated_max = data_temporary[y][x];
            }
        }
    }

    const double integrated_range = integrated_max - integrated_min;

    // normalize to output levels
    for(std::size_t y = 0; y < height; y++)
    {
        for(std::size_t x = 0; x < width; x++)
        {
            data[y][x] = static_cast<T>((data_temporary[y][x] - integrated_min) * static_cast<double>(output_levels - 1) / integrated_range);
        }
    }

    return;
}

template <typename T>
void Noise2D<T>::generate_white_noise()
{
    std::random_device rd;
    std::mt19937 mt(rd());

    for(std::size_t y = 0; y < height; y++)
    {
        for(std::size_t x = 0; x < width; x++)
        {
            if constexpr(std::is_integral<T>::value)
            {
                std::uniform_int_distribution<> dis(0, output_levels - 1);
                data[y][x] = dis(mt);
            }
            else
            {
                std::uniform_real_distribution<> dis(0, 1);
                data[y][x] = dis(mt);
            }
        }
    }

    return;
}

template <typename T>
void Noise2D<T>::generate_perlin_noise(std::size_t seed, double frequency)
{
    if(frequency <= 0.0)
    {
        throw std::out_of_range("generate_perlin_noise invalid frequency, must be greater than 0.0");
    }

    for(std::size_t y = 0; y < height; y++)
    {
        for(std::size_t x = 0; x < width; x++)
        {
            // convert each pixel coordinate to a floating point sample coordinate scaled to the desired frequency
            const double yf = static_cast<double>(y) * frequency;
            const double xf = static_cast<double>(x) * frequency;

            const double noise = sample_perlin_point(seed, xf, yf);

            if constexpr(std::is_integral<T>::value)
            {
                // normalize to [0, output_levels) range
                data[y][x] = static_cast<T>((noise + 1.0) * static_cast<double>(output_levels) / 2.0);
            }
            else
            {
                // normalize to [0, 1] range
                data[y][x] = static_cast<T>((noise + 1.0) / 2.0);
            }
        }
    }

    return;
}

template <typename T>
void Noise2D<T>::generate_fractal_noise(double starting_frequency, std::size_t octaves)
{
    if(starting_frequency <= 0.0)
    {
        throw std::out_of_range("generate_fractal_noise invalid starting frequency, must be greater than 0.0");
    }
    if(octaves < 1)
    {
        throw std::out_of_range("generate_fractal_noise invalid number of octaves, must be greater than or equal to 1");
    }

    std::random_device rd;
    std::size_t seed = rd();

    for(std::size_t y = 0; y < height; y++)
    {
        for(std::size_t x = 0; x < width; x++)
        {
            double noise = 0.0;
            double frequency = starting_frequency;
            double amplitude = 1.0;

            for(std::size_t octave = 0; octave < octaves; octave++)
            {
                const double yf = static_cast<double>(y) * frequency;
                const double xf = static_cast<double>(x) * frequency;

                noise += amplitude * sample_perlin_point(seed, xf, yf);

                frequency *= 2.0;
                amplitude /= 2.0;
            }

            if constexpr(std::is_integral<T>::value)
            {
                // normalize to [0, output_levels) range
                data[y][x] = static_cast<T>((noise + 1.0) * static_cast<double>(output_levels) / 2.0);
            }
            else
            {
                // normalize to [0, 1] range
                data[y][x] = static_cast<T>((noise + 1.0) / 2.0);
            }
        }
    }

    return;
}

template <typename T>
void Noise2D<T>::generate_blue_noise_initial_binary_pattern(double sigma)
{
    const std::size_t num_pixels = height * width;
    const std::size_t num_minority_pixels = static_cast<std::size_t>(std::max(1, static_cast<int>(static_cast<double>(num_pixels) * COVERAGE))); // must have at least 1 minority pixel
    std::vector<std::pair<int, int>> remaining_coordinates = std::vector<std::pair<int, int>>(num_pixels, {-1, -1});
    std::random_device rd;
    std::mt19937 mt(rd());

    // fill remaining_coordinates with all coordinates in the binary pattern matrix
    for(std::size_t y = 0; y < height; y++)
    {
        for(std::size_t x = 0; x < width; x++)
        {
            // ((y * height) + x) maps 2d array coords to a 1d array
            std::size_t index_remaining_coordinates = (y * height) + x;
            remaining_coordinates[index_remaining_coordinates].first = x;
            remaining_coordinates[index_remaining_coordinates].second = y;
        }
    }

    // shuffle remaining coordinates array
    for(int index_remaining_coordinates = num_pixels - 1; index_remaining_coordinates >= 0; index_remaining_coordinates--)
    {
        std::uniform_int_distribution<> dis(0, index_remaining_coordinates);
        const int index_random = dis(mt);
        std::pair<int, int> temp = remaining_coordinates[index_remaining_coordinates];
        remaining_coordinates[index_remaining_coordinates] = remaining_coordinates[index_random];
        remaining_coordinates[index_random] = temp;
    }

    // fill binary pattern with num_minority_pixels ones
    for(std::size_t index_remaining_coordinates = 0; index_remaining_coordinates < num_pixels; index_remaining_coordinates++)
    {
        std::pair<int, int> coordinate = remaining_coordinates[index_remaining_coordinates];
        if(index_remaining_coordinates < num_minority_pixels)
        {
            binary_pattern_initial[coordinate.second][coordinate.first] = 1;
        }
    }

    energy_lut.create(binary_pattern_initial, sigma);

    // limiting iterations ensures that this will not get stuck forever if there is a bug
    std::size_t count = 0;
    while(count < 100)
    {
        // find the tightest cluster and remove it
        std::pair<int, int> coordinate_tightest_cluster = energy_lut.coordinate_highest_energy;
        binary_pattern_initial[coordinate_tightest_cluster.second][coordinate_tightest_cluster.first] = 0;
        energy_lut.update(binary_pattern_initial, coordinate_tightest_cluster.first, coordinate_tightest_cluster.second, sigma);

        // find the largest void and fill it
        std::pair<int, int> coordinate_largest_void = energy_lut.coordinate_lowest_energy;
        binary_pattern_initial[coordinate_largest_void.second][coordinate_largest_void.first] = 1;
        energy_lut.update(binary_pattern_initial, coordinate_largest_void.first, coordinate_largest_void.second, sigma);

        // exit condition when removing the tightest cluster creates the largest void
        if((coordinate_tightest_cluster.first == coordinate_largest_void.first) && (coordinate_tightest_cluster.second == coordinate_largest_void.second))
        {
            break;
        }

        count++;
    }

    return;
}

template <typename T>
void Noise2D<T>::generate_blue_noise_rank_data_phase_1(double sigma)
{
    int rank = std::max(1, static_cast<int>(static_cast<double>(width * height) * COVERAGE)) - 1; // must have at least 1 minority pixel (one)

    binary_pattern_copy(binary_pattern_initial, binary_pattern_prototype);
    energy_lut.create(binary_pattern_prototype, sigma);

    while(rank >= 0)
    {
        std::pair<int, int> coordinate_tightest_cluster = energy_lut.coordinate_highest_energy;
        binary_pattern_prototype[coordinate_tightest_cluster.second][coordinate_tightest_cluster.first] = 0;
        energy_lut.update(binary_pattern_prototype, coordinate_tightest_cluster.first, coordinate_tightest_cluster.second, sigma);
        blue_noise_rank_data[coordinate_tightest_cluster.second][coordinate_tightest_cluster.first] = rank;
        rank--;
    }

    return;
}

template <typename T>
void Noise2D<T>::generate_blue_noise_rank_data_phase_2(double sigma)
{
    const int num_pixels_half = width * height / 2;
    int rank = std::max(1, static_cast<int>(static_cast<double>(width * height) * COVERAGE)); // must have at least 1 minority pixel (one)

    binary_pattern_copy(binary_pattern_initial, binary_pattern_prototype);
    energy_lut.create(binary_pattern_prototype, sigma);

    while(rank < num_pixels_half)
    {
        std::pair<int, int> coordinate_largest_void = energy_lut.coordinate_lowest_energy;
        binary_pattern_prototype[coordinate_largest_void.second][coordinate_largest_void.first] = 1;
        energy_lut.update(binary_pattern_prototype, coordinate_largest_void.first, coordinate_largest_void.second, sigma);
        blue_noise_rank_data[coordinate_largest_void.second][coordinate_largest_void.first] = rank;
        rank++;
    }

    return;
}

template <typename T>
void Noise2D<T>::generate_blue_noise_rank_data_phase_3(double sigma)
{
    const int num_pixels = width * height;
    int rank = num_pixels / 2;

    binary_pattern_invert(binary_pattern_prototype);
    energy_lut.create(binary_pattern_prototype, sigma);

    while(rank < num_pixels)
    {
        std::pair<int, int> coordinate_tightest_cluster = energy_lut.coordinate_highest_energy;
        binary_pattern_prototype[coordinate_tightest_cluster.second][coordinate_tightest_cluster.first] = 0;
        energy_lut.update(binary_pattern_prototype, coordinate_tightest_cluster.first, coordinate_tightest_cluster.second, sigma);
        blue_noise_rank_data[coordinate_tightest_cluster.second][coordinate_tightest_cluster.first] = rank;
        rank++;
    }

    return;
}

template <typename T>
void Noise2D<T>::normalize_blue_noise_rank_data()
{
    for(std::size_t y = 0; y < height; y++)
    {
        for(std::size_t x = 0; x < width; x++)
        {
            if constexpr(std::is_integral<T>::value)
            {
                data[y][x] = static_cast<T>(static_cast<double>(blue_noise_rank_data[y][x]) * static_cast<double>(output_levels) / static_cast<double>(width * height));
            }
            else
            {
                // the max value in blue_noise_rank_data is width * height, so this normalizes to 0-1 range
                data[y][x] = static_cast<T>(static_cast<double>(blue_noise_rank_data[y][x]) / static_cast<double>(width * height));
            }
        }
    }

    return;
}

template <typename T>
void Noise2D<T>::binary_pattern_copy(std::vector<std::vector<int>>& binary_pattern_source, std::vector<std::vector<int>>& binary_pattern_destination)
{
    for(std::size_t y = 0; y < height; y++)
    {
        for(std::size_t x = 0; x < width; x++)
        {
            binary_pattern_destination[y][x] = binary_pattern_source[y][x];
        }
    }

    return;
}

template <typename T>
void Noise2D<T>::binary_pattern_invert(std::vector<std::vector<int>>& binary_pattern)
{
    for(std::size_t y = 0; y < height; y++)
    {
        for(std::size_t x = 0; x < width; x++)
        {
            binary_pattern[y][x] = binary_pattern[y][x] ? 0 : 1;
        }
    }

    return;
}

template <typename T>
double Noise2D<T>::sample_perlin_point(std::size_t seed, double x, double y)
{
    // find the four surrounding grid intersection coordinates
    const double x0 = static_cast<double>(static_cast<int>(x));
    const double y0 = static_cast<double>(static_cast<int>(y));
    const double x1 = x0 + 1.0;
    const double y1 = y0 + 1.0;

    // determine the x and y offset of the sample coordinate from the top left corner
    const double dx = x - x0;
    const double dy = y - y0;

    // generate a random gradient vector at each surrounding grid intersection
    const std::pair<double, double> graident_vector_00 = noise2d::random_gradient_vector(seed, x0, y0);
    const std::pair<double, double> graident_vector_01 = noise2d::random_gradient_vector(seed, x0, y1);
    const std::pair<double, double> graident_vector_10 = noise2d::random_gradient_vector(seed, x1, y0);
    const std::pair<double, double> graident_vector_11 = noise2d::random_gradient_vector(seed, x1, y1);

    // calculate the distance vector from the sampling coordinate to the surrounding grid intersection coordinates
    const std::pair<double, double> distance_vector_00 = noise2d::distance_vector(x0, y0, x, y);
    const std::pair<double, double> distance_vector_01 = noise2d::distance_vector(x0, y1, x, y);
    const std::pair<double, double> distance_vector_10 = noise2d::distance_vector(x1, y0, x, y);
    const std::pair<double, double> distance_vector_11 = noise2d::distance_vector(x1, y1, x, y);

    // calculate the dot product between the sample coordinate and each surrounding grid intersection coordinate
    const double dot_product_00 = noise2d::dot_product(graident_vector_00, distance_vector_00);
    const double dot_product_01 = noise2d::dot_product(graident_vector_01, distance_vector_01);
    const double dot_product_10 = noise2d::dot_product(graident_vector_10, distance_vector_10);
    const double dot_product_11 = noise2d::dot_product(graident_vector_11, distance_vector_11);

    // interpolate between the horizontal components
    const double interpolation_top = noise2d::interpolate_quintic(dot_product_00, dot_product_10, dx);
    const double interpolation_bottom = noise2d::interpolate_quintic(dot_product_01, dot_product_11, dx);

    // interpolate between the interpolated components
    const double interpolation_total = noise2d::interpolate_quintic(interpolation_top, interpolation_bottom, dy);

    return interpolation_total;
}

template <typename T>
Noise2D<T>::EnergyLUT::EnergyLUT(std::size_t width, std::size_t height)
{
    this->height = height;
    this->width = width;
    LUT = std::vector<std::vector<double>>(height, std::vector<double>(width, 0.0));
    value_lowest_energy = DBL_MAX;
    value_highest_energy = 0.0;
    coordinate_lowest_energy = std::pair<int, int>(-1, -1);
    coordinate_highest_energy = std::pair<int, int>(-1, -1);
    return;
}

template <typename T>
void Noise2D<T>::EnergyLUT::create(const std::vector<std::vector<int>>& binary_pattern, double sigma)
{
    const double half_width = static_cast<double>(width) / 2.0;
    const double half_height = static_cast<double>(height) / 2.0;
    const double two_sigma_squared = 2.0 * sigma * sigma;
    double dy = 0.0;
    double dx = 0.0;

    value_lowest_energy = DBL_MAX;
    value_highest_energy = 0.0;

    for(std::size_t y = 0; y < height; y++)
    {
        for(std::size_t x = 0; x < width; x++)
        {
            // reset LUT value before calculating energy
            LUT[y][x] = 0.0;

            // calculate the contribution of each pixel in the binary pattern
            for(std::size_t j = 0; j < height; j++)
            {
                dy = std::abs(static_cast<double>(y) - static_cast<double>(j));

                if(dy > half_height)
                {
                    dy = height - dy;
                }

                for(std::size_t i = 0; i < width; i++)
                {
                    // only pixels of value 1 contribute to the energy
                    if(binary_pattern[j][i] == 1)
                    {
                        dx = std::abs(static_cast<double>(x) - static_cast<double>(i));
                
                        if(dx > half_width)
                        {
                            dx = width - dx;
                        }

                        LUT[y][x] += exp(-1 * ((dx * dx) + (dy * dy)) / two_sigma_squared);
                    }
                }
            }

            // only 0s (majority pixels) in the binary pattern can be considered the largest void
            if(binary_pattern[y][x] == 0 && LUT[y][x] < value_lowest_energy)
            {
                value_lowest_energy = LUT[y][x];
                coordinate_lowest_energy = std::pair<int, int>(x, y);
            }

            // only 1s (minority pixels) in the binary pattern can be considered the largest cluster
            if(binary_pattern[y][x] == 1 && LUT[y][x] > value_highest_energy)
            {
                value_highest_energy = LUT[y][x];
                coordinate_highest_energy = std::pair<int, int>(x, y);
            }
        }
    }

    return;
}

template <typename T>
void Noise2D<T>::EnergyLUT::update(const std::vector<std::vector<int>>& binary_pattern, std::size_t x, std::size_t y, double sigma)
{
    const double half_width = static_cast<double>(width) / 2.0;
    const double half_height = static_cast<double>(height) / 2.0;
    const double two_sigma_squared = 2 * sigma * sigma;
    double dy = 0.0;
    double dx = 0.0;
    double gaussian_value = 0.0;

    value_lowest_energy = DBL_MAX;
    value_highest_energy = 0.0;

    // calculate the contribution of each pixel in the binary pattern
    for(std::size_t j = 0; j < height; j++)
    {
        dy = std::abs(static_cast<double>(y) - static_cast<double>(j));

        if(dy > half_height)
        {
            dy = height - dy;
        }

        for(std::size_t i = 0; i < width; i++)
        {
            dx = std::abs(static_cast<double>(x) - static_cast<double>(i));
    
            if(dx > half_width)
            {
                dx = width - dx;
            }

            gaussian_value = exp(-1 * ((dx * dx) + (dy * dy)) / two_sigma_squared);

            // if the newly updated pixel is a 1, then add to the other pixels
            if(binary_pattern[y][x] == 1)
            {
                LUT[j][i] += gaussian_value;
            }
            // if the newly updated pixel is a 0, then subtract from the other pixels
            else
            {
                LUT[j][i] -= gaussian_value;
            }

            // only 0s (majority pixels) in the binary pattern can be considered the largest void
            if(binary_pattern[j][i] == 0 && LUT[j][i] < value_lowest_energy)
            {
                value_lowest_energy = LUT[j][i];
                coordinate_lowest_energy = std::pair<int, int>(i, j);
            }

            // only 1s (minority pixels) in the binary pattern can be considered the largest cluster
            if(binary_pattern[j][i] == 1 && LUT[j][i] > value_highest_energy)
            {
                value_highest_energy = LUT[j][i];
                coordinate_highest_energy = std::pair<int, int>(i, j);
            }
        }
    }

    return;
}

template <typename T, typename K>
std::vector<std::vector<T>> noise2d::convolve(const std::vector<std::vector<T>>& matrix, const std::vector<std::vector<K>>& kernel, double leaky_integrator)
{
    const std::size_t matrix_height = matrix.size();
    const std::size_t matrix_width = matrix[0].size();
    const std::size_t kernel_height = kernel.size();
    const std::size_t kernel_width = kernel[0].size();
    const std::size_t kernel_height_half = kernel_height / 2;
    const std::size_t kernel_width_half = kernel_width / 2;
    std::vector<std::vector<T>> convolved_matrix = std::vector<std::vector<T>>(matrix_height, std::vector<T>(matrix_width, 0.0));
    double sum = 0.0;

    for(std::size_t my = 0; my < matrix_height; my++)
    {
        for(std::size_t mx = 0; mx < matrix_width; mx++)
        {
            sum = 0;
            
            for(std::size_t ky = 0; ky < kernel_height; ky++)
            {
                for(std::size_t kx = 0; kx < kernel_width; kx++)
                {
                    std::size_t dy = (matrix_height + ((my + ky - kernel_height_half) % matrix_height)) % matrix_height;
                    std::size_t dx = (matrix_width + ((mx + kx - kernel_width_half) % matrix_width)) % matrix_width;
                    sum += static_cast<double>(matrix[dy][dx]) * static_cast<double>(kernel[ky][kx]) * leaky_integrator;
                }
            }
            
            convolved_matrix[my][mx] = static_cast<T>(sum);
        }
    }

    return convolved_matrix;
}

template <typename T>
std::vector<std::vector<T>> noise2d::gaussian_kernel(std::size_t size, double sigma)
{
    std::vector<std::vector<T>> kernel = std::vector<std::vector<T>>(size, std::vector<T>(size, static_cast<T>(0)));

    const double NOISE2D_PI = acos(-1.0);
    const int half_size = size / 2;
    const float two_sigma_squared = 2.0 * sigma * sigma;
    double sum = 0.0;

    for(int y = -half_size; y < half_size + 1; y++)
    {
        for(int x = -half_size; x < half_size + 1; x++)
        {
            kernel[y + half_size][x + half_size] = exp(-1 * (x * x + y * y) / two_sigma_squared) / (NOISE2D_PI * two_sigma_squared);
            sum += kernel[y + half_size][x + half_size];
        }
    }

    for(std::size_t y = 0; y < size; y++)
    {
        for(std::size_t x = 0; x < size; x++)
        {
            kernel[y][x] /= sum;
        }
    }

    return kernel;
}

template <typename T>
double noise2d::dot_product(const std::pair<T, T>& v0, const std::pair<T, T>& v1)
{
    return static_cast<double>(v0.first * v1.first + v0.second * v1.second);
}

template <typename T>
T noise2d::clamp(T value, T min, T max)
{
    T clamped_value = value;

    if(value < min)
    {
        clamped_value = min;
    }
    else if(value > max)
    {
        clamped_value = max;
    }

    return clamped_value;
}

template <typename T>
T noise2d::interpolate_linear(T v0, T v1, double t)
{
    const double t_clamped = noise2d::clamp(t, 0.0, 1.0);
    return static_cast<T>(static_cast<double>(v0) + t_clamped * static_cast<double>(v1 - v0));
}

template <typename T>
T noise2d::interpolate_cubic(T v0, T v1, double t)
{
    const double t_clamped = noise2d::clamp(t, 0.0, 1.0);
    return static_cast<T>(static_cast<double>(v0) + (t_clamped * t_clamped * (3.0 - t_clamped * 2.0)) * static_cast<double>(v1 - v0));
}

template <typename T>
T noise2d::interpolate_quintic(T v0, T v1, double t)
{
    const double t_clamped = noise2d::clamp(t, 0.0, 1.0);
    return static_cast<T>(static_cast<double>(v0) + (t_clamped * t_clamped * t_clamped * (t_clamped * (t_clamped * 6.0 - 15.0) + 10.0)) * static_cast<double>(v1 - v0));
}

std::pair<double, double> noise2d::distance_vector(double x0, double y0, double x1, double y1)
{
    return std::pair<double, double>(x1 - x0, y1 - y0);
}

std::pair<double, double> noise2d::random_gradient_vector(std::size_t seed, std::size_t x, std::size_t y)
{
    std::pair<double, double> vector = std::pair<double, double>(0.0, 0.0);
    std::seed_seq sseq{seed, x, y}; // seed the random number generator based on the x and y coordinates and a constant seed so that the outcome is repeatable
    std::mt19937 mt(sseq);
    std::uniform_real_distribution<> dist(0.0, 2.0 * PI);
    const double random = dist(mt);
    vector.first = sin(random);
    vector.second = cos(random);
    return vector;
}

#endif
