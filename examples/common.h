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

#ifndef NOISE2D_EXAMPLES_COMMON_H
#define NOISE2D_EXAMPLES_COMMON_H

#include "fourier.h" // dft wrapper
#include "image.h" // png encoder/decoder wrapper
#include "noise2d.h" // noise generation functions
#include <chrono> // std::chrono
#include <cstdint> // std::uint64_t
#include <filesystem> // std::filesystem
#include <functional> // std::function
#include <iostream> // std::cout, std::cerr, std::endl
#include <string> // std::string
#include <vector> // std::vector

using NoiseGenerator = std::function<void(noise2d::Noise2D<int>&)>;

inline std::vector<std::vector<int>> create_matrix_from_noise(const noise2d::Noise2D<int>& noise)
{
    std::vector<std::vector<int>> matrix = std::vector<std::vector<int>>(noise.height(), std::vector<int>(noise.width(), 0));

    for(std::size_t y = 0; y < noise.height(); y++)
    {
        for(std::size_t x = 0; x < noise.width(); x++)
        {
            matrix[y][x] = noise(x, y);
        }
    }

    return matrix;
}

// returns the microseconds elapsed since a starting time
inline std::string microseconds_since(std::chrono::time_point<std::chrono::steady_clock> time_start)
{
    const std::chrono::nanoseconds elapsed = std::chrono::steady_clock::now() - time_start;
    return std::to_string(elapsed.count() / 1000) + " us";
}

// generates one size of noise and its Fourier transform, saves both to .png files, and prints benchmarks
inline void run_size(const std::string& name, std::size_t size, const NoiseGenerator& generate)
{
    const std::size_t output_levels = 256;
    const std::uint64_t seed = 1; // a fixed seed makes every output image reproducible
    const std::string size_name = std::to_string(size) + "x" + std::to_string(size);
    const std::filesystem::path output_directory = "output";
    noise2d::Noise2D<int> noise = noise2d::Noise2D<int>(size, size, output_levels, seed);
    Image image = Image();

    std::chrono::time_point<std::chrono::steady_clock> time_start = std::chrono::steady_clock::now();
    generate(noise);
    std::cout << size_name << " time: " << microseconds_since(time_start) << std::endl;

    const std::vector<std::vector<int>> matrix = create_matrix_from_noise(noise);
    image.create_from_matrix(matrix);
    image.save((output_directory / (name + "_" + size_name + ".png")).string().c_str());

    time_start = std::chrono::steady_clock::now();
    Fourier2D fourier_2d = Fourier2D(matrix, true, true);
    fourier_2d.dft();
    fourier_2d.normalize_transform(output_levels);
    std::cout << size_name << " fourier time: " << microseconds_since(time_start) << std::endl;

    image.create_from_matrix(fourier_2d.get_transform());
    image.save((output_directory / (name + "_" + size_name + "_fourier.png")).string().c_str());

    return;
}

// runs an example at every size, returning its exit code
inline int run_example(const std::string& name, const NoiseGenerator& generate)
{
    // create output directory if it does not exist
    try
    {
        std::filesystem::create_directories("output");
    }
    catch(const std::filesystem::filesystem_error& e)
    {
        std::cerr << "Filesystem error: " << e.what() << std::endl;
        return 1;
    }

    for(std::size_t size : {2, 4, 8, 16, 32, 64, 128, 256})
    {
        run_size(name, size, generate);
    }

    return 0;
}

#endif
