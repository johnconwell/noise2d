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

#include "fftw3.h" // for dft functions
#include "fourier.h" // dft wrapper
#include "image.h" // png encoder/decoder wrapper
#include "lodepng.h" // png encoder/decoder
#include "noise2d.h" // noise generation functions
#include <chrono> // std::chrono::time_point, std::chrono::steady_clock, std::chrono::nanoseconds
#include <filesystem> // std::filesystem
#include <iostream> // std::cout, std::endl

std::vector<std::vector<int>> create_matrix_from_noise(Noise2D<int> noise, std::size_t width, std::size_t height);
std::string generate_perlin_noise(int width, int height, int output_levels, std::size_t seed, double frequency, bool fourier, bool benchmark);

int main()
{
    // create output directory if it does not exist
    std::filesystem::path dirPath = "output";
    try
    {
        std::filesystem::create_directory(dirPath);
    }
    catch(const std::filesystem::filesystem_error& e)
    {
        std::cerr << "Filesystem error: " << e.what() << std::endl;
    }

    std::size_t output_levels = 256;
    std::size_t seed = 1;
    double frequency = 0.2;
    bool fourier = true;
    bool benchmark = true;

    std::cout << generate_perlin_noise(2, 2, output_levels, seed, frequency, fourier, benchmark) << std::endl;
    std::cout << generate_perlin_noise(4, 4, output_levels, seed, frequency, fourier, benchmark) << std::endl;
    std::cout << generate_perlin_noise(8, 8, output_levels, seed, frequency, fourier, benchmark) << std::endl;
    std::cout << generate_perlin_noise(16, 16, output_levels, seed, frequency, fourier, benchmark) << std::endl;
    std::cout << generate_perlin_noise(32, 32, output_levels, seed, frequency, fourier, benchmark) << std::endl;
    std::cout << generate_perlin_noise(64, 64, output_levels, seed, frequency, fourier, benchmark) << std::endl;

    return 0;
}

std::string generate_perlin_noise(int width, int height, int output_levels, std::size_t seed, double frequency, bool fourier, bool benchmark)
{
    std::string output = "";
    std::chrono::time_point<std::chrono::steady_clock> time_start;
    std::chrono::time_point<std::chrono::steady_clock> time_end;
    Noise2D<int> perlin_noise = Noise2D<int>(width, height, output_levels);
    Image image = Image();
    char file_name[1000];
    sprintf(file_name, "output\\perlin_noise_%ix%i.png", width, height);

    if(benchmark)
    {
        char heading[100];
        sprintf(heading, "%ix%i time: ", width, height);
        output += heading;
        time_start = std::chrono::steady_clock::now();
    }

    perlin_noise.generate_perlin_noise(seed, frequency);

    if(benchmark)
    {
        time_end = std::chrono::steady_clock::now();
        output += std::to_string(std::chrono::nanoseconds(time_end - time_start).count() / 1000) + " us\n";
    }

    std::vector<std::vector<int>> matrix = create_matrix_from_noise(perlin_noise, width, height);
    image.create_from_matrix(matrix);
    image.save(file_name);

    if(fourier)
    {
        if(benchmark)
        {
            char heading[100];
            sprintf(heading, "%ix%i fourier time: ", width, height);
            output += heading;
            time_start = std::chrono::steady_clock::now();
        }

        Fourier2D fourier_2d = Fourier2D(matrix, true, true);
        fourier_2d.dft();
        fourier_2d.normalize_transform(output_levels);

        if(benchmark)
        {
            time_end = std::chrono::steady_clock::now();
            output += std::to_string(std::chrono::nanoseconds(time_end - time_start).count() / 1000) + " us\n";
        }

        sprintf(file_name, "output\\perlin_noise_%ix%i_fourier.png", width, height);
        image.create_from_matrix(fourier_2d.get_transform());
        image.save(file_name);
    }

    return output;
}

std::vector<std::vector<int>> create_matrix_from_noise(Noise2D<int> noise, std::size_t width, std::size_t height)
{
    std::vector<std::vector<int>> matrix = std::vector<std::vector<int>>(height, std::vector<int>(width, 0));

    for(std::size_t y = 0; y < height; y++)
    {
        for(std::size_t x = 0; x < width; x++)
        {
            matrix[y][x] = noise.get_noise_at(x, y);
        }
    }

    return matrix;
}
