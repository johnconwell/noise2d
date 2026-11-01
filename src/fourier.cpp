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

#include "fourier.h"
#include <algorithm> // std::max, std::min
#include <climits> // INT_MAX, INT_MIN
#include <cmath> // std::log1p, std::sqrt

Fourier2D::Fourier2D()
{
    dataset = std::vector<std::vector<int>>(0, std::vector<int>(0, 0));
    height = 0;
    width = 0;
    transform = std::vector<std::vector<int>>(0, std::vector<int>(0, 0));
    transform_log_magnitude = std::vector<std::vector<double>>(0, std::vector<double>(0, 0.0));
    remove_dc_offset = false;
    center_output = false;
    return;
}

Fourier2D::Fourier2D(std::vector<std::vector<int>> dataset, bool remove_dc_offset, bool center_output)
{
    this->dataset = dataset;
    this->height = dataset.size();
    this->width = dataset[0].size();
    this->transform = std::vector<std::vector<int>>(height, std::vector<int>(width, 0));
    this->transform_log_magnitude = std::vector<std::vector<double>>(height, std::vector<double>(width, 0.0));
    this->remove_dc_offset = remove_dc_offset;
    this->center_output = center_output;
    return;
}

// returns the dataset
std::vector<std::vector<int>> Fourier2D::get_dataset()
{
    return dataset;
}

// returns the transform
std::vector<std::vector<int>> Fourier2D::get_transform()
{
    return transform;
}

// computes the dft of the dataset and stores the result in transform
void Fourier2D::dft()
{
    const std::size_t size = width * height;

    fftw_complex *in = fftw_alloc_complex(size);
    fftw_complex *out = fftw_alloc_complex(size);
    fftw_plan plan = fftw_plan_dft_2d(height, width, in, out, FFTW_FORWARD, FFTW_ESTIMATE);

    // fill the in array with data from dataset
    for(std::size_t y = 0; y < height; y++)
    {
        for(std::size_t x = 0; x < width; x++)
        {
            in[y * width + x][0] = static_cast<double>(dataset[y][x]);
            in[y * width + x][1] = 0.0;
        }
    }

    // remove DC offset by subtracting the mean of the data
    if(remove_dc_offset)
    {
        double mean = 0.0;

        for(std::size_t y = 0; y < height; y++)
        {
            for(std::size_t x = 0; x < width; x++)
            {
                mean += in[y * width + x][0];
            }
        }

        mean /= static_cast<double>(size);

        for(std::size_t y = 0; y < height; y++)
        {
            for(std::size_t x = 0; x < width; x++)
            {
                in[y * width + x][0] -= mean;
            }
        }
    }

    // center output
    if(center_output)
    {
        for(std::size_t y = 0; y < height; y++)
        {
            for(std::size_t x = 0; x < width; x++)
            {
                in[y * width + x][0] *= ((x + y) & 1) ? -1.0 : 1.0;
            }
        }
    }

    fftw_execute(plan);

    // fill output vector with the results of the transform
    for(std::size_t y = 0; y < height; y++)
    {
        for(std::size_t x = 0; x < width; x++)
        {
            // a log scale keeps the low-magnitude structure of the spectrum visible
            transform_log_magnitude[y][x] = std::log1p(magnitude(out[y * width + x]));
            transform[y][x] = static_cast<int>(transform_log_magnitude[y][x]);
        }
    }

    fftw_destroy_plan(plan);
    fftw_free(in);
    fftw_free(out);

    return;
}

// normalizes the dataset (maps values to [0, output_levels - 1])
void Fourier2D::normalize_dataset(std::size_t output_levels)
{
    normalize(dataset, output_levels);
    return;
}

// normalize the transform (maps log magnitudes to [0, output_levels - 1])
void Fourier2D::normalize_transform(std::size_t output_levels)
{
    double array_min = transform_log_magnitude[0][0];
    double array_max = transform_log_magnitude[0][0];

    for(std::size_t y = 0; y < height; y++)
    {
        for(std::size_t x = 0; x < width; x++)
        {
            array_min = std::min(array_min, transform_log_magnitude[y][x]);
            array_max = std::max(array_max, transform_log_magnitude[y][x]);
        }
    }

    const double array_range = array_max > array_min ? array_max - array_min : 1.0; // prevent divide by 0 errors if all values are the same

    for(std::size_t y = 0; y < height; y++)
    {
        for(std::size_t x = 0; x < width; x++)
        {
            transform[y][x] = static_cast<int>((transform_log_magnitude[y][x] - array_min) * static_cast<double>(output_levels - 1) / array_range);
        }
    }

    return;
}

std::string Fourier2D::to_string()
{
    std::string output = "";

    output += "Dataset:\n";
    for(std::size_t y = 0; y < height; y++)
    {
        for(std::size_t x = 0; x < width; x++)
        {
            output += std::to_string(dataset[y][x]) + " ";
        }
        
        output += "\n";
    }

    output += "Transform:\n";
    for(std::size_t y = 0; y < height; y++)
    {
        for(std::size_t x = 0; x < width; x++)
        {
            output += std::to_string(transform[y][x]) + " ";
        }
        
        output += "\n";
    }

    return output;
}

// returns the magnitude of a complex value
double Fourier2D::magnitude(fftw_complex& value)
{
    return std::sqrt(value[0] * value[0] + value[1] * value[1]);
}

// normalizes the array (maps values to [0, output_levels - 1])
void Fourier2D::normalize(std::vector<std::vector<int>> &array, std::size_t output_levels)
{
    int array_min = INT_MAX;
    int array_max = INT_MIN;

    for(std::size_t y = 0; y < height; y++)
    {
        for(std::size_t x = 0; x < width; x++)
        {
            array_min = std::min(array_min, array[y][x]);
            array_max = std::max(array_max, array[y][x]);
        }
    }

    const int array_range = std::max(array_max - array_min, 1); // prevent divide by 0 errors if all values are the same

    // normalize to output levels
    for(std::size_t y = 0; y < height; y++)
    {
        for(std::size_t x = 0; x < width; x++)
        {
            array[y][x] = (array[y][x] - array_min) * (output_levels - 1) / array_range;
        }
    }

    return;
}
