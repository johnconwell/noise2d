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

#ifndef __FOURIER_H
#define __FOURIER_H

#include <cmath> // min(), max()
#include <cstddef> // size_t
#include <fftw3.h> // fft types + functions
#include <string> // std::string
#include <vector> // std::vector

class Fourier2D
{
public:
    Fourier2D();
    Fourier2D(std::vector<std::vector<int>> dataset, bool remove_dc_offset, bool center_output);
    std::vector<std::vector<int>> get_dataset();
    std::vector<std::vector<int>> get_transform();
    void dft();
    void idft();
    void normalize_dataset(std::size_t output_levels);
    void normalize_transform(std::size_t output_levels);

    std::string to_string();

private:
    void normalize(std::vector<std::vector<int>> &array, std::size_t output_levels);
    double magnitude(fftw_complex &value);

    std::vector<std::vector<int>> dataset;
    std::size_t height;
    std::size_t width;
    std::vector<std::vector<int>> transform;
    bool remove_dc_offset;
    bool center_output;
};

#endif
