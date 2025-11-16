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

#include "image.h"
#include <iostream>

Color::Color()
{
    r = 0;
    g = 0;
    b = 0;
    a = Color::CHANNEL_MAX;
    return;
}

Color::Color(int16_t r, int16_t g, int16_t b, int16_t a)
{
    this->r = r;
    this->g = g;
    this->b = b;
    this->a = a;
    return;
}

// initializes empty image
Image::Image()
{
    pixels.resize(0);
    width = 0;
    height = 0;
    return;
}

// saves a png to the specified path
std::size_t Image::save(const char* file_name)
{
    std::size_t error = lodepng::encode(file_name, pixels, width, height);

    if(error)
    {
        std::cout << "error: save - " << error << ": "<< lodepng_error_text(error) << std::endl;
    }

    return error;
}

// fills image with a grayscale representation of specified threshold map
void Image::create_from_matrix(const std::vector<std::vector<int>>& matrix)
{
    height = matrix.size();
    width = matrix[0].size();
    pixels.resize(width * height * Color::NUM_BYTES_COLOR);

    for(std::size_t y = 0; y < height; y++)
    {
        for(std::size_t x = 0; x < width; x++)
        {
            const std::size_t index_pixels = Color::NUM_BYTES_COLOR * width * y + Color::NUM_BYTES_COLOR * x;
            pixels[index_pixels + 0] = matrix[y][x];
            pixels[index_pixels + 1] = matrix[y][x];
            pixels[index_pixels + 2] = matrix[y][x];
            pixels[index_pixels + 3] = Color::CHANNEL_MAX;
        }
    }

    return;
}
