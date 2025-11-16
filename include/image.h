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

#ifndef __IMAGE_H
#define __IMAGE_H

#include "lodepng.h" // png encode/decode functions
#include <cstdint> // int16_t
#include <vector> // std::vector

struct Color
{
    Color();
    Color(int16_t r, int16_t g, int16_t b, int16_t a);

    static inline constexpr int NUM_BYTES_COLOR = 4;
    static inline constexpr int INDEX_R = 0;
    static inline constexpr int INDEX_G = 1;
    static inline constexpr int INDEX_B = 2;
    static inline constexpr int INDEX_A = 3;
    static inline constexpr int CHANNEL_MAX = 255;

    int16_t r;
    int16_t g;
    int16_t b;
    int16_t a;
};

class Image
{
public:
    Image();
    std::size_t save(const char* file_name);
    void create_from_matrix(const std::vector<std::vector<int>>& matrix);

private:
    std::vector<unsigned char> pixels;
    unsigned int width;
    unsigned int height;
};

#endif
