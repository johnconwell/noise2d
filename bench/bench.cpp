#include "noise2d.h"
#include <chrono> // std::chrono
#include <cstdio> // std::printf
#include <functional> // std::function

using noise2d::Noise2D;

static void run(const char* name, std::size_t size, const std::function<void(Noise2D<int>&)>& generate)
{
    Noise2D<int> noise(size, size, 256, 1);
    const std::chrono::time_point<std::chrono::steady_clock> time_start = std::chrono::steady_clock::now();
    generate(noise);
    const std::chrono::time_point<std::chrono::steady_clock> time_end = std::chrono::steady_clock::now();
    std::printf("%-8s %4zux%-4zu %10.3f ms\n", name, size, size, std::chrono::duration<double, std::milli>(time_end - time_start).count());
    return;
}

int main()
{
    for(std::size_t size : {64, 256, 512})
    {
        run("white", size, [](Noise2D<int>& noise) { noise.generate_white_noise(); });
        run("brown", size, [](Noise2D<int>& noise) { noise.generate_brown_noise(); });
        run("perlin", size, [](Noise2D<int>& noise) { noise.generate_perlin_noise(0.1); });
        run("fractal", size, [](Noise2D<int>& noise) { noise.generate_fractal_noise(0.05, 8); });

        if(size <= 256)
        {
            run("blue", size, [](Noise2D<int>& noise) { noise.generate_blue_noise(); });
        }
    }

    return 0;
}
