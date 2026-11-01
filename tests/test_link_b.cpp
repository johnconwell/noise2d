#define NOISE2D_GLOBAL_ALIAS
#include "noise2d.h"
#include "test_helpers.h"

int link_test_b()
{
    Noise2D<int> noise(8, 8);
    noise.generate_white_noise();
    noise.generate_brown_noise();
    noise.generate_blue_noise();
    noise.generate_perlin_noise(0.2);
    noise.generate_fractal_noise(0.1, 3);
    return noise.get_noise_at(0, 0);
}
