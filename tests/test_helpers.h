#ifndef NOISE2D_TEST_HELPERS_H
#define NOISE2D_TEST_HELPERS_H

#include "noise2d.h"
#include <set> // std::set
#include <utility> // std::pair
#include <vector> // std::vector

template <typename T>
std::size_t distinct_values(const noise2d::Noise2D<T>& noise)
{
    return std::set<T>(noise.values().begin(), noise.values().end()).size();
}

template <typename T>
std::pair<T, T> min_max(const noise2d::Noise2D<T>& noise)
{
    const auto extremes = std::minmax_element(noise.values().begin(), noise.values().end());
    return std::pair<T, T>(*extremes.first, *extremes.second);
}

template <typename T>
bool equal(const noise2d::Noise2D<T>& noise_a, const noise2d::Noise2D<T>& noise_b)
{
    return noise_a.values() == noise_b.values();
}

// generators called from separate translation units, so that linking them together proves the header is header-only
int link_test_a();
int link_test_b();

#endif
