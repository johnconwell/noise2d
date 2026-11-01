cmake_minimum_required(VERSION 3.14)
project(noise2d LANGUAGES CXX)

# examples and tests are only built by default when this is the top-level project
if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    set(NOISE2D_TOP_LEVEL ON)
else()
    set(NOISE2D_TOP_LEVEL OFF)
endif()

option(NOISE2D_BUILD_EXAMPLES "Build the example programs (requires FFTW)" ${NOISE2D_TOP_LEVEL})
option(NOISE2D_BUILD_TESTS "Build the test suite" ${NOISE2D_TOP_LEVEL})

add_library(noise2d INTERFACE)
add_library(noise2d::noise2d ALIAS noise2d)
target_include_directories(noise2d INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(noise2d INTERFACE cxx_std_17)

if(NOISE2D_BUILD_EXAMPLES)
    find_path(FFTW3_INCLUDE_DIR fftw3.h)
    find_library(FFTW3_LIBRARY NAMES fftw3 fftw3-3)

    if(NOT FFTW3_INCLUDE_DIR OR NOT FFTW3_LIBRARY)
        message(FATAL_ERROR "FFTW not found. Install it (see README.md) or configure with -DNOISE2D_BUILD_EXAMPLES=OFF.")
    endif()

    add_library(noise2d_example_support STATIC src/fourier.cpp src/image.cpp src/lodepng.cpp)
    target_include_directories(noise2d_example_support PUBLIC include ${FFTW3_INCLUDE_DIR})
    target_compile_features(noise2d_example_support PUBLIC cxx_std_17)
    target_link_libraries(noise2d_example_support PUBLIC ${FFTW3_LIBRARY})

    foreach(example blue_noise brown_noise white_noise perlin_noise fractal_noise)
        add_executable(${example} examples/example_${example}.cpp)
        target_link_libraries(${example} PRIVATE noise2d noise2d_example_support)
    endforeach()
endif()

if(NOISE2D_BUILD_TESTS)
    enable_testing()
    add_executable(noise2d_tests tests/test_main.cpp tests/test_link_a.cpp tests/test_link_b.cpp)
    target_link_libraries(noise2d_tests PRIVATE noise2d)
    add_test(NAME noise2d_tests COMMAND noise2d_tests)
endif()
