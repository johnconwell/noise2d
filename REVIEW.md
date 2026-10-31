# Noise2D Project Review

**Reviewed:** 31 Oct 2026
**Scope:** `include/noise2d.h` (the library), `README.md` (the spec), `examples/*.cpp`, `src/fourier.cpp`, `src/image.cpp`, `Makefile`. Vendored third-party code (`lodepng`, `fftw3.h`) was not reviewed.

**Method:** I read all of the code above. I then compiled `noise2d.h` with g++ 13.3 (`-std=c++17 -Wall -Wextra`, with AddressSanitizer/UBSan for crash checks) and ran small test programs to confirm each suspected bug. Findings marked **[verified]** were reproduced; the rest come from reading the code.

---

## 1. Summary

The library is small, readable and well commented, and its public API is easy to use. The Void-and-Cluster blue noise is a faithful implementation of Ulichney's paper for square images. However:

- **The library is not actually header-only.** Including `noise2d.h` from two `.cpp` files gives a linker error.
- **Non-square images are broken.** Blue noise reads and writes out of bounds, and the Fourier helper used by the examples has the same indexing bug.
- **Brown noise does not match its spec.** It is a Gaussian blur, not integrated white noise, and the `leaky_integrator` parameter does nothing.
- **Perlin noise gives a flat image at any whole-number frequency**, which contradicts the README's claim that values above 1 look like white noise.
- **Output ranges differ between generators**, and for floating-point `T`, `generate_brown_noise` ignores the README's promise.
- **The default `make` target cannot build**, and there are no tests.

The highest-value fixes are small: add `inline` to two functions, change `y * height + x` to `y * width + x` in two places, and update the README to describe what brown and fractal noise actually do.

---

## 2. Architecture

### 2.1 Structure

```
include/noise2d.h      ← the library (template class Noise2D<T> + nested EnergyLUT + noise2d:: helpers)
examples/*.cpp         ← one program per noise color; nearly identical copy-paste
src/fourier.cpp/.h     ← FFTW wrapper (examples only)
src/image.cpp/.h       ← lodepng wrapper (examples only)
bin/, lib/             ← prebuilt Windows libfftw3-3.dll (committed twice)
docs/                  ← README images
```

Keeping the library separate from the example-only dependencies (FFTW, lodepng) is a good choice: `noise2d.h` really does depend only on the standard library.

### 2.2 Design observations

| # | Observation | Impact | Suggestion |
|---|---|---|---|
| A1 | One class (`Noise2D<T>`) holds the output matrix **plus** all scratch state for blue noise: `blue_noise_rank_data`, `binary_pattern_initial`, `binary_pattern_prototype`, and the `EnergyLUT` with its own W×H `double` matrix. All of it is allocated in the constructor, even if you only want white noise. | About 5× the memory you need, and it causes bug B4 (state carries over between calls). | Make the blue-noise buffers local variables in `generate_blue_noise()`, or a separate helper object built for each call. |
| A2 | Storage is `std::vector<std::vector<T>>`. | Every row is a separate allocation, memory isn't contiguous, and the data is cache-unfriendly. | Use a flat `std::vector<T>` indexed `y * width + x`. |
| A3 | The only way to read data is `get_noise_at(x, y)`, which checks bounds on every call. | Bulk reads are slow. The examples get around this by copying the **whole object by value** (`create_matrix_from_noise(Noise2D<int> noise, …)`), which copies every scratch buffer too. | Add `const T* data() const`, or `const std::vector<T>& values() const`. Pass by `const&` in the examples. |
| A4 | Each `generate_*` method overwrites the same matrix, so the object is really "a matrix plus some fill functions." | That's workable, but the class name and the README ("base class for 2D noise generation") suggest more. Note the class is `final`, so it can't be a base class. | Fix the README wording, or switch to free functions that return a matrix. |
| A5 | Seeding is inconsistent. Only Perlin takes a seed. White, brown, blue and fractal use `std::random_device` internally and can't be reproduced. | You can't regenerate a texture or write a deterministic test. On older MinGW, `std::random_device` is also known to be deterministic, giving the *same* "random" output on every run. | Give every generator an optional `seed` parameter (or set one in the constructor), and use a single `std::mt19937_64` member. |
| A6 | `T` isn't constrained, and `output_levels` isn't checked against `T`'s range. | `Noise2D<signed char>` with the default 256 levels produces negative values **[verified: min = −128]**. `Noise2D<bool>` and `Noise2D<std::string>` compile in odd ways or fail with confusing errors. | Add `static_assert(std::is_arithmetic_v<T>)`, and clamp or throw when `output_levels - 1 > std::numeric_limits<T>::max()`. |
| A7 | Width and height aren't validated. | A 0×0 object crashes `generate_blue_noise` **[verified: segfault]**. `width * height` is also stored in `int` in places, so it overflows above about 46 000². | Throw `std::invalid_argument` in the constructor when width or height is 0. |
| A8 | Leftover code: the `coverage` member is never set or read, `interpolate_linear` and `interpolate_cubic` are unused, `noise2d::clamp` duplicates `std::clamp`, and `PI` is defined twice (`noise2d::PI` and a local `NOISE2D_PI`). | Clutter. | Remove them. |
| A9 | The helpers in the `noise2d::` namespace are public, while the class itself sits in the global namespace. | Inconsistent, and `Noise2D` could clash with other names. | Put everything in `namespace noise2d`, with helpers in `noise2d::detail`. |
| A10 | Generation is O(N²) and O(N·samples) with no precomputation (see §5). | Blue noise beyond about 128² and Perlin beyond about 256² become impractically slow. | See §5. |

---

## 3. Bugs

Severity: 🔴 crash, UB, or build break · 🟠 wrong output · 🟡 minor or latent

### 🔴 B1 — The header isn't header-only: two TUs give a multiple-definition link error **[verified]**
`noise2d.h:985` and `:990`: `noise2d::distance_vector` and `noise2d::random_gradient_vector` are ordinary (non-template, non-`inline`) functions defined in the header.

```
b.cpp: multiple definition of `noise2d::distance_vector(double, double, double, double)'
b.cpp: multiple definition of `noise2d::random_gradient_vector(unsigned long, unsigned long, unsigned long)'
```
This breaks the main promise in the README's Usage section ("`#include "noise2d.h"` in any files where you access the API"). The examples don't catch it because each one is a single TU.
**Fix:** mark both functions `inline`. Also make `PI` `inline constexpr`.

### 🔴 B2 — Blue noise is out of bounds for non-square images **[verified with ASan]**
`noise2d.h:527`: `index = (y * height) + x` should be `(y * width) + x`.
- 16×8 → heap-buffer-overflow **read**
- 8×16 → heap-buffer-overflow **write**

For square images the bug is invisible, which is why all the examples work.

### 🔴 B3 — `gaussian_kernel` writes out of bounds for even sizes, and crashes for size 0 **[verified with ASan]**
`noise2d.h:917–925`: the loop runs from `-half` to `+half` inclusive, which is `2*half + 1` entries. For `size = 4` that's 5, so it writes past the end of the array. For `kernel_size = 0` the kernel is empty, and `convolve` then reads `kernel[0]`.
The README describes `kernel_size` as freely chosen ("A size of 3 corresponds to a 3x3 kernel").
**Fix:** require an odd size of at least 1 (throw otherwise), or build the kernel with exactly `size` entries.

### 🟠 B4 — Calling `generate_blue_noise()` twice corrupts the result **[verified]**
`binary_pattern_initial` is never cleared, so the second call adds a new set of random 1s on top of the old pattern. Phase 1 then ranks only `num_minority` of those cells. The extra 1s keep their stale ranks from the first call, so ranks get duplicated.
- One call on 16×16: 256 distinct ranks (correct).
- Two calls on the same object: **233 distinct ranks.**

**Fix:** reset the buffers at the start of each call, or make them local (A1).

### 🟠 B5 — Brown noise isn't brown noise, and `leaky_integrator` does nothing
`generate_brown_noise` (`:350`) does a **single 3×3 Gaussian blur** of white noise and then normalizes by min and max.
- A Gaussian blur is a low-pass filter. It does not give the 1/f² spectrum that defines brown (Brownian) noise, and nothing is integrated (no cumulative sum).
- `convolve` (`:900`) multiplies **every term** by `leaky_integrator`. That just scales the whole image by a constant, and the min/max normalization then cancels it. **So the parameter has no effect at all, for any value.**

The README says brown noise is made "by first generating white noise, then integrating the result", and describes `leaky_integrator` as applied "each step". The code does neither.
**Fix, option 1:** implement real integration, e.g. a leaky running sum along rows then columns (`s = s*leak + w`), or spectral synthesis (white noise in the FFT domain scaled by 1/f², then an inverse FFT).
**Fix, option 2:** rename it "Gaussian-blurred noise", remove `leaky_integrator`, and update the docs.

### 🟠 B6 — `convolve` wraps around incorrectly for sizes that aren't powers of two **[verified]**
`noise2d.h:898`: `(my + ky - half) % h` is computed in `size_t`. When `my + ky < half`, the value wraps to about 2⁶⁴, and `2⁶⁴ mod h` only equals the correct offset when `h` is a power of two.
In a test on a 3×3 matrix with a shifted delta kernel, the expected wrapped `1` **disappeared completely** (the output was all zeros). Every example uses power-of-two sizes, so this has never shown up.
**Fix:** `(my + ky + h - half) % h`, which assumes `half < h`; otherwise compute in signed arithmetic.

### 🟠 B7 — Perlin noise is constant at whole-number frequencies **[verified]**
With `x * frequency` a whole number, every sample lands exactly on a lattice point. There the distance vector is (0, 0), so every dot product, and the noise, is 0.

| frequency | distinct output values (32×32) |
|---|---|
| 0.2 | 129 |
| 1.0 | **1** |
| 2.0 | **1** |
| 3.0 | **1** |

The README says "Values greater than 1 increasingly resemble white noise". In fact 1, 2, 3… give a flat grey image. Fractal noise has the same problem: any octave whose frequency is a whole number adds nothing, e.g. `starting_frequency = 0.125` makes octaves 4–8 contribute nothing. As a related artifact, pixel (0, 0) is always exactly mid-grey.
**Fix:** add a sub-pixel offset when sampling (e.g. `(x + 0.5) * frequency`), or a per-octave offset. Document that frequency means cycles per pixel and that values ≥ 0.5 alias.

### 🟠 B8 — Fractal noise can go outside `[0, output_levels)`
`generate_fractal_noise` (`:485–504`) adds octaves with amplitudes 1, ½, ¼… (8 octaves sum to 1.99), but normalizes as if the range were `[-1, 1]`. The worst case for integer output is about `[-52, 308]` with 256 levels. In 30 random 64×64 runs the observed range was [16, 240], so it's rare in practice but not ruled out. The output also isn't scaled to use the full range.
**Fix:** divide by the sum of amplitudes (and by about 0.707, the real Perlin range; see S2), then clamp.

### 🟠 B9 — Fourier helper (examples): same non-square indexing bug, and `idft()` isn't an inverse
`src/fourier.cpp:73, 87, 97, 109, 121` and the `idft` equivalents use `in[y * height + x]`, which goes out of bounds for non-square input.
Other problems:
- `idft()` uses `FFTW_FORWARD` on magnitudes with the phase thrown away, so it doesn't invert anything.
- The DC-offset mean is accumulated in an `int`, truncating each `double`.
- `pow(-1, x + y)` is called once per element.

### 🟡 B10 — Smaller issues
- `generate_brown_noise`: `integrated_max` starts at `DBL_MIN`, which is the smallest **positive** double, not the most negative. That's harmless here only because the values are positive. Use `std::numeric_limits<double>::lowest()`. If `integrated_range == 0` (e.g. 1×1), the code divides by zero, and casting the resulting NaN to `T` is UB.
- `EnergyLUT::create` and `update` reset the min/max values but not the stored coordinates. If there are no candidates (all 0s or all 1s), the previous call's coordinate is used silently. With 0×0 that coordinate is `(-1, -1)`, which causes the crash in A7.
- `generate_white_noise` uses `std::uniform_int_distribution<>` (`int`). Converting `output_levels - 1` from `size_t` overflows for levels above `INT_MAX`. It also builds a new distribution for every pixel; build it once outside the loop.
- `__NOISE2D_H` (and `__FOURIER_H`, `__IMAGE_H`) use reserved identifiers (double underscore). Use `NOISE2D_H`.
- Math functions are called unqualified (`acos`, `exp`, `sin`, `cos`, `pow`), relying on `<cmath>` also putting names in the global namespace. Use `std::`. `std::pair` needs `<utility>`, and `fourier.cpp` uses `INT_MAX` without including `<climits>`.
- Comment typo: `std::mt19337`. The Makefile comment says `fractak_noise.exe`.

---

## 4. Gaps between the code and the README (spec)

| # | README says | Code does | Severity |
|---|---|---|---|
| S1 | "C++17 **header-only** library… `#include` in **any files**" | Links only from a single TU (B1) | 🔴 |
| S2 | "If noise is an integral type, then values range from [0, output_levels)", and implies the full range is used | Perlin uses only about [0.17, 0.81] of the range **[verified, double output over 512×512]**, because 2D Perlin with unit gradients peaks at ±0.707, not ±1. Fractal can go **outside** the range (B8). Brown hits `output_levels − 1` at exactly one pixel. | 🟠 |
| S3 | "If noise is a floating-point type, [output_levels] is not used" | `generate_brown_noise` uses it for floating-point `T`, so `Noise2D<double>` brown noise ranges over **[0, 255]** **[verified]** while every other generator gives [0, 1] | 🟠 |
| S4 | Brown noise: "first generating white noise, then integrating the result" (intro) **and** "convolving white noise with a Gaussian kernel" (API section) | The README contradicts itself. The code does a Gaussian blur, and `leaky_integrator` has no effect (B5) | 🟠 |
| S5 | Fractal noise: "Method to generate fractal noise **by convolving white noise with a Gaussian kernel**" | Copy-paste error: the code adds Perlin octaves | 🟡 |
| S6 | Perlin `frequency`: "Values greater than 1 increasingly resemble white noise" | Whole-number frequencies produce a constant image (B7) | 🟠 |
| S7 | Perlin and fractal: "Throws std::out_of_range if frequency is **less than** 0" | Throws for `<= 0`. The code matches the "must be greater than 0" sentence, so only the Throws table is wrong. `std::invalid_argument` would be the more natural exception. | 🟡 |
| S8 | `get_noise_at`: throws "if x **exceeds** width" | Throws for `x >= width`. The code is correct; the wording is off by one. | 🟡 |
| S9 | "Type T can be any integral or floating-point type" | Narrow types overflow with the default 256 levels (A6) | 🟡 |
| S10 | Width and height have no documented constraints | Non-square sizes crash blue noise (B2); a zero size crashes (A7); brown noise wraps incorrectly for sizes that aren't powers of two (B6) | 🔴 |
| S11 | `kernel_size` accepts any `size_t` | Even sizes and 0 are UB (B3) | 🔴 |
| S12 | Fractal noise is a stack of Perlin layers, and Perlin is repeatable from a seed | `generate_fractal_noise` has no `seed` parameter, and white, brown and blue noise can't be reproduced either (A5) | 🟡 |
| S13 | The README describes Noise2D as "the base class" | The class is `final` | 🟡 |
| S14 | README images are 256×256, and the README says each example "generates various sizes of noise matrixes" | Examples only go up to 64×64, so the README images can't be regenerated from the repo. At the current speed, 256² blue noise would take a very long time (§5). | 🟡 |
| S15 | Build instructions: `make blue_noise` etc. | Those targets work. Plain `make` (the default `main` target) is broken (§6). | 🟡 |
| S16 | Blue noise rank normalization comment: "max value is width * height, so this normalizes to 0–1" | The highest rank is `N − 1`, so floating-point output is [0, (N−1)/N] (e.g. max = 0.996 for 16×16). That's fine, but the comment and spec are inaccurate. | 🟡 |
| — | Typos in the README: "white nois fourier", "psuedo" (also in the header), "generated by layers Perlin noise", "matrixes" | | 🟡 |

---

## 5. Performance

Measured with g++ -O2 on Linux, in the cloud workspace; absolute times will differ on your machine, but the scaling holds.

| Generator | Size | Time |
|---|---|---|
| Blue | 16×16 | 2 ms |
| Blue | 32×32 | 22 ms |
| Blue | 64×64 | 369 ms |
| Perlin | 64×64 | **318 ms** |
| Fractal (8 octaves) | 64×64 | **2.5 s** |

- **Perlin and fractal are very slow** for what they do, at about 80 µs per pixel. `random_gradient_vector` builds a `std::seed_seq` and seeds a **full `std::mt19937` (624-word state)** for each of the 4 corners of every sample, so neighbouring pixels recompute the same corner gradients again and again. At that rate a 256×256 fractal would take about 40 s.
  **Fix:** use a 256-entry permutation table built once from the seed (which is what Perlin's Improved Noise actually does), or a cheap integer hash such as PCG or xxhash of `(seed, x, y)` mapped to an angle. Either should make it hundreds of times faster.
- **Blue noise is O(N²) in `EnergyLUT::create`** (every cell against every cell, with an `exp()` call each time) and O(N) per `update`. Since `update` runs for every pixel, the total is O(N²) as well, so time grows about 16× each time the side length doubles: 256² would take about 90 s, and 512² roughly 25 minutes.
  **Fix:** precompute a W×H table of the Gaussian by toroidal distance, which removes every `exp()` from the inner loop. Limit the kernel to about 3σ (a few dozen cells for σ = 1.9), so each update becomes O(1) instead of O(N). For large sizes, do the `create` step with an FFT convolution.
- Smaller items: the white-noise distribution is rebuilt for every pixel; nested vectors (A2); examples copy the whole object by value (A3).

---

## 6. Build system, examples, and repo hygiene

- **The default `make` target is broken.** `main` depends on `$(OBJ)`, built by the pattern rule `$(ODIR)/%.o : $(EDIR)/%.cpp $(SDIR)/%.cpp`. That rule needs a file with the same name in **both** `examples/` and `src/`, which never exists, so the rule can't match. Even if it did, linking several `example_*.o` files together would give multiple `main` definitions. Perlin and fractal are missing from `_OBJ` too. Remove the rule, or turn it into an `all:` target that builds the five examples.
- The named targets (`blue_noise` etc.) aren't `.PHONY`, and their output (`bin/blue_noise.exe`) never matches the target name, so every `make` recompiles `lodepng.cpp` (320 KB) from scratch. Build objects once into `obj/`, which the folder already exists for.
- Windows-only: `"output\\…"` paths, `rm -f` in `clean` (that needs a POSIX shell), and the prebuilt DLL. Use `std::filesystem::path("output") / name`. Consider CMake with `find_package(FFTW3)` so it builds on Linux and macOS.
- `libfftw3-3.dll` (2.7 MB) is committed **twice** (`bin/` and `lib/`). Keep one copy, or download it during the build. The FFTW license (GPL) also differs from this repo's MIT license; it only applies to the examples, but it's worth a note in the README.
- The examples are five near-identical 130-line files. A single `example.cpp` that takes the noise type as a command-line argument, or a shared helper, would remove about 500 duplicated lines. `sprintf` into fixed-size buffers should become `snprintf` or `std::to_string`.
- The Fourier images use linear magnitude. A log scale (`log(1 + |F|)`) is standard and would make the brown and fractal spectra much easier to read.
- **No tests, no CI.** Even a dozen assertions would have caught B1–B8: two-TU link, non-square sizes, output range, repeated calls, determinism with a seed, and frequency = 1.0.

---

## 7. Recommended fix order

1. **Quick fixes (minutes each):** `inline` on the two free functions (B1); `y * width + x` in `noise2d.h:527` and `fourier.cpp` (B2, B9); validate width, height and `kernel_size` (A7, B3); reset blue-noise state at the start of each call (B4); fix the `convolve` modulo (B6).
2. **Correctness:** add a half-pixel sample offset to Perlin (B7); normalize fractal by its total amplitude (B8); make the output range consistent for floating-point `T` in every generator (S2, S3).
3. **Spec alignment:** either implement real brown noise or rename it and drop `leaky_integrator` (B5, S4); fix the copy-paste and wording errors in the README (S5–S8, S13, S16, typos).
4. **API:** add seeds everywhere (A5); `static_assert` and range checks on `T` (A6); bulk data access (A3); move the class into a namespace (A9).
5. **Performance:** use a permutation table or hash for Perlin gradients; a precomputed, truncated Gaussian for blue noise (§5).
6. **Infrastructure:** fix the Makefile or move to CMake; add a small test suite, run with sanitizers, in CI; regenerate the README images from an example that actually produces 256×256.
