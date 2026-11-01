# Noise2D Implementation Plan

**Based on:** `REVIEW.md` (31 Oct 2026)
**Goal:** Address every finding in the review: bugs B1–B10, design observations A1–A10, spec gaps S1–S16, the performance items in §5, and the build/repo items in §6.

The plan follows the fix order recommended in review §7, with one change: a minimal test harness comes first (Phase 0), so that each fix ships with a regression test that failed before the fix and passes after it.

---

## Overview

| Phase | Theme | Review items | Breaking API change? | Rough effort |
|---|---|---|---|---|
| 0 | Test harness | §6 (tests) | No | 1–2 h |
| 1 | Crash, UB and build fixes | B1, B2, B3, B4, B6, B9, A7, B10 (partial), S1, S10, S11 | No | 2–3 h |
| 2 | Output correctness | B7, B8, B10 (partial), S2, S3, S6, S16 | Output values change | 2–3 h |
| 3 | Brown noise and spec alignment | B5, S4, S5, S7, S8, S13, README typos | Yes (`generate_brown_noise` parameters) | 3–4 h |
| 4 | API cleanup | A1, A2, A3, A4, A5, A6, A8, A9, B10 (rest), S9, S12 | Yes (namespace, seeds) | 4–6 h |
| 5 | Performance | A10, §5 | No | 4–6 h |
| 6 | Build system, examples, CI | §6, S14, S15 | No (library) | 4–6 h |

Phases 0–2 can ship as a patch release (e.g. `v1.0.1`). Phases 3–4 change the public API and should be a single minor/major release (e.g. `v2.0.0`) so users only migrate once. Phases 5–6 can land before or after that release.

Suggested branch strategy: one branch and PR per phase, merged in order. Each PR must pass the test suite under ASan/UBSan before merging.

---

## Phase 0 — Test harness

**Why first:** The review notes that "even a dozen assertions would have caught B1–B8". Every later phase adds tests here.

### Tasks
1. Create `tests/` with:
   - `tests/test_main.cpp` — a tiny assert-style runner (a `CHECK(expr)` macro that records failures and a `main` that returns non-zero on any failure). No third-party framework, to keep the "standard library only" property. (If you prefer a framework, `doctest.h` is a single header and MIT-licensed.)
   - `tests/test_link_a.cpp` and `tests/test_link_b.cpp` — both `#include "noise2d.h"` and instantiate `Noise2D<int>` and call every generator. Linking them together is the B1 regression test.
2. Add a `test` target to the Makefile:
   ```make
   test:
   	$(CC) -std=c++17 -Wall -Wextra -Werror -g -fsanitize=address,undefined \
   	    -I$(IDIR) tests/*.cpp -o $(BDIR)/tests
   	$(BDIR)/tests
   ```
   (ASan is unavailable with MinGW on Windows; on Windows, run without `-fsanitize` locally and rely on Linux CI for sanitizer runs — see Phase 6.)
3. Add helper functions to the test file: `distinct_values(noise)`, `min_max(noise)`, and `equal(noise_a, noise_b)`.

### Done when
- `make test` builds and runs (it will fail at first because of B1; that's expected and is fixed in Phase 1).

---

## Phase 1 — Crash, UB and build fixes

All of these are small, local edits to `include/noise2d.h` and `src/fourier.cpp`.

### 1.1 Header-only linkage (B1, S1)
- Mark `noise2d::distance_vector` and `noise2d::random_gradient_vector` (`noise2d.h:985`, `:990`) as `inline`.
- Change `noise2d::PI` to `inline constexpr double PI = ...;`.
- Audit for any other non-template, non-`inline` definitions in the header (`grep` for functions in `namespace noise2d` and any `static` data members that aren't `inline`).
- **Test:** the two-TU link test from Phase 0 now builds.

### 1.2 Non-square indexing in blue noise (B2, S10)
- `noise2d.h:527`: change `index = (y * height) + x` to `index = (y * width) + x`.
- Search the rest of the header for `* height` used as a row stride and fix any other occurrence.
- **Tests:** generate blue noise at 16×8, 8×16, 1×7 and 7×1 under ASan; check the result is a permutation of ranks `0 … N−1` (for integer `T` with `output_levels = N`) or has `N` distinct values.

### 1.3 Non-square indexing in the Fourier helper (B9)
`src/fourier.cpp` (examples only):
- Change every `in[y * height + x]` (lines 73, 87, 97, 109, 121 and the `idft` equivalents) to `in[y * width + x]`.
- Accumulate the DC-offset mean in a `double`, not an `int`.
- Replace `pow(-1, x + y)` with `((x + y) & 1) ? -1.0 : 1.0`.
- Add `#include <climits>` (uses `INT_MAX`), or remove the `INT_MAX` use if no longer needed.
- `idft()`: either fix it properly (keep the complex spectrum, run `FFTW_BACKWARD`, divide by `N`) or delete it if nothing calls it. **Recommendation:** delete it unless an example needs it; a broken inverse is worse than none.

### 1.4 Validate dimensions (A7, B10 — EnergyLUT coordinates)
- In the `Noise2D` constructor, throw `std::invalid_argument` if `width == 0` or `height == 0`.
- Also throw if `width * height` would overflow `std::size_t`, and replace any `int` holding `width * height` (e.g. `num_pixels`, `rank` in phases 1–3, the `int index_remaining_coordinates` loop at `:534`) with `std::size_t` or `std::ptrdiff_t`.
- In `EnergyLUT::create` / `update`, reset the stored coordinates together with the min/max values, and `assert` (or throw `std::logic_error`) if no candidate was found instead of silently reusing the previous coordinate.
- **Tests:** constructing 0×N or N×0 throws; 1×1 blue noise works and returns a single value.

### 1.5 `gaussian_kernel` sizes (B3, S11)
- In `generate_brown_noise` (and in `gaussian_kernel` itself), throw `std::invalid_argument` if `kernel_size == 0` or `kernel_size` is even.
- Document the constraint ("odd, ≥ 1") in the README.
- (Phase 3 may remove this function entirely; fixing it now still makes the patch release safe.)
- **Test:** `kernel_size` of 0, 2 and 4 throws; 1, 3 and 5 work under ASan.

### 1.6 Blue noise state reuse (B4)
- At the start of `generate_blue_noise()`, clear `binary_pattern_initial`, `binary_pattern_prototype` and `blue_noise_rank_data` (fill with 0 / sentinel) and re-create the `EnergyLUT`.
- This is the quick fix; Phase 4 (A1) removes the member state entirely.
- **Test:** calling `generate_blue_noise()` twice on the same 16×16 object yields 256 distinct ranks both times.

### 1.7 `convolve` wrap-around (B6)
- `noise2d.h:898`: replace `(my + ky - half) % h` with `(my + ky + h - half) % h` (and the same for the `x` axis).
- Guard the `half >= h` case: either throw if `kernel_size > min(width, height)` or compute in signed arithmetic with a proper positive modulo: `((i % n) + n) % n`.
- **Test:** the review's 3×3 matrix with a shifted delta kernel; the wrapped `1` must appear in the expected cell. Also test 5×7.

### 1.8 Minor safety items from B10
- `generate_brown_noise`: initialise `integrated_max` with `std::numeric_limits<double>::lowest()` (not `DBL_MIN`), and `integrated_min` with `max()`.
- If `integrated_range == 0`, write the mid-value (or 0) instead of dividing by zero.
- **Test:** 1×1 brown noise doesn't trigger UBSan.

### Done when
- `make test` passes under ASan/UBSan, including all new tests above.
- All five examples still build and produce the same images for square sizes (blue noise output will differ only by randomness).

---

## Phase 2 — Output correctness

### 2.1 Define the output-range contract (S2, S3, S16)
Write down one rule and apply it to every generator:

> **Integral `T`:** values are in `[0, output_levels − 1]` and generators should use the full range.
> **Floating-point `T`:** values are in `[0, 1]`; `output_levels` is ignored.

Implement it once as a private helper, e.g.
```cpp
T quantize(double v01) const; // v01 in [0,1] → T, clamped
```
and route every generator's final write through it. This removes five copies of the `if constexpr(std::is_integral<T>::value)` block.

- **Brown noise (S3):** stop using `output_levels` for floating-point `T` (currently gives `[0, 255]`).
- **Blue noise (S16):** float output is `rank / (N − 1)` (range `[0, 1]`) for consistency with the contract; guard `N == 1`.
- **Perlin (S2):** scale the raw value by `1 / 0.70710678` (the true 2D Perlin bound with unit gradients) before mapping `[-1, 1] → [0, 1]`, then clamp.

**Tests:** for each generator and for `T` = `unsigned char`, `int`, `float` and `double`, check `min >= 0` and `max <= upper bound`; for float `T`, check `max <= 1.0`.

### 2.2 Perlin at whole-number frequencies (B7, S6)
- Sample at `(x + 0.5) * frequency`, `(y + 0.5) * frequency`.
- For fractal noise, also add a per-octave offset (e.g. derived from the seed and octave index) so octaves don't share lattice alignment.
- Document in the README that `frequency` is in cycles per pixel and that values ≥ 0.5 alias (Nyquist). Replace the sentence "Values greater than 1 increasingly resemble white noise".
- **Tests:** at frequencies 1.0, 2.0, 3.0 on 32×32, the output has more than one distinct value; pixel (0, 0) is not always the mid-value across seeds.

### 2.3 Fractal range (B8)
- Divide the octave sum by the total amplitude `Σ 0.5^i` (and apply the Perlin 0.707 scale from 2.1), then clamp via `quantize`.
- **Test:** across many seeds at 64×64, all integer values are in `[0, output_levels − 1]`; for float `T` in `[0, 1]`.

### Done when
- All range tests pass for every generator and type.
- README images are regenerated (or a note is added that they will be regenerated in Phase 6).

---

## Phase 3 — Brown noise and spec alignment

### 3.1 Decide what brown noise is (B5, S4) — **decided: Option A′ with a 1/f² power spectrum, see Decisions**
The README's intro says "integrate white noise"; the code does a Gaussian blur, and `leaky_integrator` has no effect.

| Option | Description | Pros | Cons |
|---|---|---|---|
| **A (recommended)** | Real brown noise by leaky integration: white noise → leaky running sum along each row (`s = s * leak + w`), then along each column of that result → normalize via `quantize`. | Matches the README intro and the name; `leaky_integrator` becomes meaningful; still standard-library only. | Not perfectly isotropic; not tileable. |
| A′ | Spectral synthesis: white noise in the frequency domain scaled by 1/f², inverse FFT. | True 1/f² spectrum, isotropic, tileable. | Needs an FFT; the library would either depend on FFTW (breaks "std only") or ship its own small FFT. |
| B | Rename to "Gaussian-blurred noise", drop `leaky_integrator`. | Smallest change. | Loses the brown-noise feature the README advertises. |

If **Option A** is chosen:
- New signature: `generate_brown_noise(double leaky_integrator = 0.999 /*, optional seed — see Phase 4 */)`.
- Remove the `kernel_size` and `sigma` parameters. If you want to keep a blur, expose it separately as `generate_blurred_noise(kernel_size, sigma)` and keep `gaussian_kernel`/`convolve` for it; otherwise delete both helpers (they then no longer need the Phase 1 fixes beyond the release they shipped in).
- Validate `0 < leaky_integrator <= 1`; throw `std::invalid_argument` otherwise.
- **Tests:** different `leaky_integrator` values produce different outputs (proves the parameter matters); the radially averaged power spectrum of a 128×128 result falls off steeper than white noise (a simple slope check on a log-log fit, computed with a naive DFT in the test to avoid an FFTW dependency).

### 3.2 README corrections
| Item | Change |
|---|---|
| S4 | Make the intro and API section describe the same brown-noise algorithm (whichever option is chosen). |
| S5 | Fractal noise API text: replace the copy-pasted "convolving white noise with a Gaussian kernel" with "sum of `octaves` layers of Perlin noise, each at double the frequency and half the amplitude of the previous". |
| S7 | Perlin/fractal Throws table: "if `frequency <= 0`". Switch the exception to `std::invalid_argument` (and update the code). |
| S8 | `get_noise_at`: "if `x >= width` or `y >= height`". |
| S13 | Replace "base class" with "class" (it's `final`). |
| S2/S3/S16 | Document the output-range contract from Phase 2.1. |
| S6 | Document the frequency semantics from Phase 2.2. |
| S10/S11 | Document the width/height and kernel constraints from Phase 1. |
| Typos | "white nois fourier" → "white noise Fourier"; "psuedo" → "pseudo" (README and header); "generated by layers Perlin noise" → "generated by layering Perlin noise"; "matrixes" → "matrices". |

### Done when
- Brown-noise tests pass and the README describes exactly what the code does.
- A reviewer can read each README API entry and find no mismatch with the code.

---

## Phase 4 — API cleanup

These changes break source compatibility, so land them together in one release with a short "Migrating from 1.x" section in the README.

### 4.1 Namespace (A9)
- Move `Noise2D` into `namespace noise2d`; move helpers (`distance_vector`, `random_gradient_vector`, `gaussian_kernel`, `convolve`, interpolation, etc.) into `noise2d::detail`.
- Optional migration aid: `#ifdef NOISE2D_GLOBAL_ALIAS` → `using Noise2D = noise2d::Noise2D;` (opt-in, so the default doesn't reintroduce the name-clash problem).

### 4.2 Seeding (A5, S12)
- Add a `std::mt19937_64 rng` member, seeded from an optional constructor argument; default-seeded from `std::random_device` only when no seed is given.
- Add `void seed(std::uint64_t)` to reseed between calls.
- Every generator draws from `rng`. Perlin drops its `seed` parameter in favour of the member seed, and every generator takes an optional trailing `seed` (see Decisions). Mention it in the migration notes.
- Fractal noise gets reproducibility automatically.
- Fix the `std::mt19337` comment typo while here (B10).
- **Tests:** same seed → identical output for every generator; different seeds → different output.

### 4.3 Constrain `T` (A6, S9)
- `static_assert(std::is_arithmetic_v<T> && !std::is_same_v<T, bool>, "...")`.
- In the constructor, for integral `T`, throw `std::invalid_argument` if `output_levels < 2` or `output_levels - 1 > std::numeric_limits<T>::max()`.
- Update the README sentence "Type T can be any integral or floating-point type" to state these constraints.
- **Tests:** `Noise2D<signed char>(8, 8, 256)` throws; `Noise2D<signed char>(8, 8, 128)` works and yields non-negative values.

### 4.4 Storage and bulk access (A2, A3)
- Replace `std::vector<std::vector<T>>` with a flat `std::vector<T> data_` indexed `y * width + x`. Do the same for internal scratch matrices.
- Add `const T* data() const noexcept`, `const std::vector<T>& values() const noexcept`, `width()`, `height()`.
- Keep `get_noise_at(x, y)` (bounds-checked) and add an unchecked `operator()(x, y)` if useful.
- Change the examples to take `const Noise2D<int>&` instead of copying by value.

### 4.5 Blue-noise scratch state (A1)
- Move `blue_noise_rank_data`, `binary_pattern_initial`, `binary_pattern_prototype` and the `EnergyLUT` out of the class members and into locals (or a `detail::VoidAndCluster` helper) constructed inside `generate_blue_noise()`.
- This makes the Phase 1.6 reset unnecessary; delete it.
- Constructor then allocates only the output buffer.

### 4.6 Remove dead code (A8)
- Delete `coverage` (member), `interpolate_linear`, `interpolate_cubic` (if still unused after Phase 2), `noise2d::clamp` (use `std::clamp`), and the duplicate `NOISE2D_PI` (keep one `inline constexpr` constant in `detail`).

### 4.7 Class role and docs (A4)
- Keep the class (it's convenient), but describe it in the README as "a 2D matrix of noise values with methods that fill it". Remove the "base class" wording (done in Phase 3).
- Optionally add free-function wrappers, e.g. `noise2d::make_blue_noise<T>(w, h, levels, seed)`, which return a filled object.

### 4.8 Remaining hygiene (B10)
- `generate_white_noise`: create the distribution once outside the loop; use `std::uniform_int_distribution<std::uint64_t>` (or `T`-appropriate type) so `output_levels > INT_MAX` doesn't overflow. For float `T`, use `std::uniform_real_distribution<T>(0, 1)`.
- Rename include guards `__NOISE2D_H`, `__FOURIER_H`, `__IMAGE_H` → `NOISE2D_H`, `NOISE2D_FOURIER_H`, `NOISE2D_IMAGE_H`.
- Qualify math calls with `std::` (`std::acos`, `std::exp`, `std::sin`, `std::cos`, `std::pow`); add `#include <utility>`, `<limits>` and `<cstdint>`; drop `<cfloat>` once `DBL_MIN`/`DBL_MAX` are gone.

### Done when
- All tests pass, including the new seeding and type-constraint tests.
- Examples compile against the new API.
- README has a "Migrating from 1.x" section listing: namespace, seed handling, brown-noise parameters, exception type changes, float output ranges.

---

## Phase 5 — Performance (A10, §5)

Record baseline timings before starting (review §5 table) using a small `bench/bench.cpp` built with `-O2`, and record new timings after each step.

### 5.1 Perlin gradient lookup
- Replace `random_gradient_vector` (which seeds a full `std::mt19937` per corner per sample) with a 512-entry permutation table built once per `generate_*` call from `rng` (Perlin's Improved Noise approach), mapping `perm[perm[x & 255] + (y & 255)]` to one of 8–16 fixed unit gradients or to an angle LUT.
- Alternative: a cheap integer hash (e.g. PCG-style or xxhash-style mix of `(seed, x, y)`) → angle. Use this if you need lattices larger than 256 without repetition.
- **Target:** 64×64 Perlin from ~318 ms to < 2 ms; 8-octave fractal from ~2.5 s to < 20 ms.
- **Tests:** existing determinism and range tests still pass. (Exact outputs will change; that's expected and part of the 2.0 release.)

### 5.2 Blue noise energy computation
1. Precompute a W×H table `g[dy][dx] = exp(-(dx² + dy²) / (2σ²))` using toroidal distances once per call. Removes all `exp()` calls from inner loops.
2. Truncate the kernel to radius `r = ceil(3σ)` (≈ 6 for σ = 1.9), so `EnergyLUT::update` touches `(2r+1)²` cells: O(1) per update instead of O(N).
3. `EnergyLUT::create`: with the truncated kernel this becomes O(N·r²). Only if sizes ≥ 512² are needed, add an optional FFT-based path (not required now).
4. Finding the min/max energy is still O(N) per step; that's acceptable for now (total O(N²) with a very small constant). If needed later, track candidates with a heap keyed on energy.
- **Target:** 64×64 from ~369 ms to < 50 ms; 256×256 in a few seconds.
- **Tests:** add a test that compares the truncated version against the exact version on 16×16 with a fixed seed: rank maps should be identical or differ only in tie-breaking; at minimum the result must still be a full permutation and pass a basic blue-noise spectrum check (low energy near DC).

### 5.3 Smaller items
- White-noise distribution built once (done in 4.8).
- Flat storage (done in 4.4).
- Examples pass by `const&` (done in 4.4).

### Done when
- `bench` results are recorded in the PR description, and 256×256 blue, Perlin and fractal noise each run in a few seconds or less.

---

## Phase 6 — Build system, examples, CI (§6, S14, S15)

### 6.1 Makefile (S15)
- Remove the broken pattern rule and the `main` target.
- Add `all: blue_noise brown_noise white_noise perlin_noise fractal_noise` as the default target.
- Build shared objects once: `$(ODIR)/lodepng.o`, `$(ODIR)/fourier.o`, `$(ODIR)/image.o` via a correct pattern rule `$(ODIR)/%.o: $(SDIR)/%.cpp $(DEPS)`; each example links against them. This stops recompiling the 320 KB `lodepng.cpp` every time.
- Make each target produce a file that matches its name (e.g. target `$(BDIR)/blue_noise$(EXE)`), with phony convenience aliases listed under `.PHONY`.
- Add `test` (Phase 0) and `bench` (Phase 5) targets.
- Fix the `fractak_noise.exe` comment typo.
- Make `clean` portable: detect `OS=Windows_NT` and use `del` there, or document that a POSIX shell (MSYS2/Git Bash) is required.

### 6.2 CMake (alongside the Makefile)
- Add `CMakeLists.txt`:
  - `noise2d` as an `INTERFACE` library (header-only) with `target_compile_features(cxx_std_17)`.
  - Examples behind `option(NOISE2D_BUILD_EXAMPLES ON)`, using `find_package(FFTW3)` or `pkg_check_modules(FFTW3 fftw3)`, so they build on Linux/macOS.
  - Tests behind `option(NOISE2D_BUILD_TESTS ON)` and registered with `ctest`.
- Keep the Makefile for existing Windows users.

### 6.3 Examples
- Replace the five near-identical files with one `examples/example.cpp` taking the noise type and sizes as command-line arguments (`example blue 16 32 64 256`), or keep five thin `main`s that call a shared `examples/common.h`. Removes ~500 duplicated lines.
- Use `std::filesystem::path("output") / name` instead of `"output\\..."`; create the directory with `std::filesystem::create_directories`.
- Replace `sprintf` into fixed buffers with `std::to_string` / `std::ostringstream` (or `snprintf` at minimum).
- Plot Fourier magnitudes on a log scale (`std::log1p(|F|)`) before normalizing.
- Include 256×256 in the default sizes so the README images can be regenerated from the repo (S14) — practical after Phase 5.
- Regenerate all `docs/*.png` and update the README sentence about "various sizes".

### 6.4 Repo hygiene
- Stop committing `libfftw3-3.dll` (both copies) and document how to get FFTW (MSYS2 `pacman -S mingw-w64-ucrt-x86_64-fftw`, or the system package manager on Linux/macOS).
- Add a README "Licensing" note: the library is MIT; the examples link FFTW, which is GPL-licensed, so distributed example binaries fall under the GPL.
- Add `output/`, `obj/*.o`, `bin/tests*`, `build/` to `.gitignore` as appropriate.

### 6.5 Continuous integration
- Add `.github/workflows/ci.yml`:
  - **ubuntu-latest, g++ and clang++:** build tests with `-Wall -Wextra -Werror -fsanitize=address,undefined`, run them.
  - **windows-latest, MinGW (MSYS2):** build tests and all examples (no sanitizers).
  - Optionally **macos-latest, clang++**.
- Add a CI badge to the README.

### Done when
- `make` with no arguments builds all five examples (or `cmake --build` does).
- CI is green on all platforms.
- README images are regenerated from the examples at 256×256.

---

## Traceability matrix

Every review item and where it is addressed.

| Item | Summary | Phase / section |
|---|---|---|
| A1 | Blue-noise scratch state in class | 1.6 (quick reset), 4.5 |
| A2 | Nested vectors | 4.4 |
| A3 | No bulk access; examples copy by value | 4.4 |
| A4 | Class is "matrix + fill functions" | 4.7, 3.2 (S13) |
| A5 | Inconsistent seeding | 4.2 |
| A6 | `T` unconstrained | 4.3 |
| A7 | Width/height unvalidated; `int` overflow | 1.4 |
| A8 | Dead code | 4.6 |
| A9 | Global namespace | 4.1 |
| A10 | Slow generation | 5.1, 5.2 |
| B1 | Not header-only | 1.1 |
| B2 | Blue noise non-square OOB | 1.2 |
| B3 | `gaussian_kernel` even/zero sizes | 1.5 |
| B4 | Blue noise repeated calls | 1.6, 4.5 |
| B5 | Brown noise isn't brown; `leaky_integrator` no-op | 3.1 |
| B6 | `convolve` modulo | 1.7 |
| B7 | Perlin constant at whole frequencies | 2.2 |
| B8 | Fractal out of range | 2.3 |
| B9 | Fourier helper indexing, `idft`, int mean, `pow` | 1.3 |
| B10 | `DBL_MIN`, divide by zero | 1.8 |
| B10 | EnergyLUT stale coordinates | 1.4 |
| B10 | White-noise distribution type/rebuild | 4.8 |
| B10 | Reserved include guards | 4.8 |
| B10 | Unqualified math, missing includes | 4.8, 1.3 (`<climits>`) |
| B10 | `mt19337` / `fractak` typos | 4.2, 6.1 |
| S1 | Header-only claim | 1.1 |
| S2 | Output range not full / exceeded | 2.1, 2.3 |
| S3 | Brown float range uses `output_levels` | 2.1 |
| S4 | Brown noise README contradiction | 3.1, 3.2 |
| S5 | Fractal description copy-paste | 3.2 |
| S6 | Perlin frequency description | 2.2, 3.2 |
| S7 | Throws `<= 0` wording / exception type | 3.2 |
| S8 | `get_noise_at` wording | 3.2 |
| S9 | "Any integral type" | 4.3 |
| S10 | Undocumented width/height constraints | 1.2, 1.4, 1.7, 3.2 |
| S11 | `kernel_size` constraint | 1.5, 3.2 |
| S12 | Fractal / others not reproducible | 4.2 |
| S13 | "Base class" wording | 3.2 |
| S14 | README images not reproducible | 6.3 |
| S15 | Default `make` broken | 6.1 |
| S16 | Blue noise normalization comment | 2.1 |
| — | README typos | 3.2 |
| §5 | Perlin gradient cost | 5.1 |
| §5 | Blue noise O(N²) with `exp()` | 5.2 |
| §5 | Smaller perf items | 5.3 |
| §6 | Broken default target / pattern rule | 6.1 |
| §6 | Non-phony targets, lodepng rebuilds | 6.1 |
| §6 | Windows-only paths / `rm -f` / DLL | 6.1, 6.2, 6.3 |
| §6 | DLL committed twice; GPL note | 6.4 |
| §6 | Duplicated examples, `sprintf` | 6.3 |
| §6 | Linear Fourier magnitude | 6.3 |
| §6 | No tests / CI | 0, 6.5 |

---

## Decisions

1. **Brown noise (Phase 3.1):** Option A′, spectral synthesis with a 1/f² **power** spectrum (amplitude scaled by 1/f). `leaky_integrator`, `kernel_size` and `sigma` are removed, along with `gaussian_kernel` and `convolve`. The library ships its own small FFT so it stays standard-library only.
2. **Perlin `seed` parameter (Phase 4.2):** dropped in favour of the object-level seed. Every `generate_*` method also takes an optional trailing `seed` that reseeds the object before generating, for reproducibility.
3. **Blue noise float range (Phase 2.1):** `rank / (N − 1)`, the full `[0, 1]`.
4. **Build system (Phase 6.2):** add CMake alongside the Makefile.
5. **FFTW DLL (Phase 6.4):** stop committing it and document installation.

## Implementation notes

Places where the implementation departs from the text above, and why:

- **Perlin sampling (2.2):** a half-pixel offset alone leaves even whole-number frequencies flat, since `(x + 0.5) * 2` is a whole number. Samples also get a random per-call (per-octave for fractal) lattice offset whose fractional part is in `[0.25, 0.75)`.
- **Perlin gradients (5.1):** uses the hash alternative, so the lattice does not repeat at 256 cells.
- **Blue noise kernel (5.2):** truncating at 3σ on its own made sparse cells tie on energy and break ties by scan order, which worsened the lowest and highest ranks and added a top-to-bottom bias at low sigma. The full Gaussian is used while the pattern is sparse, which brought those ranks level with the exact kernel at no measurable cost.
- **Blue noise initial pattern:** the swap loop's limit is now the number of cells rather than 100, which would stop too early at large sizes.
- **`gaussian_kernel` / `convolve` (1.5, 1.7):** deleted by Decision 1, so the fixes and tests for them are not needed. No `generate_blurred_noise` was added.
- **Validation added beyond the plan:** `sigma` must be positive and finite, and frequencies too large to sample accurately throw.
- **Compile-time migration guard:** old calls such as `generate_brown_noise(0.999)` and `generate_perlin_noise(1, 0.1)` would otherwise compile with a different meaning, because a `double` converts to the optional seed. Deleted overloads reject them.
- **Vendored `include/fftw3.h`:** removed with the DLLs, since installing FFTW provides a matching header.

Benchmark (`g++ -O2`, this repo's Windows machine, 64×64 unless noted):

| Generator | Before | After |
|---|---|---|
| Perlin | 117 ms | 0.07 ms |
| Fractal (8 octaves) | 936 ms | 0.33 ms |
| Blue | 146 ms | 7.5 ms |
| Brown | 0.17 ms (a blur, not brown noise) | 0.45 ms |
| Blue, 256×256 | not measured | 4.4 s |
| Fractal (8 octaves), 256×256 | not measured | 5.2 ms |
