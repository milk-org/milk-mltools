# AGENTS.md — Developer Guide for `milk-mltools`

## Repository Role & Build Integration

- `milk-mltools` is a `milk` plugin repository linked into `/home/guyon/src/milk/plugins/milk-mltools-src`.
- **No root `CMakeLists.txt`**: `milk` discovers plugin subdirectories at `maxdepth 2` (`plugins/*/*/CMakeLists.txt`). Keep build definitions strictly inside [mltools/CMakeLists.txt](mltools/CMakeLists.txt) and [mltools_calc/CMakeLists.txt](mltools_calc/CMakeLists.txt).
- **CMake variable hygiene**: Never use `${PROJECT_SOURCE_DIR}` after `project(...)` to locate `milk` headers — always use `${CMAKE_SOURCE_DIR}/src/...` (with `${CMAKE_CURRENT_SOURCE_DIR}` for local sources).
- **Shared timing parser**: Do not duplicate `.txt` parsing code here; reuse `COREMOD_memory/tlm_timing.h` (`milk_tlm_read_file`, `milk_tlm_peek_span`, `milk_tlm_write_file`, `milk_tlm_free`).
- **CMake rebuild & isolated install**:
  ```bash
  cd /home/guyon/src/milk
  /home/guyon/.local/envs/cacao/bin/cmake --build _build_dev -- -j128
  /home/guyon/.local/envs/cacao/bin/cmake --install _build_dev --prefix "$PWD/_build_dev/_install"
  ```
  Never install into `/usr`, `/usr/local`, or modify `_build`.

## Testing & Verification

Always isolate `MILK_SHM_DIR`, `MILK_CALC_CACHE`, and `TMPDIR` per test run; never touch `/milk/shm`, `/dev/shm`, or write into `/data/sdata`.

- CTest (`-h1` smoke tests + `@calc` frontend checks):
  ```bash
  /home/guyon/.local/envs/cacao/bin/ctest --test-dir /home/guyon/src/milk/_build_dev \
      --output-on-failure -R "mlt-|mcalc-front"
  ```
- Python E2E suite (run outside `milk/python/tests` to avoid its broken `pyMilk` symlink):
  ```bash
  /home/guyon/.local/envs/cacao/bin/pytest -v /home/guyon/src/milk-mltools/tests/test_mltools_e2e.py
  ```
- Read-only real SCExAO benchmark:
  ```bash
  /home/guyon/.local/envs/cacao/bin/python /home/guyon/src/milk-mltools/tests/bench_sdata.py
  ```

## Coding & `milk` Conventions

- **License & formatting**: Every `.c`/`.h` source must begin with `// SPDX-FileCopyrightText:` and `// SPDX-License-Identifier: LGPL-3.0-or-later`, match `.clang-format`, stay `<= 100` chars per line, and compile with `-Wall -Wextra -Werror -Wformat=2 -Wformat-overflow -Wformat-truncation`.
- **FPS rules**:
  - String FPS parameters are capped at `FUNCTION_PARAMETER_STRMAXLEN - 1 == 63` bytes (`fps_types.h`). In `mltools/commands.d/mltools.bash`, always wrap user paths through `__mlt_short_path` (inputs/dirs) or `__mlt_short_outpath` (output files) before calling FPS units (`mlt.tlmload`, `mlt.tlmindex`, `mlt.tlmsave`, `iofits.*`).
  - Keep `EXPORT_FPSEXEC(milk, "mlt-<id>", runnable_mlt_<id>);` on a single line so `CMakeLists.txt` regex registers `milk-fpsexec-mlt-<id>` and CTest `-h1`.
  - FPS outputs written outside procinfo need `fpsresync_modvar_to_fps(dcfpsptr, dcfpsbindings, dcfpsnbindings);` after `*_run()`.
  - `COREMOD_memory`'s `image_read_keyword_L()` returns the keyword slot index (`>= 0`, or `-1` if absent), NOT `RETURN_SUCCESS` (`0`).
- **Tier 2 `@calc` extensions (`mltools_calc/mlt_calc_ext.c`)**:
  - Parameterized X-macros must not use an argument letter colliding with a port name (use `(M)` instead of `(X)` when a port is named `X`).
  - Tier 2 image inputs are promoted to `FLOAT` (`im->array.F`) before `_init` / `_exec`.
- **`@calc` expression quirks in `mltools/commands.d/mltools.bash`**:
  - `posi(x)` returns `(x > 0) ? 1 : 0` (step function, not ReLU — use `max(x, 0.0)` to clamp).
  - Slice suffixes `[a:b]` attach only to bare identifiers (`_tmp = expr; _tmp[*, 0:k-1]`).
  - `iofits.loadfits` takes 3 arguments (`<file> <imname> <errmode>`).
  - Clean up every `@L:` temporary image on both success and failure (`__mlt_rm_images`) and remove symlinks (`__mlt_cleanup`).
