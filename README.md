# milk-mltools

Machine learning and time-series analysis plugin for [milk](https://github.com/milk-org/milk), operating directly on FITS / Rice-compressed `.fits.fz` telemetry cubes and companion ASCII `.txt` timing logs produced by `milk-logshim` (`@grab --timestamps` 4-column logs also supported).

Built in three thin layers that maximize reuse of existing `milk` infrastructure (`COREMOD_memory`, `COREMOD_iofits`, CFITSIO, OpenMP, OpenBLAS/LAPACKE/FFTW3, `@calc` JIT, and `@args` bash functions):

1. **Layer 1 — C telemetry I/O & synchronization (`mltools/`)**
   - Built into `libmilkmltools.so` and standalone `milk-fpsexec-mlt-{tlmindex,tlmload,tlmsync,tlmsave}` executables (no `libCLIcore` or `libmilkcalc` dependency).
   - Shared parser `milk_tlm_*` lives upstream in `COREMOD_memory/tlm_timing.{h,c}`.
2. **Layer 2 — `@calc` C extensions (`mltools_calc/`)**
   - `libmltools_calc.so` installed to `lib/milk/calc.d/` and auto-loaded at `@calc` startup.
   - Provides OpenMP-accelerated `mlt_xcov`, `mlt_arwhiten`, `mlt_arfilt1`, `mlt_peak1d`, `mlt_mask`, and `mlt_unmask` compiled into `@calc` JIT kernels alongside upstream `fft(im, axis)`, `ifft(im, axis)`, and `eigh(C, k)`.
3. **Layer 3 — High-level workflows & CLI (`mltools/commands.d/mltools.bash`, `mltools/scripts/milk-mlt-*`)**
   - Interactive `mlt.*` commands (`mlt.tlmcheck`, `mlt.tlmsynth`, `mlt.pca`, `mlt.pcaproj`, `mlt.latency`, `mlt.cca`, `mlt.psd`) plus OS shell wrappers `milk-mlt-{check,pca,latency,cca}`.

## Command & Extension Reference

| Layer | Name | Purpose |
|---|---|---|
| C FPS (`mlt.*` / `milk-fpsexec-mlt-*`) | `mlt.tlmindex` | Scan paired FITS+`.txt` cubes; report drops, orphans, frame rate, jitter, UT span |
| C FPS | `mlt.tlmload` | Multi-threaded CFITSIO loader with time slicing, ROI, binning, pupil mask, and `(6, N)` `DOUBLE` timing image `<out>_t` |
| C FPS | `mlt.tlmsync` | Align stream B onto stream A timestamps at lag $\tau$ (`linear`, `zoh`, `nearest`, `window`, `expint`) |
| C FPS | `mlt.tlmsave` | Write 2D `(P, N)` or 3D `(nx, ny, N)` image + `<in>_t` timing as `.fits` + 7-col `.txt` pair |
| C FPS | `mlt.aoopt` | Closed-loop AO modal Welch PSDs, empirical ETFs, PyWFS optical gains, and stability-constrained optimal modal gains |
| C FPS | `mlt.psfrecon` | Polychromatic AO PSF reconstruction (`hybrid`, `short_exp`, `otf`) from modal telemetry, DM modes, optical gains, and Kolmogorov fitting halos |
| C FPS | `mlt.sysid` | Joint 2-WFS + 1-DM linear system ID (`B = Mab A + Mdb D + b0` & virtual upstream `A = Mba B - Mda_virt D + a0`) via shared AR($p$) whitening & CV |
| `@calc` C op | `mlt_xcov` | Normalized subspace or raw cross-covariance across `[-maxlag, +maxlag]` |
| `@calc` C op | `mlt_arwhiten` | Fit pooled causal AR($p$) Levinson-Durbin filter on `X` and whiten both `X` and `Y` identically |
| `@calc` C op | `mlt_arfilt1` | Single-input causal AR($p$) whitening filter |
| `@calc` C op | `mlt_arfit` / `mlt_firfilt` | Fit pooled causal AR($p$) FIR taps `W(z)` on `X` and apply the identical FIR filter to arbitrary streams |
| `@calc` C op | `mlt_peak1d` | Parabolic sub-sample peak position, height, and FWHM on a 1D curve |
| `@calc` C op | `mlt_mask` / `mlt_unmask` | Fast 3D-to-`(P, K)` mask gather and `(P, K)`-to-3D scatter |
| Bash (`mlt.*`) / OS CLI (`milk-mlt-*`) | `mlt.tlmcheck` / `milk-mlt-check` | Single- or two-stream health check and overlap duration summary |
| Bash (`mlt.*`) | `mlt.tlmsynth` | Physics-grounded multi-rate, sub-frame lagged synthetic telemetry generator |
| Bash (`mlt.*`) / OS CLI (`milk-mlt-*`) | `mlt.pca` / `milk-mlt-pca` | Exact `cov`, `gram`, or randomized `rsvd` PCA + modal time-series export (`modes.fits`, `coeffs_*.fits/.txt`) |
| Bash (`mlt.*`) | `mlt.pcaproj` | Project any stream onto saved `modes.fits` + `mean.fits` |
| Bash (`mlt.*`) / OS CLI (`milk-mlt-*`) | `mlt.latency` / `milk-mlt-latency` | Modal PCA + AR($p$) whitening + coarse FFT/xcov + continuous timestamp scan (`lag > 0` $\iff B$ lags $A$) |
| Bash (`mlt.*`) / OS CLI (`milk-mlt-*`) | `mlt.cca` / `milk-mlt-cca` | Regularized PCA-subspace SVD-CCA with leak-free contiguous held-out temporal CV |
| Bash (`mlt.*`) | `mlt.psd` | Welch one-sided PSD and optional two-stream magnitude-squared coherence + cross-phase |
| Bash (`mlt.*`) / OS CLI (`milk-mlt-*`) | `mlt.aoloop` / `milk-mlt-aoopt` | Path-safe closed-loop modal PSD, optical gain, and optimal gain solver |
| Bash (`mlt.*`) / OS CLI (`milk-mlt-*`) | `mlt.psfr` / `milk-mlt-psfrecon` | Path-safe polychromatic PSF reconstruction and Strehl/profile benchmark |
| Bash (`mlt.*`) / OS CLI (`milk-mlt-*`) | `mlt.calib2wfs1dm` / `milk-mlt-calib2wfs1dm` | 3-stream multi-rate sync (`window` + `expint`), causal plant lag scan, and `mlt.sysid` (`Mab`, `Mdb`, `Mba`, `Mda_virt`) |
| Bash (`mlt.*`) / OS CLI (`milk-mlt-*`) | `mlt.aotlm` / `milk-mlt-aotlm` | End-to-end telemetry health, dual-stage AO optimization, PSFR, and optional 2-WFS/1-DM calibration (`TELEMETRY_REPORT.md`) |

## Quickstart

Symlink into `milk/plugins/milk-mltools-src` and build normally with `milk` (`CFITSIO` and `OpenBLAS/LAPACKE` enabled):

```bash
ln -s ../../milk-mltools /path/to/milk/plugins/milk-mltools-src
cmake -S /path/to/milk -B /path/to/milk/_build_dev -DCMAKE_BUILD_TYPE=Release
cmake --build /path/to/milk/_build_dev -- -j$(nproc)
cmake --install /path/to/milk/_build_dev --prefix /path/to/install
```

Inside `milk-cli` or from the OS shell (`milk-mlt-*`):

```bash
# Check two telemetry directories and find time overlap
milk-mlt-check /data/sdata/20260101/dm00disp03 dm00disp03 /data/sdata/20260101/ocam2d ocam2d

# Compute top-50 PCA modes on 30,000 frames (auto picks rsvd for P=14400)
milk-mlt-pca /data/sdata/20260101/ocam2d ocam2d /tmp/ocam_pca 50 auto 1 21048.0 "" 30000

# Measure DM -> OCAM2D AO loop latency on a 30s overlapping window
milk-mlt-latency /data/sdata/20260101/dm00disp03 dm00disp03 /data/sdata/20260101/ocam2d ocam2d \
    /tmp/dm_ocam_lat 25.0 1.5 0.05 8 8 4 21048.059963 +30s

# Compute regularized SVD-CCA with held-out temporal validation at lag=+0.9586 ms
milk-mlt-cca /data/sdata/20260101/dm00disp03 dm00disp03 /data/sdata/20260101/ocam2d ocam2d \
    /tmp/dm_ocam_cca 10 40 40 1e-3 1e-3 0.0009586 0.7 21048.059963 +30s
```

## Real SCExAO Benchmark (`/data/sdata/20260101`)

Measured on `/data/sdata/20260101` (`iiwi`, `dm00disp03`, `ocam2d` Rice-compressed `120x120x30000` `.fits.fz`) via [tests/bench_sdata.py](tests/bench_sdata.py):

| Stage | Dataset / Workload | Measured Result | Warm Wall Time |
|---|---|---|---|
| `mlt.tlmcheck` | `iiwi`: 1,875 cubes, 18,730,139 frames (`998.00 Hz`) | Detected 15 orphan `.txt`, `10,475,945` drops, `has_acq=1` | `0.492 s` (`10.881 s` cold) |
| `mlt.tlmcheck` | `dm00disp03` (1,786 cubes, 53.22M frames, `1979.38 Hz`, `has_acq=0`) vs `ocam2d` (1,896 cubes, 56.84M frames, `1996.01 Hz`, `has_acq=1`) | Overlap `38,072.50 s` (`[21043.060, 59115.557] s`) | `1.221 s` (`43.864 s` cold) |
| `mlt.pca` | `ocam2d` 30,000 frames `120x120` `.fits.fz` (`k=50`, `method=auto`/`rsvd`) | `11,091 frames/s` (`0.64 GB/s` float32), `explained_50=0.1000` | `2.705 s` (`21.335 s` forced `cov`) |
| `mlt.latency` | `dm00disp03` vs `ocam2d` (`30 s` window, `N=30,000`, `nmodes=8`, `whiten=8`) | `tau = +0.9586 +/- 0.0060 ms`, `peak_corr=0.3769`, `fwhm=0.692 ms`, `drift=1.19 ppm` | `5.810 s` |
| `mlt.cca` | `dm00disp03` vs `ocam2d` (`k=10`, `pca=40`, `train_frac=0.7` CV at `tau=+0.9586 ms`) | Held-out `rho_test` up to `0.6884` (vs `rho_train=0.946-0.984`) | `5.658 s` |

## Documentation & Tests

- [docs/algorithms.md](docs/algorithms.md) — Mathematical formulations, timing layout, lag sign convention, AR whitening, and CCA cross-validation.
- [docs/usage.md](docs/usage.md) — Detailed CLI usage, output files, ROI/binning/masking workflows, and standalone FPS tools.
- [mltools/README.md](mltools/README.md) — C FPS units and bash/OS wrappers.
- [mltools_calc/README.md](mltools_calc/README.md) — `@calc` C extension operators.
- [tests/test_mltools_e2e.py](tests/test_mltools_e2e.py) — Synthetic E2E verification against `numpy`/`scipy`.
- License: `LGPL-3.0-or-later` ([LICENSES/LGPL-3.0-or-later.txt](LICENSES/LGPL-3.0-or-later.txt)).
