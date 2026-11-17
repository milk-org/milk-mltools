# Usage Guide

`milk-mltools` provides three ways to run every tool: interactive `milk-cli` commands (`mlt.*`), OS shell wrappers (`milk-mlt-*`), and standalone FPS binaries (`milk-fpsexec-mlt-*`).

## 1. Checking Telemetry Directories (`mlt.tlmcheck` / `milk-mlt-check`)

Inspect cube counts, dropped `cnt0` frames, orphan `.fits`/`.txt` files, median FPS, jitter, `t_acq` availability, UT epoch (`MLTEPOCH`), and optionally two-stream overlap:

```bash
# OS CLI (single stream)
milk-mlt-check /data/sdata/20260101/iiwi iiwi

# Two-stream overlap + machine-readable key=value report
milk-mlt-check /data/sdata/20260101/dm00disp03 dm00disp03 \
    /data/sdata/20260101/ocam2d ocam2d /tmp/overlap.txt
```

Inside `milk-cli`:
```bash
mlt.tlmcheck /data/sdata/20260101/dm00disp03 dm00disp03 /data/sdata/20260101/ocam2d ocam2d
```

## 2. Loading, ROI/Masking, Syncing & Saving in `milk-cli`

### `mlt.tlmload`
Load paired `.fits`/`.fits.fz` + `.txt` telemetry in parallel into local (`@L:name`) or SHM memory alongside `<out>_t` `(6, N)` `DOUBLE` timing:

```bash
# Load 2000 frames starting at UT relative time 21048.0 s, binned 2x2 inside ROI + mask
mlt.tlmload .dir=/data/sdata/20260101/ocam2d .stream=ocam2d .out=@L:cam \
    .tstart=21048.0 .maxframes=2000 .roi=10:109,10:109 .bin=2 .mask=/tmp/pupil.fits
```

Supported time-bound formats for `.tstart` / `.tend`: `HH:MM:SS[.fff]`, `+seconds` (relative to start or `.tstart`), relative seconds since `00:00:00 UT`, or full Unix seconds. Timestamp source `.tsrc` accepts `auto` (prefers `t_acq`, falls back to `t_log` when `t_acq == 0`), `acq`, or `log`.

### `mlt.tlmsync`
Resample stream `B` onto reference timeline `A_t` at lag `.lag` seconds (`lag > 0` samples `B` at `t_A + lag`):

```bash
mlt.tlmsync cam_t dm dm_t dm_synced .lag=0.0009586 .mode=linear
@calc -v i0,i1,nv "s.dm_synced.kw.SYNCI0; s.dm_synced.kw.SYNCI1; s.dm_synced.kw.SYNCNV"
```

### `mlt.tlmsave`
Write a 2D `(P, N)` modal time series or 3D `(nx, ny, N)` cube together with `<in>_t` timing back to standard `.fits` + 7-column `.txt` files:

```bash
mlt.tlmsave dm_synced cam_t /tmp/out_tlm dm_synced
```

## 3. PCA & Modal Projection (`mlt.pca`, `mlt.pcaproj`, `milk-mlt-pca`)

```bash
milk-mlt-pca /data/sdata/20260101/ocam2d ocam2d /tmp/pca_out 50 auto 1 21048.0 +15s 0 1 /tmp/mask.fits
```

Arguments (`mlt.pca -h` prints full `@args` help):
`dir [stream] outdir [nmodes=50] [method=auto|cov|gram|rsvd] [demean=1] [tstart] [tend] [maxframes=0] [decim=1] [mask] [roi] [bin=1]`

Outputs written to `outdir/`:
- `modes.fits` — `(nx, ny, k)` spatial PCA modes (masked pixels zeroed if `mask` provided).
- `mean.fits` — `(nx, ny)` temporal mean frame.
- `eigval.txt` — Mode index, variance eigenvalue, fraction, cumulative fraction.
- `coeffs_<HH:MM:SS>.fits` + `.txt` — `(k, N)` modal coefficient telemetry pair.
- `pca_summary.txt` — Solver, dimensions, total variance, cumulative explained variance.

Project a subsequent window or another dataset onto those modes:
```bash
milk-cli -e -c "mlt.pcaproj /data/sdata/20260101/ocam2d ocam2d /tmp/pca_out/modes.fits /tmp/pca_out/mean.fits /tmp/proj_out 21100.0 +10s"
```

## 4. Two-Stream Latency (`mlt.latency` / `milk-mlt-latency`)

```bash
milk-mlt-latency /data/sdata/20260101/dm00disp03 dm00disp03 /data/sdata/20260101/ocam2d ocam2d \
    /tmp/lat_out 30.0 1.5 0.05 8 8 4 21048.059963 +30s
```

Key parameters:
`dira streama dirb streamb outdir [maxlag_ms=50] [fine_ms=2] [step_ms=0.05] [nmodes=8] [whiten_order=8] [nblocks=4] [tstart] [tend] [maxframes=0] [tsrc=auto]`

Outputs in `outdir/`:
- `latency_summary.txt` — `tau_sec`, `tau_ms`, `tau_coarse_ms`, `tau_frames_a`, `tau_err_ms`, `peak_corr`, `fwhm_ms`, `drift_ppm`, `fps_a`, `fps_b`, `n_overlap`, `sync_mode`.
- `latency_curve.fits` & `latency_curve.txt` — Continuous fine-scan subspace RMS cross-correlation vs lag (ms).

## 5. Regularized CCA (`mlt.cca` / `milk-mlt-cca`)

```bash
milk-mlt-cca /data/sdata/20260101/dm00disp03 dm00disp03 /data/sdata/20260101/ocam2d ocam2d \
    /tmp/cca_out 10 40 40 1e-3 1e-3 auto 0.7 21048.059963 +30s
```

Passing `lag=auto` estimates latency via `mlt.latency` first (saving `outdir/latency/`) and applies that lag when aligning `B` to `A`.

Outputs in `outdir/`:
- `cca_rho.txt` — Canonical correlations (`mode`, `rho_train`, `rho_test` on held-out tail, `rho_full`).
- `cca_A_maps.fits`, `cca_B_maps.fits` — Canonical weight filters $W_A$, $W_B$.
- `cca_A_patterns.fits`, `cca_B_patterns.fits` — Forward Haufe activation patterns $\Sigma_A W_A$, $\Sigma_B W_B$.
- `cca_U_*.fits/.txt`, `cca_V_*.fits/.txt` — Canonical variate telemetry pairs.
- `cca_summary.txt` — Sample counts, ridge settings, and leading correlations.

## 6. Welch PSD & Coherence (`mlt.psd`)

```bash
# Single stream PSD
milk-cli -e -c "mlt.psd /data/sdata/20260101/ocam2d ocam2d /tmp/psd_out 1024 0.5 '' '' 0.0 21048.0 +20s"

# Two-stream PSD + coherence + cross-phase (equal pixel count streams)
milk-cli -e -c "mlt.psd /tmp/synthA streamA /tmp/coh_out 512 0.5 /tmp/synthB streamB 0.002"
```

Outputs `psd.fits`, `freq.fits`, `psd_mean.txt`, plus `coherence.fits` and `phase.fits` in two-stream mode.

## 7. Standalone FPS Binaries

All four C commands also exist as standalone executables without `milk-cli`:
- `milk-fpsexec-mlt-tlmindex`
- `milk-fpsexec-mlt-tlmload`
- `milk-fpsexec-mlt-tlmsync`
- `milk-fpsexec-mlt-tlmsave`

Example:
```bash
milk-fpsexec-mlt-tlmload .dir=/tmp/sim .stream=streamA .out=@L:x .maxframes=500
```
