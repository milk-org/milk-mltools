# `mltools` — Core C FPS Units, Bash Workflows, and OS CLI Wrappers

## Files

- [mltools.h](mltools.h), [mltools.c](mltools.c): `milk` module registration (`MODULE_SHORTNAME_DEFAULT "mlt"`).
- [tlm_fileset.h](tlm_fileset.h), [tlm_fileset.c](tlm_fileset.c): Parallel directory scanner, cube+`.txt` pairing, orphan counter, relative/HH:MM:SS/Unix time-bound parser, and `mlt_load_timing_arg()` accepting either `<out>_t` images (with `MLTEPOCH`) or `.txt` files via `COREMOD_memory/tlm_timing.h`.
- [tlm_index_unit.c](tlm_index_unit.c): `mlt.tlmindex` / `milk-fpsexec-mlt-tlmindex` — multi-threaded `.txt` scan reporting cube/frame counts, orphans, `cnt0` drops, median FPS, MAD jitter, `has_acq`, and UT bounds.
- [tlm_load_unit.c](tlm_load_unit.c): `mlt.tlmload` / `milk-fpsexec-mlt-tlmload` — OpenMP CFITSIO cube reader (`.fits` and Rice `.fits.fz`) with ROI extraction, $b \times b$ binning, pupil masking, temporal decimation, and `(6, N)` `DOUBLE` timing matrix `<out>_t`.
- [tlm_sync_unit.c](tlm_sync_unit.c): `mlt.tlmsync` / `milk-fpsexec-mlt-tlmsync` — binary-search temporal resampler (`linear`, `zoh`, `nearest`, `window`) accounting for lag $\tau$ and day-epoch offset.
- [tlm_save_unit.c](tlm_save_unit.c): `mlt.tlmsave` / `milk-fpsexec-mlt-tlmsave` — FITS cube + 7-column `logshmim` ASCII `.txt` writer with UT/MJD/`MLTEPOCH` keywords.
- [commands.d/mltools.bash](commands.d/mltools.bash): High-level `@args` / `@calc` workflows (`mlt.tlmcheck`, `mlt.tlmsynth`, `mlt.pca`, `mlt.pcaproj`, `mlt.latency`, `mlt.cca`, `mlt.psd`).
- `scripts/milk-mlt-{check,pca,latency,cca}`: Thin OS executable wrappers around `milk-cli`.
