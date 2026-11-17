# `mltools_calc` — C Extension Library for `@calc`

Compiles `mlt_calc_ext.c` into `libmltools_calc.so` and installs it to `lib/milk/calc.d/libmltools_calc.so`, where `milk`'s `@calc` frontend auto-loads it on startup via `mcalc_ext_autoload()`. Links only against `libm`, `libgomp`, and `libc` (via `milk::calc_ext` ABI).

## Exported `@calc` Operators

- `mlt_xcov(A, B, maxlag=50, norm=1)`: Multi-channel OpenMP cross-covariance / normalized subspace RMS correlation curve across integer sample lags `[-maxlag, +maxlag]`. When `norm=1`, normalizes squared Frobenius correlation by `min(ka, kb)` so identical subspaces peak at `1.0`.
- `Xw, Yw = mlt_arwhiten(X, Y, order=10, ridge=1e-5)`: Fits pooled Toeplitz Yule-Walker AR($p$) coefficients via Levinson-Durbin on `X` and applies that identical causal FIR whitening filter to both `X` and `Y`.
- `mlt_arfilt1(X, order=10, ridge=1e-5)`: Single-input Tier 1c causal AR($p$) whitening operator.
- `pos, val, fwhm = mlt_peak1d(curve, x0=0.0, dx=1.0)`: Sub-sample quadratic peak localization, peak value, and interpolated FWHM in physical coordinates.
- `mlt_mask(cube, mask)`: Gathers active mask pixels (`mask > 0.5`) from a 2D/3D image into a 2D `(P, K)` matrix.
- `mlt_unmask(mat, mask)`: Scatters a 2D `(P, K)` matrix back into `(nx, ny, K)` (or 2D `(nx, ny)` when `K == 1`).
