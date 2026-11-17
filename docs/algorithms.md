# Algorithms & Conventions

## 1. Hybrid Telemetry Timing & Epoch Representation

On disk, telemetry cubes use 7-column `logshmim` ASCII timing files (`cnt0 cnt1 t_log_sec_f epoch_sec t_acq_sec_f frame_idx main_idx`) or 4-column `@grab --timestamps` files, parsed by upstream `COREMOD_memory/tlm_timing.{h,c}` using integer nanosecond arithmetic (`int64_t`) so no sub-microsecond precision is lost in `~1.77e9 s` Unix timestamps.

Inside `milk`, `mlt.tlmload` exposes timing as a `(6, N)` `DOUBLE` image `<out>_t` paired with a `long` FITS-style keyword `MLTEPOCH` storing UT day `00:00:00` in integer Unix seconds (`t_epoch_sec`):

| Row (`y`) | Quantity | Units |
|---|---|---|
| `0` | Effective timestamp `t_eff - t_epoch_sec` (`t_acq` if available and `tsrc != log`, else `t_log`) | `s` |
| `1` | Log timestamp `t_log - t_epoch_sec` | `s` |
| `2` | Acquisition timestamp `t_acq - t_epoch_sec` (`0.0` if missing) | `s` |
| `3` | `cnt0` hardware/stream counter | count |
| `4` | `cnt1` secondary counter | count |
| `5` | `main_idx` cube frame index | index |

Subtracting `MLTEPOCH` keeps relative timestamps around `0..86400 s` with ~`1e-11 s` IEEE-754 `double` resolution while `t_epoch_sec` preserves exact cross-stream and cross-midnight alignment (`epoch_a - epoch_b` added in `mlt.tlmsync`).

## 2. Temporal Alignment & Lag Sign Convention

Across `mlt.tlmsynth`, `mlt.tlmsync`, `mlt.latency`, and `mlt.cca`, the scalar lag $\tau$ satisfies:

> $\tau > 0$ means **stream $B$ lags behind stream $A$**: physical events arrive in $A$ first and appear in $B$ roughly $\tau$ seconds later, i.e. $B(t + \tau) \approx A(t)$ or $B(t) \approx A(t - \tau)$.

Consequently, `mlt.tlmsync` resamples $B$ onto $A$'s timestamps $t_{A,i}$ by evaluating $B$ at query times:
$$t_{\text{query}, i} = t_{A, i} + \tau + (\text{epoch}_A - \text{epoch}_B)$$

Supported interpolation modes in `tlm_sync_unit.c`:
- `linear`: causal binary search + linear interpolation between bracketing samples of $B$ (default when $f_B \lesssim 1.3 f_A$).
- `zoh`: previous sample (zero-order hold) with $t_{B,j} \le t_{\text{query},i}$.
- `nearest`: closest timestamp in $B$.
- `window`: anti-aliased boxcar mean of $B$ samples falling inside $[t_{\text{query},i} - \Delta t_A/2,\; t_{\text{query},i} + \Delta t_A/2]$ (auto-selected when $f_B > 1.3 f_A$ to avoid aliasing high-rate telemetry downsampled to a slower camera).

Frames where $t_{\text{query},i} \notin [t_{B,\min}, t_{B,\max}]$ are zeroed and the contiguous valid index range `[SYNCI0, SYNCI1]` (`SYNCNV` frames) is recorded as keywords.

## 3. Principal Component Analysis (`mlt.pca` & `mlt.pcaproj`)

Given column-major frames $X \in \mathbb{R}^{P \times N}$ (extracted ROI/binned/pupil-masked into `P` active pixels):
1. Subtract temporal mean $\mu = \frac{1}{N} \sum_t x_t$ when `demean=1`: $X_c = X - \mu \mathbf{1}^T$.
2. Choose solver (`method=auto` selects `gram` if $N < P$ and $N \le 4096$, `cov` if $P \le 8192$, else `rsvd`):
   - **`cov`**: eigendecompose $C = \frac{1}{N-1} X_c X_c^T \in \mathbb{R}^{P \times P}$ top-$k$ pairs via LAPACK `LAPACKE_dsyevr` (`@calc` `eigh(C, k)`).
   - **`gram`**: eigendecompose $G = \frac{1}{N-1} X_c^T X_c \in \mathbb{R}^{N \times N}$ top-$k$ via `eigh(G, k)`, recovering spatial modes $U = X_c V \Sigma^{-1} / \sqrt{N-1}$.
   - **`rsvd`**: Halko-Martinsson-Tropp randomized subspace iteration with oversampling $k+16$ and one power iteration $Y = X_c (X_c^T Q)$ for fast $O(PNk)$ top-$k$ decomposition on large pixel counts (e.g. `120x120`).
3. Canonicalize eigenvector signs via weighted skewness `itot(U^3 * ramp, 0) >= 0` and compute modal coefficients $A = U^T X_c \in \mathbb{R}^{k \times N}$.

`mlt.pcaproj` loads saved `modes.fits` + `mean.fits`, applies the matching mask/ROI/binning, and outputs $A = U^T (X - \mu)$ plus residual RMS and explained variance fraction.

## 4. Sub-Frame Time Latency (`mlt.latency`)

Closed-loop AO telemetry exhibits strong temporal autocorrelation ($1/f^\alpha$ turbulence, AR processes), which broadens naive cross-correlation peaks and biases discrete lag estimates. `mlt.latency` combines four stages:

1. **Spatial Compression**: Reduce both $A$ and $B$ to compact $k$-mode temporal matrices $A_m \in \mathbb{R}^{k_A \times N_A}$, $B_m \in \mathbb{R}^{k_B \times N_B}$ (`eigh` or `rsvd`).
2. **Causal AR($p$) Levinson-Durbin Whitening (`mlt_arwhiten`)**: Fit a single pooled Toeplitz Yule-Walker AR($p$) polynomial $A(z) = 1 - \sum_{m=1}^p a_m z^{-m}$ (with diagonal ridge stabilization) on $A_m$ and filter **both** $A_m$ and synced $B_m$ with the **identical** filter $A(z)$ so phase delay is completely preserved while narrowing the autocorrelation peak to the impulse bandwidth.
3. **Subspace RMS Cross-Correlation (`mlt_xcov`)**: For each discrete sample lag $\ell$, compute:
   $$\rho_{\text{RMS}}(\ell) = \sqrt{ \frac{1}{\min(k_A, k_B)} \sum_{i=1}^{k_A} \sum_{j=1}^{k_B} r_{ij}(\ell)^2 } \in [0, 1]$$
   over overlapping samples after causal burn-in, yielding unity (`1.0`) for identical subspaces and sharp sub-frame coarse peaks (`mlt_peak1d` parabolic interpolation).
4. **Continuous Timestamp Refinement & Drift**: Scan `mlt.tlmsync` across a fine grid $[\tau_{\text{coarse}} - \text{fine\_ms},\; \tau_{\text{coarse}} + \text{fine\_ms}]$ at `step_ms`, re-evaluating equal common support `[s0, s1]` across endpoints, fit `mlt_peak1d` to obtain $\tau$ and FWHM, and split the valid window into `nblocks` contiguous segments to measure empirical uncertainty ($\sigma/\sqrt{M-1}$) and clock drift in $\text{ppm}$.

## 5. Regularized SVD-CCA with Contiguous Held-Out CV (`mlt.cca`)

1. Synchronize $B$ to $A$ at $\tau$ (`lag=auto` invokes `mlt.latency`) and restrict to valid overlap `[SYNCI0, SYNCI1]` ($N_{\text{val}}$ frames).
2. Split contiguously into first $\lfloor f_{\text{train}} N_{\text{val}} \rfloor$ training frames and trailing test frames to prevent autocorrelation leakage across random splits.
3. **Fit exclusively on the training block**:
   - Compute top-$k_A$, $k_B$ PCA subspaces ($V_A, \lambda_A$) and ($V_B, \lambda_B$).
   - Apply relative ridge shrinkage: $s_{A,i} = \sqrt{\lambda_{A,i} + r_A \lambda_{A,0}}$ (and analogously $s_{B,j}$).
   - Whiten projections $Z_A = s_A^{-1} (V_A^T X_{A,c,\text{tr}})$, $Z_B = s_B^{-1} (V_B^T X_{B,c,\text{tr}})$ and compute compact cross-covariance $K = \frac{1}{N_{\text{tr}}-1} Z_A Z_B^T = P \operatorname{diag}(\rho_{\text{train}}) Q^T$ via SVD.
   - Canonical filters in pixel space: $W_A = V_A s_A^{-1} P_{:, 0:k}$, $W_B = V_B s_B^{-1} Q_{:, 0:k}$.
4. **Out-of-sample evaluation**: Project held-out test frames using training means and $(W_A, W_B)$ to report unbiased `rho_test`. Refit on full $N_{\text{val}}$ window to export high-SNR weights (`cca_*_maps.fits`), Haufe forward loading patterns $\Sigma_A W_A$, $\Sigma_B W_B$ (`cca_*_patterns.fits`), and variates (`cca_U`, `cca_V`).

## 6. Welch PSD & Two-Stream Coherence (`mlt.psd`)

Segments each channel into overlapping Hann-windowed blocks of even length $L$, computes per-axis FFTs via `@calc` `fft(..., 1)`, and normalizes one-sided PSD ($\text{units}^2/\text{Hz}$, accounting for DC/Nyquist factor of 2). When a second stream is provided, aligns it via `mlt.tlmsync` and computes ensemble magnitude-squared coherence:
$$C_{xy}(f) = \frac{|\langle S_{xy}(f) \rangle|^2}{\langle S_{xx}(f) \rangle \langle S_{yy}(f) \rangle}, \qquad \phi_{xy}(f) = \arg \langle S_{xy}(f) \rangle$$
