// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    ao_math.h
 * @brief   Shared FITS/SHM array loaders, radix-2 FFTs, pupil/DM stencils, & radial PSF metrics
 */

#ifndef MILK_MLTOOLS_AO_MATH_H
#define MILK_MLTOOLS_AO_MATH_H

#include <stddef.h>
#include <stdint.h>

#include "libmilkcommon/milkDebugTools.h"

#ifdef __cplusplus
extern "C"
{
#endif

#define MLT_DM_STENCIL 16

    typedef struct
    {
        char   key[16];
        char   type; /* 'D' = double, 'L' = long, 'S' = string */
        double val_d;
        long   val_l;
        char   val_s[64];
        char   comment[64];
    } mlt_fits_kw;

    /**
     * mlt_load_array_f32 - Resolve an in-memory MILK image, FITS file, or telemetry dir to float32
     * @spec:       Image name (`@L:...`, SHM, local), `.fits[.fz]` path, or telemetry directory
     * @stream_opt: Optional stream filter when @spec is a telemetry directory (may be NULL/"")
     * @buf_out:    Heap-allocated float32 buffer of size `nx * ny * nz` (caller frees)
     * @nx_out:     Output axis-0 dimension
     * @ny_out:     Output axis-1 dimension (`1` for 1D)
     * @nz_out:     Output axis-2 dimension (`1` for 1D/2D)
     */
    errno_t mlt_load_array_f32(const char *spec,
                               const char *stream_opt,
                               float     **buf_out,
                               uint32_t   *nx_out,
                               uint32_t   *ny_out,
                               uint32_t   *nz_out);

    /**
     * mlt_write_fits_f32 - Write a 1D/2D/3D float32 FITS image with optional header keywords
     */
    errno_t mlt_write_fits_f32(const char        *path,
                               const float       *data,
                               int                naxis,
                               const long        *naxes,
                               const mlt_fits_kw *kws,
                               size_t             nkws);

    /**
     * mlt_next_pow2_u32 - Smallest power of two >= @v (minimum 2)
     */
    uint32_t mlt_next_pow2_u32(uint32_t v);

    /**
     * mlt_fft1d_c2c - Thread-safe in-place radix-2 1D complex FFT (`re_im` length `2 * n`)
     * @re_im:   Interleaved `[Re, Im]` buffer (`n` must be a power of two >= 2)
     * @n:       Transform length
     * @inverse: `0` for forward DFT, non-zero for `1/n`-scaled inverse DFT
     */
    void mlt_fft1d_c2c(float *re_im, uint32_t n, int inverse);

    /**
     * mlt_fft2d_c2c - OpenMP-parallel in-place radix-2 2D complex FFT (`nx`, `ny` powers of two)
     */
    void mlt_fft2d_c2c(float *re_im, uint32_t nx, uint32_t ny, int inverse);

    /**
     * mlt_make_pupil - Synthesize 4x4 subpixel anti-aliased obstructed telescope pupil `[nwf*nwf]`
     */
    void mlt_make_pupil(uint32_t nwf,
                        float    pup_diam,
                        float    cobs,
                        int      nspider,
                        float    spider_w,
                        float    spider_deg,
                        float   *pup_out);

    /**
     * mlt_dm_modes_to_active_pupil - Interpolate `[nact_x, nact_y, nmodes]` DM modes onto active
     *                                pupil pixels using a 4x4 Gaussian partition-of-unity stencil
     * @modes_dm:   Input DM modes `[nact_x * nact_y * nmodes]` (or `[nwf * nwf * nmodes]`)
     * @nact_x:     Mode grid X size
     * @nact_y:     Mode grid Y size
     * @nmodes:     Number of modes
     * @nwf:        Target pupil grid size
     * @dm_diam:    DM actuator grid span as fraction of @nwf
     * @if_coupl:   Inter-actuator Gaussian coupling at 1 pitch (`[0.01..0.45]`)
     * @opd_scale:  Wavefront OPD [um] per mode unit (`2.0` for reflective DM stroke [um])
     * @act_idx:    Active pupil pixel indices `[n_act]` in `[0, nwf * nwf)`
     * @n_act:      Number of active pupil pixels
     * @u_act_out:  Output row-major matrix `[n_act * nmodes]` (`u_act[p * nmodes + m]`)
     */
    errno_t mlt_dm_modes_to_active_pupil(const float    *modes_dm,
                                         uint32_t        nact_x,
                                         uint32_t        nact_y,
                                         uint32_t        nmodes,
                                         uint32_t        nwf,
                                         float           dm_diam,
                                         float           if_coupl,
                                         float           opd_scale,
                                         const uint32_t *act_idx,
                                         uint32_t        n_act,
                                         float         **u_act_out);

    /**
     * mlt_psf_metrics - Compute peak, FWHM [pixels], EE(radius_pix), and azimuthally averaged
     *                   radial profile on a 2D `[ncam x ncam]` PSF image
     */
    void mlt_psf_metrics(const float *img,
                         uint32_t     ncam,
                         float        ee_rad_pix,
                         double      *peak_out,
                         double      *fwhm_pix_out,
                         double      *ee_out,
                         float       *rad_prof_out,
                         uint32_t     nbins);

#ifdef __cplusplus
}
#endif

#endif /* MILK_MLTOOLS_AO_MATH_H */
