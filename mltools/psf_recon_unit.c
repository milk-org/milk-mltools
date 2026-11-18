// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    psf_recon_unit.c
 * @brief   Telemetry-driven polychromatic AO PSF reconstruction (`mlt.psfrecon`)
 */

// MILK_CMAKE_MANDATE_CFITSIO
// MILK_CMAKE_REQUEST_BLAS

#ifndef _GNU_SOURCE
#    define _GNU_SOURCE
#endif

#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef HAVE_MKL
#    include <mkl_cblas.h>
#elif defined(HAVE_BLAS)
#    include <cblas.h>
#endif

#include "milk_runnable.h"
#include "ao_math.h"
#include "tlm_fileset.h"

#ifndef M_PI
#    define M_PI 3.14159265358979323846
#endif

extern void openblas_set_num_threads(int num_threads) __attribute__((weak));

static char    p_mval[FUNCTION_PARAMETER_STRMAXLEN]      = "";
static char    p_modes_dm[FUNCTION_PARAMETER_STRMAXLEN]  = "";
static char    p_pup[FUNCTION_PARAMETER_STRMAXLEN]       = "";
static char    p_ncpa[FUNCTION_PARAMETER_STRMAXLEN]      = "";
static char    p_ogain[FUNCTION_PARAMETER_STRMAXLEN]     = "";
static char    p_ref_psf[FUNCTION_PARAMETER_STRMAXLEN]   = "";
static char    p_outdir[FUNCTION_PARAMETER_STRMAXLEN]    = ".";
static char    p_outreport[FUNCTION_PARAMETER_STRMAXLEN] = "";
static char    p_method[FUNCTION_PARAMETER_STRMAXLEN]    = "hybrid";
static int64_t p_nwf                                     = 256;
static int64_t p_ncam                                    = 256;
static double  p_lambda_um                               = 1.65;
static double  p_bw_frac                                 = 0.18;
static int64_t p_nlambda                                 = 3;
static double  p_pixscale_ld                             = 2.50;
static double  p_tel_diam_m                              = 8.20;
static double  p_pup_diam                                = 0.9375;
static double  p_cobs                                    = 0.30;
static double  p_dm_diam                                 = 0.9986413;
static double  p_if_coupling                             = 0.15;
static double  p_opd_scale                               = 2.0;
static double  p_ogain_scalar                            = 1.0;
static double  p_seeing_arcsec                           = 0.60;
static double  p_kappa_fit                               = 0.30;
static int     p_denoise                                 = 0;
static int64_t p_maxframes                               = 0;
static int64_t p_decim                                   = 1;

static int64_t o_nframes_used   = 0;
static double  o_sr_recon       = 0.0;
static double  o_sr_ctrl        = 0.0;
static double  o_sr_meas        = 0.0;
static double  o_sr_abs_err     = 0.0;
static double  o_fwhm_ld        = 0.0;
static double  o_fwhm_meas_ld   = 0.0;
static double  o_ee_122         = 0.0;
static double  o_radial_rms_rel = 0.0;
static double  o_corr_2d        = 0.0;
static double  o_inband_rms_nm  = 0.0;
static double  o_noise_rms_nm   = 0.0;
static double  o_fitting_rms_nm = 0.0;

#define FPS_PARAMS_PSFRECON(X)                                                                   \
    X(".mval", p_mval, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT,                                   \
      "Residual modal telemetry (image/FITS/dir)")                                               \
    X(".modes_dm", p_modes_dm, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT,                           \
      "DM modal basis cube CMmodesDM [nx,ny,M]")                                                 \
    X(".pup", p_pup, FPTYPE_STRING, 0, FPFLAG_DEFAULT_INPUT,                                     \
      "Optional 2D pupil amplitude FITS/image")                                                  \
    X(".ncpa", p_ncpa, FPTYPE_STRING, 0, FPFLAG_DEFAULT_INPUT,                                   \
      "Optional 2D static NCPA OPD [um] FITS/image")                                             \
    X(".ogain", p_ogain, FPTYPE_STRING, 0, FPFLAG_DEFAULT_INPUT,                                 \
      "Optional per-mode optical gain vector [M,1]")                                             \
    X(".ref_psf", p_ref_psf, FPTYPE_STRING, 0, FPFLAG_DEFAULT_INPUT,                             \
      "Optional measured long-exposure PSF for validation")                                      \
    X(".outdir", p_outdir, FPTYPE_DIRNAME, 1, FPFLAG_DEFAULT_INPUT,                              \
      "Output directory for reconstructed PSF FITS")                                             \
    X(".outreport", p_outreport, FPTYPE_FILENAME, 0, FPFLAG_DEFAULT_INPUT,                       \
      "Optional key=value summary report")                                                       \
    X(".method", p_method, FPTYPE_STRING_NOT_STREAM, 0, FPFLAG_DEFAULT_INPUT,                    \
      "Method: hybrid|short_exp|otf")                                                            \
    X(".nwf", &p_nwf, FPTYPE_INT64, 0, FPFLAG_DEFAULT_INPUT, "Pupil grid size [pix]")            \
    X(".ncam", &p_ncam, FPTYPE_INT64, 0, FPFLAG_DEFAULT_INPUT, "PSF detector size [pix]")        \
    X(".lambda_um", &p_lambda_um, FPTYPE_FLOAT64, 0, FPFLAG_DEFAULT_INPUT,                       \
      "Central wavelength [um]")                                                                 \
    X(".bw_frac", &p_bw_frac, FPTYPE_FLOAT64, 0, FPFLAG_DEFAULT_INPUT,                           \
      "Fractional filter bandwidth dlambda/lambda")                                              \
    X(".nlambda", &p_nlambda, FPTYPE_INT64, 0, FPFLAG_DEFAULT_INPUT,                             \
      "Polychromatic spectral slices")                                                           \
    X(".pixscale_ld", &p_pixscale_ld, FPTYPE_FLOAT64, 0, FPFLAG_DEFAULT_INPUT,                   \
      "Detector sampling [pix per lambda0/D]")                                                   \
    X(".tel_diam_m", &p_tel_diam_m, FPTYPE_FLOAT64, 0, FPFLAG_DEFAULT_INPUT,                     \
      "Telescope diameter [m]")                                                                  \
    X(".pup_diam", &p_pup_diam, FPTYPE_FLOAT64, 0, FPFLAG_DEFAULT_INPUT,                         \
      "Clear pupil diameter fraction of nwf")                                                    \
    X(".cobs", &p_cobs, FPTYPE_FLOAT64, 0, FPFLAG_DEFAULT_INPUT,                                 \
      "Central obscuration ratio")                                                               \
    X(".dm_diam", &p_dm_diam, FPTYPE_FLOAT64, 0, FPFLAG_DEFAULT_INPUT,                           \
      "DM actuator span fraction of nwf")                                                        \
    X(".if_coupling", &p_if_coupling, FPTYPE_FLOAT64, 0, FPFLAG_DEFAULT_INPUT,                   \
      "DM inter-actuator Gaussian coupling")                                                     \
    X(".opd_scale", &p_opd_scale, FPTYPE_FLOAT64, 0, FPFLAG_DEFAULT_INPUT,                       \
      "Wavefront OPD [um] per DM stroke unit (2.0 refl)")                                        \
    X(".ogain_scalar", &p_ogain_scalar, FPTYPE_FLOAT64, 0, FPFLAG_DEFAULT_INPUT,                 \
      "Scalar optical gain when .ogain unset")                                                   \
    X(".seeing_arcsec", &p_seeing_arcsec, FPTYPE_FLOAT64, 0, FPFLAG_DEFAULT_INPUT,               \
      "Seeing FWHM @ 0.50 um [arcsec] (0=no fitting halo)")                                      \
    X(".kappa_fit", &p_kappa_fit, FPTYPE_FLOAT64, 0, FPFLAG_DEFAULT_INPUT,                       \
      "Hudgin DM fitting error constant (~0.28)")                                                \
    X(".denoise", &p_denoise, FPTYPE_ONOFF, 0, FPFLAG_DEFAULT_INPUT,                             \
      "Wiener-shrink temporal WFS noise floor")                                                  \
    X(".maxframes", &p_maxframes, FPTYPE_INT64, 0, FPFLAG_DEFAULT_INPUT,                         \
      "Max frames after decimation (0=all)")                                                     \
    X(".decim", &p_decim, FPTYPE_INT64, 0, FPFLAG_DEFAULT_INPUT, "Temporal stride (>=1)")        \
    X(".nframes_used", &o_nframes_used, FPTYPE_INT64, 0, FPFLAG_DEFAULT_OUTPUT,                  \
      "Frames integrated in reconstruction")                                                     \
    X(".sr_recon", &o_sr_recon, FPTYPE_FLOAT64, 0, FPFLAG_DEFAULT_OUTPUT,                        \
      "Reconstructed Strehl peak [0..1]")                                                        \
    X(".sr_ctrl", &o_sr_ctrl, FPTYPE_FLOAT64, 0, FPFLAG_DEFAULT_OUTPUT,                          \
      "Controlled-modes-only Strehl peak [0..1]")                                                \
    X(".sr_meas", &o_sr_meas, FPTYPE_FLOAT64, 0, FPFLAG_DEFAULT_OUTPUT,                          \
      "Measured reference Strehl peak [0..1]")                                                   \
    X(".sr_abs_err", &o_sr_abs_err, FPTYPE_FLOAT64, 0, FPFLAG_DEFAULT_OUTPUT,                    \
      "Absolute Strehl reconstruction error")                                                    \
    X(".fwhm_ld", &o_fwhm_ld, FPTYPE_FLOAT64, 0, FPFLAG_DEFAULT_OUTPUT,                          \
      "Reconstructed FWHM [lambda/D]")                                                           \
    X(".fwhm_meas_ld", &o_fwhm_meas_ld, FPTYPE_FLOAT64, 0, FPFLAG_DEFAULT_OUTPUT,                \
      "Measured reference FWHM [lambda/D]")                                                      \
    X(".ee_122", &o_ee_122, FPTYPE_FLOAT64, 0, FPFLAG_DEFAULT_OUTPUT,                            \
      "Reconstructed encircled energy within 1.22 lambda/D")                                     \
    X(".radial_rms_rel", &o_radial_rms_rel, FPTYPE_FLOAT64, 0, FPFLAG_DEFAULT_OUTPUT,            \
      "Relative RMS radial profile error vs ref_psf")                                            \
    X(".corr_2d", &o_corr_2d, FPTYPE_FLOAT64, 0, FPFLAG_DEFAULT_OUTPUT,                          \
      "2D pixel correlation vs ref_psf")                                                         \
    X(".inband_rms_nm", &o_inband_rms_nm, FPTYPE_FLOAT64, 0, FPFLAG_DEFAULT_OUTPUT,              \
      "In-band controlled residual OPD RMS [nm]")                                                \
    X(".noise_rms_nm", &o_noise_rms_nm, FPTYPE_FLOAT64, 0, FPFLAG_DEFAULT_OUTPUT,                \
      "Estimated WFS noise OPD RMS [nm]")                                                        \
    X(".fitting_rms_nm", &o_fitting_rms_nm, FPTYPE_FLOAT64, 0, FPFLAG_DEFAULT_OUTPUT,            \
      "Out-of-band Kolmogorov fitting OPD RMS [nm]")

MILK_PARAMS_DEFINE(mlt_psfrecon, FPS_PARAMS_PSFRECON);

/* Evaluate 2D complex Soummer MFT: efoc[ncam,ncam] = U[ncam,nwf] * epup[nwf,nwf] * V[nwf,ncam] */
static void eval_mft_2d(const float *u_c,
                        const float *epup,
                        const float *v_c,
                        float       *tmp,
                        float       *efoc,
                        uint32_t     nwf,
                        uint32_t     ncam)
{
#ifdef HAVE_BLAS
    const float one[2]  = { 1.0f, 0.0f };
    const float zero[2] = { 0.0f, 0.0f };
    cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int) ncam, (int) nwf, (int) nwf, one,
                u_c, (int) nwf, epup, (int) nwf, zero, tmp, (int) nwf);
    cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int) ncam, (int) ncam, (int) nwf, one,
                tmp, (int) nwf, v_c, (int) ncam, zero, efoc, (int) ncam);
#else
    memset(tmp, 0, (size_t) ncam * nwf * 2u * sizeof(float));
    for (uint32_t yf = 0; yf < ncam; yf++)
    {
        for (uint32_t yp = 0; yp < nwf; yp++)
        {
            float ur = u_c[((size_t) yf * nwf + yp) * 2u];
            float ui = u_c[((size_t) yf * nwf + yp) * 2u + 1u];
            for (uint32_t xp = 0; xp < nwf; xp++)
            {
                float er = epup[((size_t) yp * nwf + xp) * 2u];
                float ei = epup[((size_t) yp * nwf + xp) * 2u + 1u];
                size_t ti = ((size_t) yf * nwf + xp) * 2u;
                tmp[ti] += ur * er - ui * ei;
                tmp[ti + 1u] += ur * ei + ui * er;
            }
        }
    }
    memset(efoc, 0, (size_t) ncam * ncam * 2u * sizeof(float));
    for (uint32_t yf = 0; yf < ncam; yf++)
    {
        for (uint32_t xp = 0; xp < nwf; xp++)
        {
            float tr = tmp[((size_t) yf * nwf + xp) * 2u];
            float ti = tmp[((size_t) yf * nwf + xp) * 2u + 1u];
            for (uint32_t xf = 0; xf < ncam; xf++)
            {
                float vr = v_c[((size_t) xp * ncam + xf) * 2u];
                float vi = v_c[((size_t) xp * ncam + xf) * 2u + 1u];
                size_t fi = ((size_t) yf * ncam + xf) * 2u;
                efoc[fi] += tr * vr - ti * vi;
                efoc[fi + 1u] += tr * vi + ti * vr;
            }
        }
    }
#endif
}

/* Precompute polychromatic Soummer MFT matrices U_c [ncam,nwf] & V_c [nwf,ncam] */
static void precompute_mft_uv(uint32_t     nwf,
                              uint32_t     ncam,
                              int          nlam,
                              const float *lam_um,
                              float        lam0,
                              float        pup_diam,
                              float        pixscale_ld,
                              double       pup_sum_amp,
                              float       *u_all,
                              float       *v_all)
{
    const float dpix = pup_diam * (float) nwf;
    const float p0   = 0.5f * (float) (nwf - 1u);
    const float f0   = (float) (ncam / 2u);
    const float nrm  = (pup_sum_amp > 0.0) ? (float) (1.0 / sqrt(pup_sum_amp)) : 1.0f;

    for (int c = 0; c < nlam; c++)
    {
        float  sc = pixscale_ld * (lam_um[c] / lam0);
        float *u  = &u_all[(size_t) c * ncam * nwf * 2u];
        float *v  = &v_all[(size_t) c * nwf * ncam * 2u];

        for (uint32_t yf = 0; yf < ncam; yf++)
        {
            float eta = ((float) yf - f0) / sc;
            for (uint32_t yp = 0; yp < nwf; yp++)
            {
                float yt  = ((float) yp - p0) / dpix;
                float ang = -2.0f * (float) M_PI * yt * eta;
                size_t idx = ((size_t) yf * nwf + yp) * 2u;
                u[idx]      = nrm * cosf(ang);
                u[idx + 1u] = nrm * sinf(ang);
            }
        }
        for (uint32_t xp = 0; xp < nwf; xp++)
        {
            float xt = ((float) xp - p0) / dpix;
            for (uint32_t xf = 0; xf < ncam; xf++)
            {
                float xi  = ((float) xf - f0) / sc;
                float ang = -2.0f * (float) M_PI * xt * xi;
                size_t idx = ((size_t) xp * ncam + xf) * 2u;
                v[idx]      = nrm * cosf(ang);
                v[idx + 1u] = nrm * sinf(ang);
            }
        }
    }
}

/* Convolve focal-plane PSF with stationary structure-function OTF exp(-0.5 * D(rho)) */
static errno_t convolve_sf_halo(const float *psf_in,
                                float       *psf_out,
                                uint32_t     ncam,
                                double       tel_diam_m,
                                double       pixscale_ld,
                                double       r0_lam_m,
                                double       sigma2_rad2,
                                double       extra_inband_rad2,
                                double       d_act_m)
{
    if (sigma2_rad2 <= 1.0e-9 && extra_inband_rad2 <= 1.0e-9)
    {
        memcpy(psf_out, psf_in, (size_t) ncam * ncam * sizeof(float));
        return RETURN_SUCCESS;
    }

    uint32_t nfft = mlt_next_pow2_u32(2u * ncam);
    float   *cbuf = (float *) calloc((size_t) nfft * nfft * 2u, sizeof(float));
    if (cbuf == NULL)
    {
        return RETURN_FAILURE;
    }

    uint32_t off = (nfft - ncam) / 2u;
    for (uint32_t y = 0; y < ncam; y++)
    {
        for (uint32_t x = 0; x < ncam; x++)
        {
            size_t di        = ((size_t) (y + off) * nfft + (x + off)) * 2u;
            cbuf[di]         = psf_in[(size_t) y * ncam + x];
            cbuf[di + 1u]    = 0.0f;
        }
    }

    mlt_fft2d_c2c(cbuf, nfft, nfft, 0);

    const double drho = (tel_diam_m * pixscale_ld) / (double) nfft;
    const double l_in = (d_act_m > 1.0e-3) ? (2.5 * d_act_m) : (0.15 * tel_diam_m);

    for (uint32_t ky = 0; ky < nfft; ky++)
    {
        int    sy = (ky <= nfft / 2u) ? (int) ky : ((int) ky - (int) nfft);
        double ry = (double) sy * drho;
        for (uint32_t kx = 0; kx < nfft; kx++)
        {
            int    sx  = (kx <= nfft / 2u) ? (int) kx : ((int) kx - (int) nfft);
            double rx  = (double) sx * drho;
            double rho = hypot(rx, ry);
            double d_tot = 0.0;

            if (sigma2_rad2 > 1.0e-9 && r0_lam_m > 1.0e-4 && rho > 0.0)
            {
                double d_kolm = 6.88 * pow(rho / r0_lam_m, 5.0 / 3.0);
                double asymp  = 2.0 * sigma2_rad2;
                d_tot += asymp * (1.0 - exp(-d_kolm / asymp));
            }
            if (extra_inband_rad2 > 1.0e-9 && rho > 0.0)
            {
                double asymp_in = 2.0 * extra_inband_rad2;
                double u        = rho / l_in;
                d_tot += asymp_in * (1.0 - exp(-pow(u, 5.0 / 3.0)));
            }

            float att = (float) exp(-0.5 * d_tot);
            size_t idx = ((size_t) ky * nfft + kx) * 2u;
            cbuf[idx] *= att;
            cbuf[idx + 1u] *= att;
        }
    }

    mlt_fft2d_c2c(cbuf, nfft, nfft, 1);

    for (uint32_t y = 0; y < ncam; y++)
    {
        for (uint32_t x = 0; x < ncam; x++)
        {
            size_t si = ((size_t) (y + off) * nfft + (x + off)) * 2u;
            float  v  = cbuf[si];
            psf_out[(size_t) y * ncam + x] = (v > 0.0f) ? v : 0.0f;
        }
    }

    free(cbuf);
    return RETURN_SUCCESS;
}

static errno_t psfrecon_run(void)
{
    int use_otf_only = (strcmp(p_method, "otf") == 0);
    int use_hybrid   = (strcmp(p_method, "hybrid") == 0 || p_method[0] == '\0');
    if (!use_otf_only && !use_hybrid && strcmp(p_method, "short_exp") != 0)
    {
        fprintf(stderr, "ERROR: unknown .method '%s' (expected hybrid|short_exp|otf)\n", p_method);
        return RETURN_FAILURE;
    }

    uint32_t nwf  = (p_nwf >= 16) ? (uint32_t) p_nwf : 256u;
    uint32_t ncam = (p_ncam >= 16) ? (uint32_t) p_ncam : 256u;
    size_t   npix_wf  = (size_t) nwf * nwf;
    size_t   npix_cam = (size_t) ncam * ncam;

    /* 1. Load modal telemetry (M, N) or (M, 1, N) */
    float   *mval_raw = NULL;
    uint32_t mx = 0, my = 0, mz = 0;
    if (mlt_load_array_f32(p_mval, NULL, &mval_raw, &mx, &my, &mz) != RETURN_SUCCESS)
    {
        return RETURN_FAILURE;
    }
    uint32_t nmodes = mx;
    uint32_t nt_raw = (mz > 1u && my == 1u) ? mz : ((mz == 1u) ? my : 0u);
    if (nmodes == 0 || nt_raw == 0)
    {
        fprintf(stderr, "ERROR: .mval '%s' must be (M,N) or (M,1,N), got %ux%ux%u\n", p_mval, mx,
                my, mz);
        free(mval_raw);
        return RETURN_FAILURE;
    }

    /* 2. Load DM modal basis cube [nact_x, nact_y, M_basis] */
    float   *modes_dm = NULL;
    uint32_t dx = 0, dy = 0, dz = 0;
    if (mlt_load_array_f32(p_modes_dm, NULL, &modes_dm, &dx, &dy, &dz) != RETURN_SUCCESS)
    {
        free(mval_raw);
        return RETURN_FAILURE;
    }
    if (dz < nmodes)
    {
        nmodes = dz;
    }

    /* 3. Load or synthesize pupil amplitude & optional static NCPA */
    float *pup_amp  = (float *) calloc(npix_wf, sizeof(float));
    float *ncpa_opd = (float *) calloc(npix_wf, sizeof(float));
    if (pup_amp == NULL || ncpa_opd == NULL)
    {
        free(mval_raw);
        free(modes_dm);
        free(pup_amp);
        free(ncpa_opd);
        return RETURN_FAILURE;
    }

    if (p_pup[0] != '\0')
    {
        float   *ptmp = NULL;
        uint32_t px = 0, py = 0, pz = 0;
        if (mlt_load_array_f32(p_pup, NULL, &ptmp, &px, &py, &pz) == RETURN_SUCCESS &&
            px == nwf && py == nwf)
        {
            memcpy(pup_amp, ptmp, npix_wf * sizeof(float));
        }
        else
        {
            mlt_make_pupil(nwf, (float) p_pup_diam, (float) p_cobs, 4, 0.012f, 45.0f, pup_amp);
        }
        free(ptmp);
    }
    else
    {
        mlt_make_pupil(nwf, (float) p_pup_diam, (float) p_cobs, 4, 0.012f, 45.0f, pup_amp);
    }

    if (p_ncpa[0] != '\0')
    {
        float   *ntmp = NULL;
        uint32_t nx0 = 0, ny0 = 0, nz0 = 0;
        if (mlt_load_array_f32(p_ncpa, NULL, &ntmp, &nx0, &ny0, &nz0) == RETURN_SUCCESS &&
            nx0 == nwf && ny0 == nwf)
        {
            memcpy(ncpa_opd, ntmp, npix_wf * sizeof(float));
        }
        free(ntmp);
    }

    uint32_t *act_idx     = (uint32_t *) malloc(npix_wf * sizeof(uint32_t));
    uint32_t  n_act       = 0;
    double    pup_sum_amp = 0.0;
    for (uint32_t p = 0; p < (uint32_t) npix_wf; p++)
    {
        if (pup_amp[p] > 0.0f)
        {
            pup_sum_amp += (double) pup_amp[p];
            act_idx[n_act++] = p;
        }
    }
    if (n_act == 0 || pup_sum_amp <= 0.0)
    {
        free(mval_raw);
        free(modes_dm);
        free(pup_amp);
        free(ncpa_opd);
        free(act_idx);
        return RETURN_FAILURE;
    }

    /* 4. Map DM modes to active pupil pixels [n_act x nmodes] in OPD [um] */
    float *u_act = NULL;
    if (mlt_dm_modes_to_active_pupil(modes_dm, dx, dy, nmodes, nwf, (float) p_dm_diam,
                                     (float) p_if_coupling, (float) p_opd_scale, act_idx, n_act,
                                     &u_act) != RETURN_SUCCESS)
    {
        free(mval_raw);
        free(modes_dm);
        free(pup_amp);
        free(ncpa_opd);
        free(act_idx);
        return RETURN_FAILURE;
    }
    free(modes_dm);

    /* Per-mode spatial OPD RMS weight [um / modal_unit] for noise/inband accounting */
    double *mode_opd_var = (double *) calloc(nmodes, sizeof(double));
    for (uint32_t p = 0; p < n_act; p++)
    {
        const float *row = &u_act[(size_t) p * nmodes];
        for (uint32_t m = 0; m < nmodes; m++)
        {
            mode_opd_var[m] += (double) row[m] * (double) row[m];
        }
    }
    for (uint32_t m = 0; m < nmodes; m++)
    {
        mode_opd_var[m] /= (double) n_act;
    }

    /* 5. Optical gain & temporal WFS noise Wiener shrinkage */
    float *ogain = (float *) malloc(nmodes * sizeof(float));
    float *w_shr = (float *) malloc(nmodes * sizeof(float));
    double def_og = (p_ogain_scalar > 0.05) ? p_ogain_scalar : 1.0;
    for (uint32_t m = 0; m < nmodes; m++)
    {
        ogain[m] = (float) def_og;
        w_shr[m] = 1.0f;
    }
    if (p_ogain[0] != '\0')
    {
        float   *otmp = NULL;
        uint32_t ox = 0, oy = 0, oz = 0;
        if (mlt_load_array_f32(p_ogain, NULL, &otmp, &ox, &oy, &oz) == RETURN_SUCCESS)
        {
            uint32_t n = (ox * oy * oz < nmodes) ? (ox * oy * oz) : nmodes;
            for (uint32_t m = 0; m < n; m++)
            {
                if (otmp[m] >= 0.05f && otmp[m] <= 2.0f)
                {
                    ogain[m] = otmp[m];
                }
            }
            free(otmp);
        }
    }

    double noise_opd_var_um2 = 0.0;
    for (uint32_t m = 0; m < nmodes; m++)
    {
        double inv_og = 1.0 / (double) ogain[m];
        double mean_m = 0.0;
        for (uint32_t t = 0; t < nt_raw; t++)
        {
            mean_m += (double) mval_raw[(size_t) t * mx + m];
        }
        mean_m /= (double) nt_raw;

        double var_m = 0.0;
        double d4_m  = 0.0;
        for (uint32_t t = 0; t < nt_raw; t++)
        {
            double v = (double) mval_raw[(size_t) t * mx + m] - mean_m;
            var_m += v * v;
            if (t >= 4u)
            {
                double dd = (double) mval_raw[(size_t) t * mx + m] -
                            4.0 * (double) mval_raw[(size_t) (t - 1u) * mx + m] +
                            6.0 * (double) mval_raw[(size_t) (t - 2u) * mx + m] -
                            4.0 * (double) mval_raw[(size_t) (t - 3u) * mx + m] +
                            (double) mval_raw[(size_t) (t - 4u) * mx + m];
                d4_m += dd * dd;
            }
        }
        var_m /= (double) nt_raw;
        double nvar = (nt_raw > 6u) ? (d4_m / (70.0 * (double) (nt_raw - 4u))) : 0.0;
        if (nvar > var_m)
        {
            nvar = var_m;
        }
        noise_opd_var_um2 += (nvar * inv_og * inv_og) * mode_opd_var[m];

        if (p_denoise && var_m > 1.0e-24)
        {
            double sig_frac = (var_m - nvar) / var_m;
            w_shr[m]        = (sig_frac > 0.0) ? (float) sqrt(sig_frac) : 0.0f;
        }
    }
    free(mode_opd_var);

    /* Select decimated frame subset [nmodes x nt_used] */
    uint32_t decim = (p_decim >= 1) ? (uint32_t) p_decim : 1u;
    uint32_t nt_used = (nt_raw + decim - 1u) / decim;
    if (p_maxframes > 0 && (uint32_t) p_maxframes < nt_used)
    {
        nt_used = (uint32_t) p_maxframes;
    }

    float *r_mat = (float *) malloc((size_t) nmodes * nt_used * sizeof(float));
    for (uint32_t k = 0; k < nt_used; k++)
    {
        uint32_t t = k * decim;
        for (uint32_t m = 0; m < nmodes; m++)
        {
            float scale = w_shr[m] / ogain[m];
            r_mat[(size_t) m * nt_used + k] = scale * mval_raw[(size_t) t * mx + m];
        }
    }
    free(mval_raw);
    free(ogain);
    free(w_shr);

    /* 6. Polychromatic channel setup & diffraction-limited reference PSF */
    int   nlam = (p_nlambda >= 1 && p_nlambda <= 15) ? (int) p_nlambda : 3;
    float lam0 = (p_lambda_um > 0.1) ? (float) p_lambda_um : 1.65f;
    float bw   = (p_bw_frac >= 0.0) ? (float) p_bw_frac : 0.18f;
    float ps0  = (p_pixscale_ld > 0.1) ? (float) p_pixscale_ld : 2.50f;
    float lam_um[16];
    for (int c = 0; c < nlam; c++)
    {
        if (nlam == 1 || bw <= 1.0e-4f)
        {
            lam_um[c] = lam0;
        }
        else
        {
            float frac = ((float) c + 0.5f) / (float) nlam - 0.5f;
            lam_um[c]  = lam0 * (1.0f + bw * frac);
        }
    }

    float *u_mft = (float *) malloc((size_t) nlam * ncam * nwf * 2u * sizeof(float));
    float *v_mft = (float *) malloc((size_t) nlam * nwf * ncam * 2u * sizeof(float));
    precompute_mft_uv(nwf, ncam, nlam, lam_um, lam0, (float) p_pup_diam, ps0, pup_sum_amp, u_mft,
                      v_mft);

    float *psf_dl    = (float *) calloc(npix_cam, sizeof(float));
    float *psf_ctrl  = (float *) calloc(npix_cam, sizeof(float));
    float *psf_recon = (float *) calloc(npix_cam, sizeof(float));
    float *opd_std   = (float *) calloc(npix_wf, sizeof(float));

    {
        float *ep0 = (float *) calloc(npix_wf * 2u, sizeof(float));
        float *tm0 = (float *) malloc((size_t) ncam * nwf * 2u * sizeof(float));
        float *ef0 = (float *) malloc(npix_cam * 2u * sizeof(float));
        float  inv_nl = 1.0f / (float) nlam;
        for (int c = 0; c < nlam; c++)
        {
            memset(ep0, 0, npix_wf * 2u * sizeof(float));
            for (uint32_t p = 0; p < n_act; p++)
            {
                uint32_t pix = act_idx[p];
                ep0[2u * pix] = pup_amp[pix];
            }
            eval_mft_2d(&u_mft[(size_t) c * ncam * nwf * 2u], ep0,
                        &v_mft[(size_t) c * nwf * ncam * 2u], tm0, ef0, nwf, ncam);
            for (size_t i = 0; i < npix_cam; i++)
            {
                float re = ef0[2u * i];
                float im = ef0[2u * i + 1u];
                psf_dl[i] += inv_nl * (re * re + im * im);
            }
        }
        free(ep0);
        free(tm0);
        free(ef0);
    }

    /* 7. Reconstruct controlled-subspace pupil OPD statistics & short-exposure PSF */
    double *sum_opd  = (double *) calloc(n_act, sizeof(double));
    double *sum2_opd = (double *) calloc(n_act, sizeof(double));
    double  sum_svar = 0.0;

    if (openblas_set_num_threads != NULL)
    {
        openblas_set_num_threads(1);
    }

    int max_th = omp_get_max_threads();
    int nth    = (max_th > 32) ? 32 : ((max_th > 0) ? max_th : 1);
    if (nt_used > 0u && (uint32_t) nth > nt_used)
    {
        nth = (int) nt_used;
    }

    const uint32_t blk_sz  = 256u;
    const size_t   tfloats = (size_t) ncam * nwf * 2u;
    float  *opd_blk  = (float *) malloc((size_t) n_act * blk_sz * sizeof(float));
    float  *epup_all = use_otf_only ? NULL : (float *) calloc((size_t) nth * npix_wf * 2u, sizeof(float));
    float  *tmp_all  = use_otf_only ? NULL : (float *) malloc((size_t) nth * tfloats * sizeof(float));
    float  *efoc_all = use_otf_only ? NULL : (float *) malloc((size_t) nth * npix_cam * 2u * sizeof(float));
    double *psf_thr  = use_otf_only ? NULL : (double *) calloc((size_t) nth * npix_cam, sizeof(double));
    double *s1_thr   = (double *) calloc((size_t) nth * n_act, sizeof(double));
    double *s2_thr   = (double *) calloc((size_t) nth * n_act, sizeof(double));

    for (uint32_t b0 = 0; b0 < nt_used; b0 += blk_sz)
    {
        uint32_t nb = (b0 + blk_sz <= nt_used) ? blk_sz : (nt_used - b0);

        /* Batch OPD synthesis: opd_blk[n_act, nb] = u_act[n_act, nmodes] * r_mat[:, b0..b0+nb] */
#ifdef HAVE_BLAS
        uint32_t p_chunk;
#    pragma omp parallel for num_threads(nth) schedule(static)
        for (p_chunk = 0; p_chunk < n_act; p_chunk += 512u)
        {
            uint32_t rows = (p_chunk + 512u <= n_act) ? 512u : (n_act - p_chunk);
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int) rows, (int) nb,
                        (int) nmodes, 1.0f, &u_act[(size_t) p_chunk * nmodes], (int) nmodes,
                        &r_mat[b0], (int) nt_used, 0.0f, &opd_blk[(size_t) p_chunk * nb],
                        (int) nb);
        }
#else
        uint32_t p_i;
#    pragma omp parallel for num_threads(nth) schedule(static)
        for (p_i = 0; p_i < n_act; p_i++)
        {
            const float *urow = &u_act[(size_t) p_i * nmodes];
            float       *orow = &opd_blk[(size_t) p_i * nb];
            memset(orow, 0, nb * sizeof(float));
            for (uint32_t m = 0; m < nmodes; m++)
            {
                float        um   = urow[m];
                const float *rrow = &r_mat[(size_t) m * nt_used + b0];
                for (uint32_t k = 0; k < nb; k++)
                {
                    orow[k] += um * rrow[k];
                }
            }
        }
#endif

        /* Accumulate piston-removed spatial variance & optional MFT short exposures */
#pragma omp parallel num_threads(nth) reduction(+ : sum_svar)
        {
            int     tid   = omp_get_thread_num();
            float  *epup  = use_otf_only ? NULL : &epup_all[(size_t) tid * npix_wf * 2u];
            float  *tmp   = use_otf_only ? NULL : &tmp_all[(size_t) tid * tfloats];
            float  *efoc  = use_otf_only ? NULL : &efoc_all[(size_t) tid * npix_cam * 2u];
            double *l_psf = use_otf_only ? NULL : &psf_thr[(size_t) tid * npix_cam];
            double *l_s1  = &s1_thr[(size_t) tid * n_act];
            double *l_s2  = &s2_thr[(size_t) tid * n_act];

            uint32_t k;
#pragma omp for schedule(static)
            for (k = 0; k < nb; k++)
            {
                double w_sum  = 0.0;
                double op_sum = 0.0;
                for (uint32_t p = 0; p < n_act; p++)
                {
                    double w = (double) pup_amp[act_idx[p]];
                    double v = (double) opd_blk[(size_t) p * nb + k] +
                               (double) ncpa_opd[act_idx[p]];
                    w_sum += w;
                    op_sum += w * v;
                }
                double piston = (w_sum > 0.0) ? (op_sum / w_sum) : 0.0;

                double f_var = 0.0;
                for (uint32_t p = 0; p < n_act; p++)
                {
                    double w = (double) pup_amp[act_idx[p]];
                    double v = (double) opd_blk[(size_t) p * nb + k] +
                               (double) ncpa_opd[act_idx[p]] - piston;
                    l_s1[p] += v;
                    l_s2[p] += v * v;
                    f_var += w * v * v;
                }
                sum_svar += (w_sum > 0.0) ? (f_var / w_sum) : 0.0;

                if (!use_otf_only)
                {
                    double inv_nl = 1.0 / (double) nlam;
                    for (int c = 0; c < nlam; c++)
                    {
                        float k_rad = (float) (2.0 * M_PI) / lam_um[c];
                        for (uint32_t p = 0; p < n_act; p++)
                        {
                            uint32_t pix = act_idx[p];
                            float    amp = pup_amp[pix];
                            float    phi = k_rad * (opd_blk[(size_t) p * nb + k] + ncpa_opd[pix] -
                                                    (float) piston);
                            epup[2u * pix]      = amp * cosf(phi);
                            epup[2u * pix + 1u] = amp * sinf(phi);
                        }
                        eval_mft_2d(&u_mft[(size_t) c * ncam * nwf * 2u], epup,
                                    &v_mft[(size_t) c * nwf * ncam * 2u], tmp, efoc, nwf, ncam);
                        for (size_t i = 0; i < npix_cam; i++)
                        {
                            double re = (double) efoc[2u * i];
                            double im = (double) efoc[2u * i + 1u];
                            l_psf[i] += inv_nl * (re * re + im * im);
                        }
                    }
                }
            }
        }
    }

    /* Parallel reduction across persistent worker accumulators (zero critical-section locks) */
#pragma omp parallel num_threads(nth)
    {
        uint32_t p;
#pragma omp for schedule(static)
        for (p = 0; p < n_act; p++)
        {
            double s1 = 0.0, s2 = 0.0;
            for (int t = 0; t < nth; t++)
            {
                s1 += s1_thr[(size_t) t * n_act + p];
                s2 += s2_thr[(size_t) t * n_act + p];
            }
            sum_opd[p]  = s1;
            sum2_opd[p] = s2;
        }
        if (!use_otf_only && nt_used > 0u)
        {
            double inv_nt = 1.0 / (double) nt_used;
            size_t i;
#pragma omp for schedule(static)
            for (i = 0; i < npix_cam; i++)
            {
                double acc = 0.0;
                for (int t = 0; t < nth; t++)
                {
                    acc += psf_thr[(size_t) t * npix_cam + i];
                }
                psf_ctrl[i] += (float) (acc * inv_nt);
            }
        }
    }

    free(epup_all);
    free(tmp_all);
    free(efoc_all);
    free(psf_thr);
    free(s1_thr);
    free(s2_thr);

    for (uint32_t p = 0; p < n_act; p++)
    {
        double mu = sum_opd[p] / (double) nt_used;
        double vr = sum2_opd[p] / (double) nt_used - mu * mu;
        opd_std[act_idx[p]] = (vr > 0.0) ? (float) (1000.0 * sqrt(vr)) : 0.0f;
    }

    double inband_var_um2 = sum_svar / (double) nt_used;
    o_inband_rms_nm       = 1000.0 * sqrt((inband_var_um2 > 0.0) ? inband_var_um2 : 0.0);
    o_noise_rms_nm        = 1000.0 * sqrt((noise_opd_var_um2 > 0.0) ? noise_opd_var_um2 : 0.0);
    o_nframes_used        = (int64_t) nt_used;

    /* 8. Out-of-band Kolmogorov fitting error (effective pitch of controlled basis) */
    double d_act_m = p_tel_diam_m;
    if (dx > 1u && p_pup_diam > 0.05)
    {
        double n_pitch_pup = (double) (dx - 1u) * (p_pup_diam / p_dm_diam);
        if (n_pitch_pup > 1.0)
        {
            d_act_m = p_tel_diam_m / n_pitch_pup;
        }
    }
    double n_pitch_modes = 2.0 * sqrt((double) nmodes / M_PI);
    if (n_pitch_modes > 1.0)
    {
        double d_modes_m = p_tel_diam_m / n_pitch_modes;
        if (d_modes_m > d_act_m)
        {
            d_act_m = d_modes_m;
        }
    }

    double r0_lam_m    = 0.0;
    double fit_var_rad = 0.0;
    if (p_seeing_arcsec > 0.01)
    {
        double seeing_rad = p_seeing_arcsec / 206264.806247;
        double r0_500     = (0.98 * 0.50e-6) / seeing_rad;
        r0_lam_m          = r0_500 * pow((double) lam0 / 0.50, 1.2);
        fit_var_rad       = p_kappa_fit * pow(d_act_m / r0_lam_m, 5.0 / 3.0);
        o_fitting_rms_nm  = (sqrt(fit_var_rad) * ((double) lam0 * 1000.0)) / (2.0 * M_PI);
    }
    else
    {
        o_fitting_rms_nm = 0.0;
    }

    double k0_rad_um = (2.0 * M_PI) / (double) lam0;
    if (use_otf_only)
    {
        double inband_rad2 = inband_var_um2 * k0_rad_um * k0_rad_um;
        (void) convolve_sf_halo(psf_dl, psf_ctrl, ncam, p_tel_diam_m, ps0, r0_lam_m, 0.0,
                                inband_rad2, d_act_m);
        (void) convolve_sf_halo(psf_dl, psf_recon, ncam, p_tel_diam_m, ps0, r0_lam_m, fit_var_rad,
                                inband_rad2, d_act_m);
    }
    else if (use_hybrid && fit_var_rad > 1.0e-9)
    {
        (void) convolve_sf_halo(psf_ctrl, psf_recon, ncam, p_tel_diam_m, ps0, r0_lam_m,
                                fit_var_rad, 0.0, d_act_m);
    }
    else
    {
        memcpy(psf_recon, psf_ctrl, npix_cam * sizeof(float));
    }

    /* 9. Quantitative PSF metrics & optional comparison against measured .ref_psf */
    uint32_t nbins    = ncam / 2u;
    float   *prof_dl  = (float *) calloc(nbins, sizeof(float));
    float   *prof_rec = (float *) calloc(nbins, sizeof(float));
    float   *prof_ref = (float *) calloc(nbins, sizeof(float));
    float   *psf_diff = (float *) calloc(npix_cam, sizeof(float));

    double pk_ctrl = 0.0, fw_rec_px = 0.0;
    mlt_psf_metrics(psf_dl, ncam, 1.22f * ps0, NULL, NULL, NULL, prof_dl, nbins);
    mlt_psf_metrics(psf_ctrl, ncam, 1.22f * ps0, &pk_ctrl, NULL, NULL, NULL, 0);
    mlt_psf_metrics(psf_recon, ncam, 1.22f * ps0, &o_sr_recon, &fw_rec_px, &o_ee_122, prof_rec,
                    nbins);
    o_sr_ctrl = pk_ctrl;
    o_fwhm_ld = fw_rec_px / (double) ps0;

    o_sr_meas        = 0.0;
    o_sr_abs_err     = 0.0;
    o_fwhm_meas_ld   = 0.0;
    o_radial_rms_rel = 0.0;
    o_corr_2d        = 0.0;

    if (p_ref_psf[0] != '\0')
    {
        float   *ref_raw = NULL;
        uint32_t rx = 0, ry = 0, rz = 0;
        if (mlt_load_array_f32(p_ref_psf, NULL, &ref_raw, &rx, &ry, &rz) == RETURN_SUCCESS &&
            rx == ncam && ry == ncam)
        {
            float *ref_2d = (float *) calloc(npix_cam, sizeof(float));
            double inv_rz = 1.0 / (double) rz;
            for (uint32_t z = 0; z < rz; z++)
            {
                const float *sl = &ref_raw[(size_t) z * npix_cam];
                for (size_t i = 0; i < npix_cam; i++)
                {
                    ref_2d[i] += (float) ((double) sl[i] * inv_rz);
                }
            }

            /* Normalize detector ADU images to Strehl units when peak > 2.0 */
            double pk_raw  = 0.0;
            double sum_raw = 0.0;
            double sum_rec = 0.0;
            for (size_t i = 0; i < npix_cam; i++)
            {
                if ((double) ref_2d[i] > pk_raw)
                {
                    pk_raw = (double) ref_2d[i];
                }
                sum_raw += (double) ref_2d[i];
                sum_rec += (double) psf_recon[i];
            }
            if (pk_raw > 2.0 && sum_raw > 1.0e-20)
            {
                float fscale = (float) (sum_rec / sum_raw);
                for (size_t i = 0; i < npix_cam; i++)
                {
                    ref_2d[i] *= fscale;
                }
            }

            double fw_ref_px = 0.0;
            mlt_psf_metrics(ref_2d, ncam, 1.22f * ps0, &o_sr_meas, &fw_ref_px, NULL, prof_ref,
                            nbins);
            o_fwhm_meas_ld = fw_ref_px / (double) ps0;
            o_sr_abs_err   = fabs(o_sr_recon - o_sr_meas);

            double mu_a = sum_rec / (double) npix_cam;
            double mu_b = 0.0;
            for (size_t i = 0; i < npix_cam; i++)
            {
                mu_b += (double) ref_2d[i];
                psf_diff[i] = psf_recon[i] - ref_2d[i];
            }
            mu_b /= (double) npix_cam;

            double sab = 0.0, saa = 0.0, sbb = 0.0;
            for (size_t i = 0; i < npix_cam; i++)
            {
                double da = (double) psf_recon[i] - mu_a;
                double db = (double) ref_2d[i] - mu_b;
                sab += da * db;
                saa += da * da;
                sbb += db * db;
            }
            if (saa > 1.0e-30 && sbb > 1.0e-30)
            {
                o_corr_2d = sab / sqrt(saa * sbb);
            }

            double r_err2 = 0.0;
            uint32_t r_cnt = 0;
            for (uint32_t b = 0; b < nbins; b++)
            {
                if (prof_ref[b] > 1.0e-5f * (float) o_sr_meas)
                {
                    double rel = ((double) prof_rec[b] - (double) prof_ref[b]) /
                                 (double) prof_ref[b];
                    r_err2 += rel * rel;
                    r_cnt++;
                }
            }
            if (r_cnt > 0)
            {
                o_radial_rms_rel = sqrt(r_err2 / (double) r_cnt);
            }
            free(ref_2d);
        }
        free(ref_raw);
    }

    /* 10. Write FITS outputs & report */
    mlt_fits_kw kws[6] = {
        { "STREHL", 'D', o_sr_recon, 0, "", "Reconstructed Strehl peak" },
        { "SR_CTRL", 'D', o_sr_ctrl, 0, "", "Controlled-subspace Strehl peak" },
        { "SR_MEAS", 'D', o_sr_meas, 0, "", "Measured reference Strehl peak" },
        { "LAM_UM", 'D', (double) lam0, 0, "", "Central wavelength [um]" },
        { "PIX_LD", 'D', (double) ps0, 0, "", "Pixels per lambda/D" },
        { "NFRAMES", 'L', 0.0, (long) nt_used, "", "Frames integrated" },
    };
    long ax_cam[2]  = { (long) ncam, (long) ncam };
    long ax_wf[2]   = { (long) nwf, (long) nwf };
    long ax_prof[2] = { (long) nbins, 4 };
    float *prof_tbl = (float *) calloc((size_t) nbins * 4u, sizeof(float));
    for (uint32_t b = 0; b < nbins; b++)
    {
        prof_tbl[b]               = ((float) b + 0.5f) / ps0;
        prof_tbl[nbins + b]       = prof_dl[b];
        prof_tbl[2u * nbins + b]  = prof_rec[b];
        prof_tbl[3u * nbins + b]  = prof_ref[b];
    }

    char fpath[MLT_PATH_MAX];
    snprintf(fpath, sizeof(fpath), "%s/psf_recon.fits", p_outdir);
    (void) mlt_write_fits_f32(fpath, psf_recon, 2, ax_cam, kws, 6);
    snprintf(fpath, sizeof(fpath), "%s/psf_ctrl_only.fits", p_outdir);
    (void) mlt_write_fits_f32(fpath, psf_ctrl, 2, ax_cam, kws, 6);
    snprintf(fpath, sizeof(fpath), "%s/psf_dl.fits", p_outdir);
    (void) mlt_write_fits_f32(fpath, psf_dl, 2, ax_cam, kws, 6);
    snprintf(fpath, sizeof(fpath), "%s/psf_diff.fits", p_outdir);
    (void) mlt_write_fits_f32(fpath, psf_diff, 2, ax_cam, kws, 6);
    snprintf(fpath, sizeof(fpath), "%s/opd_std_nm.fits", p_outdir);
    (void) mlt_write_fits_f32(fpath, opd_std, 2, ax_wf, kws, 6);
    snprintf(fpath, sizeof(fpath), "%s/radial_profile.fits", p_outdir);
    (void) mlt_write_fits_f32(fpath, prof_tbl, 2, ax_prof, kws, 6);

    if (p_outreport[0] != '\0')
    {
        FILE *fp = fopen(p_outreport, "w");
        if (fp != NULL)
        {
            fprintf(fp, "method=%s\n", p_method);
            fprintf(fp, "lambda_um=%.4f\n", (double) lam0);
            fprintf(fp, "nframes_used=%ld\n", (long) o_nframes_used);
            fprintf(fp, "sr_recon=%.6f\n", o_sr_recon);
            fprintf(fp, "sr_ctrl=%.6f\n", o_sr_ctrl);
            fprintf(fp, "sr_meas=%.6f\n", o_sr_meas);
            fprintf(fp, "sr_abs_err=%.6f\n", o_sr_abs_err);
            fprintf(fp, "fwhm_ld=%.4f\n", o_fwhm_ld);
            fprintf(fp, "fwhm_meas_ld=%.4f\n", o_fwhm_meas_ld);
            fprintf(fp, "ee_122=%.6f\n", o_ee_122);
            fprintf(fp, "radial_rms_rel=%.6f\n", o_radial_rms_rel);
            fprintf(fp, "corr_2d=%.6f\n", o_corr_2d);
            fprintf(fp, "inband_rms_nm=%.3f\n", o_inband_rms_nm);
            fprintf(fp, "noise_rms_nm=%.3f\n", o_noise_rms_nm);
            fprintf(fp, "fitting_rms_nm=%.3f\n", o_fitting_rms_nm);
            fclose(fp);
        }
    }

    if (!dcquiet)
    {
        printf("mlt.psfrecon [%.2f um, %s]: SR_recon=%.4f (ctrl=%.4f, meas=%.4f, |err|=%.4f), "
               "r2=%.4f, inband=%.1f nm, fit=%.1f nm\n",
               (double) lam0, p_method, o_sr_recon, o_sr_ctrl, o_sr_meas, o_sr_abs_err, o_corr_2d,
               o_inband_rms_nm, o_fitting_rms_nm);
    }

    free(pup_amp);
    free(ncpa_opd);
    free(act_idx);
    free(u_act);
    free(r_mat);
    free(u_mft);
    free(v_mft);
    free(psf_dl);
    free(psf_ctrl);
    free(psf_recon);
    free(opd_std);
    free(sum_opd);
    free(sum2_opd);
    free(opd_blk);
    free(prof_dl);
    free(prof_rec);
    free(prof_ref);
    free(psf_diff);
    free(prof_tbl);
    return RETURN_SUCCESS;
}

static MILK_HOT errno_t psfrecon_compute(const MILK_RUNNABLE *self)
{
    errno_t ret = RETURN_SUCCESS;
    DEBUG_TRACE_FSTART();
    MILK_PROCINFO_INIT(self)
    MILK_PROCINFO_LOOPSTART(self)
    ret = psfrecon_run();
    MILK_PROCINFO_END(self)
    if (dcfpsptr != NULL)
    {
        fpsresync_modvar_to_fps(dcfpsptr, dcfpsbindings, dcfpsnbindings);
    }
    DEBUG_TRACE_FEXIT();
    return ret;
}

const MILK_RUNNABLE runnable_mlt_psfrecon = {
    .kind = MILK_RUNNABLE_FPS,
    .app  = { .fps_name         = "mltpsfrecon",
              .cmdkey           = "psfrecon",
              .description      = "reconstruct long-exposure AO PSF from modal telemetry",
              .description_long = "Reconstruct polychromatic AO-corrected PSFs from modal "
                                  "residual telemetry, DM modes, optical gains, and Kolmogorov "
                                  "fitting-error halos, with optional reference PSF validation." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(mlt_psfrecon),
    .compute = psfrecon_compute,
};

EXPORT_FPSEXEC(milk, "mlt-psfrecon", runnable_mlt_psfrecon);
