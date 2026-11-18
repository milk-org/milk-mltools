// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    ao_opt_unit.c
 * @brief   Closed-loop AO modal PSD, ETF, optical-gain, and optimal integrator gain analyzer
 */

// MILK_CMAKE_MANDATE_CFITSIO

#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "milk_runnable.h"
#include "ao_math.h"
#include "tlm_fileset.h"

#ifndef M_PI
#    define M_PI 3.14159265358979323846
#endif

static char    p_mval_wfs[FUNCTION_PARAMETER_STRMAXLEN]   = "";
static char    p_mval_dm[FUNCTION_PARAMETER_STRMAXLEN]    = "";
static char    p_mval_ol[FUNCTION_PARAMETER_STRMAXLEN]    = "";
static char    p_mgain[FUNCTION_PARAMETER_STRMAXLEN]      = "";
static char    p_mmult[FUNCTION_PARAMETER_STRMAXLEN]      = "";
static char    p_selfrm_cal[FUNCTION_PARAMETER_STRMAXLEN] = "";
static char    p_selfrm_sky[FUNCTION_PARAMETER_STRMAXLEN] = "";
static char    p_outdir[FUNCTION_PARAMETER_STRMAXLEN]     = ".";
static char    p_outreport[FUNCTION_PARAMETER_STRMAXLEN]  = "";
static double  p_fs_hz                                    = 2000.0;
static double  p_latency_fr                               = 2.0;
static double  p_loopgain                                 = 0.35;
static double  p_loopmult                                 = 0.98;
static double  p_ogain_scalar                             = 1.0;
static int     p_est_ogain                                = 0;
static int     p_recompute_ol                             = 1;
static int64_t p_nseg                                     = 512;
static double  p_pm_deg                                   = 45.0;
static double  p_gm_db                                    = 6.0;
static double  p_gmax                                     = 1.20;

static int64_t o_nmodes         = 0;
static int64_t o_nframes        = 0;
static double  o_rms_ol         = 0.0;
static double  o_rms_curr       = 0.0;
static double  o_rms_opt        = 0.0;
static double  o_var_red_pct    = 0.0;
static double  o_mean_ogain     = 1.0;
static double  o_tt_ogain       = 1.0;
static double  o_median_bw_hz   = 0.0;
static double  o_dom_vib_hz     = 0.0;
static int64_t o_n_overshoot    = 0;
static int64_t o_n_noise_dom    = 0;

#define FPS_PARAMS_AOOPT(X)                                                                      \
    X(".mval_wfs", p_mval_wfs, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT,                           \
      "WFS modal residuals (image/FITS/dir)")                                                    \
    X(".mval_dm", p_mval_dm, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT,                             \
      "DM modal commands (image/FITS/dir)")                                                      \
    X(".mval_ol", p_mval_ol, FPTYPE_STRING, 0, FPFLAG_DEFAULT_INPUT,                             \
      "Optional recorded pseudo-OL modes")                                                       \
    X(".mgain", p_mgain, FPTYPE_STRING, 0, FPFLAG_DEFAULT_INPUT,                                 \
      "Optional per-mode gain vector (aolX_mgain)")                                              \
    X(".mmult", p_mmult, FPTYPE_STRING, 0, FPFLAG_DEFAULT_INPUT,                                 \
      "Optional per-mode leak vector (aolX_mmult)")                                              \
    X(".selfrm_cal", p_selfrm_cal, FPTYPE_STRING, 0, FPFLAG_DEFAULT_INPUT,                       \
      "Optional calib selfRM cube [M,M,Z]")                                                      \
    X(".selfrm_sky", p_selfrm_sky, FPTYPE_STRING, 0, FPFLAG_DEFAULT_INPUT,                       \
      "Optional on-sky selfRM cube [M,M,Z]")                                                     \
    X(".outdir", p_outdir, FPTYPE_DIRNAME, 1, FPFLAG_DEFAULT_INPUT,                              \
      "Output directory for PSD/ETF/gain FITS")                                                  \
    X(".outreport", p_outreport, FPTYPE_FILENAME, 0, FPFLAG_DEFAULT_INPUT,                       \
      "Optional key=value summary report")                                                       \
    X(".fs_hz", &p_fs_hz, FPTYPE_FLOAT64, 0, FPFLAG_DEFAULT_INPUT, "Loop frame rate [Hz]")       \
    X(".latency_fr", &p_latency_fr, FPTYPE_FLOAT64, 0, FPFLAG_DEFAULT_INPUT,                     \
      "Total loop delay [frames]")                                                               \
    X(".loopgain", &p_loopgain, FPTYPE_FLOAT64, 0, FPFLAG_DEFAULT_INPUT,                         \
      "Fallback scalar loop gain")                                                               \
    X(".loopmult", &p_loopmult, FPTYPE_FLOAT64, 0, FPFLAG_DEFAULT_INPUT,                         \
      "Fallback scalar leak multiplier")                                                         \
    X(".ogain_scalar", &p_ogain_scalar, FPTYPE_FLOAT64, 0, FPFLAG_DEFAULT_INPUT,                 \
      "Fallback scalar optical gain")                                                            \
    X(".est_ogain", &p_est_ogain, FPTYPE_ONOFF, 0, FPFLAG_DEFAULT_INPUT,                         \
      "Estimate optical gain from telemetry steps if selfRM absent")                             \
    X(".recompute_ol", &p_recompute_ol, FPTYPE_ONOFF, 0, FPFLAG_DEFAULT_INPUT,                   \
      "Reconstruct OL modes using latency & optical gain")                                       \
    X(".nseg", &p_nseg, FPTYPE_INT64, 0, FPFLAG_DEFAULT_INPUT, "Welch FFT window length")        \
    X(".pm_deg", &p_pm_deg, FPTYPE_FLOAT64, 0, FPFLAG_DEFAULT_INPUT, "Min phase margin [deg]")   \
    X(".gm_db", &p_gm_db, FPTYPE_FLOAT64, 0, FPFLAG_DEFAULT_INPUT, "Min gain margin [dB]")       \
    X(".gmax", &p_gmax, FPTYPE_FLOAT64, 0, FPFLAG_DEFAULT_INPUT, "Max candidate modal gain")     \
    X(".nmodes", &o_nmodes, FPTYPE_INT64, 0, FPFLAG_DEFAULT_OUTPUT, "Number of modes analyzed")  \
    X(".nframes", &o_nframes, FPTYPE_INT64, 0, FPFLAG_DEFAULT_OUTPUT, "Frames analyzed")         \
    X(".rms_ol", &o_rms_ol, FPTYPE_FLOAT64, 0, FPFLAG_DEFAULT_OUTPUT,                            \
      "Quadrature pseudo-OL modal RMS")                                                          \
    X(".rms_curr", &o_rms_curr, FPTYPE_FLOAT64, 0, FPFLAG_DEFAULT_OUTPUT,                        \
      "Current sky-referred residual modal RMS")                                                 \
    X(".rms_opt", &o_rms_opt, FPTYPE_FLOAT64, 0, FPFLAG_DEFAULT_OUTPUT,                          \
      "Predicted optimal residual modal RMS")                                                    \
    X(".var_red_pct", &o_var_red_pct, FPTYPE_FLOAT64, 0, FPFLAG_DEFAULT_OUTPUT,                  \
      "Predicted variance reduction [%]")                                                        \
    X(".mean_ogain", &o_mean_ogain, FPTYPE_FLOAT64, 0, FPFLAG_DEFAULT_OUTPUT,                    \
      "Mean modal optical gain")                                                                 \
    X(".tt_ogain", &o_tt_ogain, FPTYPE_FLOAT64, 0, FPFLAG_DEFAULT_OUTPUT,                        \
      "Tip/Tilt optical gain")                                                                   \
    X(".median_bw_hz", &o_median_bw_hz, FPTYPE_FLOAT64, 0, FPFLAG_DEFAULT_OUTPUT,                \
      "Median 0-dB rejection bandwidth [Hz]")                                                    \
    X(".dom_vib_hz", &o_dom_vib_hz, FPTYPE_FLOAT64, 0, FPFLAG_DEFAULT_OUTPUT,                    \
      "Dominant closed-loop spectral peak [Hz]")                                                 \
    X(".n_overshoot", &o_n_overshoot, FPTYPE_INT64, 0, FPFLAG_DEFAULT_OUTPUT,                    \
      "Modes with net variance amplification")                                                   \
    X(".n_noise_dom", &o_n_noise_dom, FPTYPE_INT64, 0, FPFLAG_DEFAULT_OUTPUT,                    \
      "Measurement-noise-dominated modes")

MILK_PARAMS_DEFINE(mlt_aoopt, FPS_PARAMS_AOOPT);

static int cmp_double_asc(const void *a, const void *b)
{
    double va = *(const double *) a;
    double vb = *(const double *) b;
    return (va > vb) - (va < vb);
}

static errno_t unpack_modal_2d(const char *spec,
                               float     **buf_out,
                               uint32_t   *nmodes_out,
                               uint32_t   *nt_out)
{
    uint32_t nx = 0;
    uint32_t ny = 0;
    uint32_t nz = 0;
    if (mlt_load_array_f32(spec, NULL, buf_out, &nx, &ny, &nz) != RETURN_SUCCESS)
    {
        return RETURN_FAILURE;
    }
    if (nz > 1u && ny == 1u)
    {
        *nmodes_out = nx;
        *nt_out     = nz;
    }
    else if (nz == 1u && ny >= 1u)
    {
        *nmodes_out = nx;
        *nt_out     = ny;
    }
    else
    {
        free(*buf_out);
        *buf_out = NULL;
        fprintf(stderr, "ERROR: modal telemetry '%s' must be (M,N) or (M,1,N), got %ux%ux%u\n",
                spec, nx, ny, nz);
        return RETURN_FAILURE;
    }
    return RETURN_SUCCESS;
}

static void extract_selfrm_ogain(const char *cal_spec,
                                 const char *sky_spec,
                                 uint32_t    nmodes,
                                 double      fallback_og,
                                 float      *ogain_out,
                                 int        *used_selfrm)
{
    *used_selfrm = 0;
    for (uint32_t m = 0; m < nmodes; m++)
    {
        ogain_out[m] = (float) fallback_og;
    }
    if (cal_spec == NULL || cal_spec[0] == '\0' || sky_spec == NULL || sky_spec[0] == '\0')
    {
        return;
    }

    float   *cal = NULL;
    float   *sky = NULL;
    uint32_t cx = 0, cy = 0, cz = 0;
    uint32_t sx = 0, sy = 0, sz = 0;
    if (mlt_load_array_f32(cal_spec, NULL, &cal, &cx, &cy, &cz) != RETURN_SUCCESS ||
        mlt_load_array_f32(sky_spec, NULL, &sky, &sx, &sy, &sz) != RETURN_SUCCESS)
    {
        free(cal);
        free(sky);
        return;
    }

    uint32_t mlim = nmodes;
    if (cx < mlim)
    {
        mlim = cx;
    }
    if (cy < mlim)
    {
        mlim = cy;
    }
    if (sx < mlim)
    {
        mlim = sx;
    }
    if (sy < mlim)
    {
        mlim = sy;
    }

    double tail_sum = 0.0;
    int    tail_cnt = 0;
    int    n_valid  = 0;

    for (uint32_t m = 0; m < mlim; m++)
    {
        double pk_c = 0.0;
        double pk_s = 0.0;
        for (uint32_t z = 0; z < cz; z++)
        {
            double v = fabs((double) cal[(size_t) z * cx * cy + (size_t) m * cx + m]);
            if (v > pk_c)
            {
                pk_c = v;
            }
        }
        for (uint32_t z = 0; z < sz; z++)
        {
            double v = fabs((double) sky[(size_t) z * sx * sy + (size_t) m * sx + m]);
            if (v > pk_s)
            {
                pk_s = v;
            }
        }
        if (pk_c > 1.0e-4 && pk_s > 1.0e-5)
        {
            double ratio = pk_s / pk_c;
            if (ratio < 0.08)
            {
                ratio = 0.08;
            }
            if (ratio > 1.50)
            {
                ratio = 1.50;
            }
            ogain_out[m] = (float) ratio;
            n_valid++;
            if (m >= (mlim * 3u) / 4u)
            {
                tail_sum += ratio;
                tail_cnt++;
            }
        }
    }

    if (n_valid > 0)
    {
        *used_selfrm = 1;
        float tail_og = (tail_cnt > 0) ? (float) (tail_sum / (double) tail_cnt) : ogain_out[0];
        for (uint32_t m = 0; m < nmodes; m++)
        {
            if (m >= mlim || ogain_out[m] == (float) fallback_og)
            {
                /* Check whether mode m was actually unprobed */
                double pk_c = 0.0;
                if (m < mlim)
                {
                    for (uint32_t z = 0; z < cz; z++)
                    {
                        double v = fabs((double) cal[(size_t) z * cx * cy + (size_t) m * cx + m]);
                        if (v > pk_c)
                        {
                            pk_c = v;
                        }
                    }
                }
                if (pk_c <= 1.0e-4)
                {
                    ogain_out[m] = tail_og;
                }
            }
        }
    }

    free(cal);
    free(sky);
}

/* Check stability margins & evaluate closed-loop sky variance cost J(g) for one mode */
static int eval_gain_cost(double        g,
                          double        alpha,
                          double        ogain,
                          uint32_t      nfreq,
                          const double *cos_wt,
                          const double *sin_wt,
                          const double *cos_w,
                          const double *sin_w,
                          const float  *s_turb,
                          double        s_noise_sky,
                          double        df,
                          double        min_gm_lin,
                          double        min_pm_rad,
                          double       *cost_out)
{
    double prev_ang  = 0.0;
    double prev_mag  = 0.0;
    int    have_prev = 0;
    double var_sum   = 0.0;

    for (uint32_t k = 1u; k < nfreq; k++)
    {
        /* Open-loop L(e^{iw}) = (ogain * g * e^{-i*w*tau}) / (1 - alpha*e^{-iw}) */
        double dr = 1.0 - alpha * cos_w[k];
        double di = alpha * sin_w[k];
        double d2 = dr * dr + di * di;
        if (d2 < 1.0e-18)
        {
            return 0;
        }

        double nr   = ogain * g * cos_wt[k];
        double ni   = -ogain * g * sin_wt[k];
        double lr   = (nr * dr + ni * di) / d2;
        double li   = (ni * dr - nr * di) / d2;
        double lmag = sqrt(lr * lr + li * li);
        double lang = atan2(li, lr); /* (-pi, pi] */

        /* Unwrap negative phase crossing -pi for gain margin check */
        if (have_prev && prev_ang < -1.0 && lang > 1.0)
        {
            double u_prev = prev_ang;
            double u_curr = lang - 2.0 * M_PI;
            double frac   = (-M_PI - u_prev) / (u_curr - u_prev);
            double m_x    = prev_mag + frac * (lmag - prev_mag);
            if (m_x * min_gm_lin >= 1.0)
            {
                return 0;
            }
        }
        /* Unity-gain crossover check for phase margin */
        if (have_prev && prev_mag >= 1.0 && lmag < 1.0)
        {
            double frac = (prev_mag - 1.0) / (prev_mag - lmag + 1.0e-30);
            double ang  = prev_ang + frac * (lang - prev_ang);
            double pm   = M_PI + ang;
            if (pm < min_pm_rad)
            {
                return 0;
            }
        }
        prev_ang  = lang;
        prev_mag  = lmag;
        have_prev = 1;

        /* Closed-loop denominator D_cl = (1 - alpha*z^{-1}) + ogain*g*z^{-tau} */
        double clr = dr + nr;
        double cli = di + ni;
        double cl2 = clr * clr + cli * cli;
        if (cl2 < 1.0e-15)
        {
            return 0;
        }

        double etf2 = d2 / cl2;
        double ntf2 = (ogain * g * ogain * g) / cl2;
        var_sum += ((double) s_turb[k] * etf2 + s_noise_sky * ntf2) * df;
    }

    *cost_out = var_sum;
    return 1;
}

static errno_t aoopt_run(void)
{
    float   *wfs_raw = NULL;
    float   *dm_raw  = NULL;
    float   *ol_raw  = NULL;
    uint32_t m_wfs = 0, t_wfs = 0;
    uint32_t m_dm = 0, t_dm = 0;

    if (unpack_modal_2d(p_mval_wfs, &wfs_raw, &m_wfs, &t_wfs) != RETURN_SUCCESS)
    {
        return RETURN_FAILURE;
    }
    if (unpack_modal_2d(p_mval_dm, &dm_raw, &m_dm, &t_dm) != RETURN_SUCCESS)
    {
        free(wfs_raw);
        return RETURN_FAILURE;
    }
    if (m_wfs != m_dm || m_wfs == 0)
    {
        fprintf(stderr, "ERROR: mode count mismatch WFS (%u) vs DM (%u)\n", m_wfs, m_dm);
        free(wfs_raw);
        free(dm_raw);
        return RETURN_FAILURE;
    }

    uint32_t nmodes = m_wfs;
    uint32_t nt     = (t_wfs < t_dm) ? t_wfs : t_dm;
    if (p_mval_ol[0] != '\0' && !p_recompute_ol)
    {
        uint32_t m_ol = 0, t_ol = 0;
        if (unpack_modal_2d(p_mval_ol, &ol_raw, &m_ol, &t_ol) == RETURN_SUCCESS)
        {
            if (m_ol == nmodes && t_ol >= 16u)
            {
                if (t_ol < nt)
                {
                    nt = t_ol;
                }
            }
            else
            {
                free(ol_raw);
                ol_raw = NULL;
            }
        }
    }

    if (nt < 32u)
    {
        fprintf(stderr, "ERROR: need >= 32 modal frames for PSD/AO optimization, got %u\n", nt);
        free(wfs_raw);
        free(dm_raw);
        free(ol_raw);
        return RETURN_FAILURE;
    }

    /* Load optional per-mode gain and leak vectors */
    float *g_curr = (float *) malloc(nmodes * sizeof(float));
    float *a_curr = (float *) malloc(nmodes * sizeof(float));
    float *ogain  = (float *) malloc(nmodes * sizeof(float));
    if (g_curr == NULL || a_curr == NULL || ogain == NULL)
    {
        free(wfs_raw);
        free(dm_raw);
        free(ol_raw);
        free(g_curr);
        free(a_curr);
        free(ogain);
        return RETURN_FAILURE;
    }
    for (uint32_t m = 0; m < nmodes; m++)
    {
        g_curr[m] = (float) p_loopgain;
        a_curr[m] = (float) p_loopmult;
    }

    if (p_mgain[0] != '\0')
    {
        float   *tmp = NULL;
        uint32_t gx = 0, gy = 0, gz = 0;
        if (mlt_load_array_f32(p_mgain, NULL, &tmp, &gx, &gy, &gz) == RETURN_SUCCESS)
        {
            uint32_t n = (gx * gy * gz < nmodes) ? (gx * gy * gz) : nmodes;
            memcpy(g_curr, tmp, n * sizeof(float));
            free(tmp);
        }
    }
    if (p_mmult[0] != '\0')
    {
        float   *tmp = NULL;
        uint32_t mx = 0, my = 0, mz = 0;
        if (mlt_load_array_f32(p_mmult, NULL, &tmp, &mx, &my, &mz) == RETURN_SUCCESS)
        {
            uint32_t n = (mx * my * mz < nmodes) ? (mx * my * mz) : nmodes;
            memcpy(a_curr, tmp, n * sizeof(float));
            free(tmp);
        }
    }

    double fb_og      = (p_ogain_scalar > 0.05) ? p_ogain_scalar : 1.0;
    int    has_selfrm = 0;
    extract_selfrm_ogain(p_selfrm_cal, p_selfrm_sky, nmodes, fb_og, ogain, &has_selfrm);

    /* Welch segment geometry */
    uint32_t nseg = (p_nseg >= 32) ? mlt_next_pow2_u32((uint32_t) p_nseg) : 256u;
    while (nseg > nt && nseg > 32u)
    {
        nseg >>= 1u;
    }
    uint32_t step  = nseg / 2u;
    uint32_t nwin  = (nt >= nseg) ? (1u + (nt - nseg) / step) : 1u;
    uint32_t nfreq = nseg / 2u + 1u;
    double   fs    = (p_fs_hz > 1.0) ? p_fs_hz : 2000.0;
    double   df    = fs / (double) nseg;

    float *hann  = (float *) malloc(nseg * sizeof(float));
    double w_pow = 0.0;
    for (uint32_t i = 0; i < nseg; i++)
    {
        double w = 0.5 * (1.0 - cos(2.0 * M_PI * (double) i / (double) nseg));
        hann[i]  = (float) w;
        w_pow += w * w;
    }
    double psd_scale = 1.0 / (fs * w_pow * (double) nwin);

    /* Precompute frequency trigonometric tables for fast transfer evaluation */
    double *cos_w  = (double *) malloc(nfreq * sizeof(double));
    double *sin_w  = (double *) malloc(nfreq * sizeof(double));
    double *cos_wt = (double *) malloc(nfreq * sizeof(double));
    double *sin_wt = (double *) malloc(nfreq * sizeof(double));
    double  tau    = (p_latency_fr >= 0.1) ? p_latency_fr : 2.0;
    for (uint32_t k = 0; k < nfreq; k++)
    {
        double w   = 2.0 * M_PI * ((double) k * df) / fs;
        cos_w[k]   = cos(w);
        sin_w[k]   = sin(w);
        cos_wt[k]  = cos(w * tau);
        sin_wt[k]  = sin(w * tau);
    }

    size_t nfm      = (size_t) nfreq * nmodes;
    float *psd_wfs  = (float *) calloc(nfm, sizeof(float));
    float *psd_ol   = (float *) calloc(nfm, sizeof(float));
    float *psd_dm   = (float *) calloc(nfm, sizeof(float));
    float *etf_emp  = (float *) calloc(nfm, sizeof(float));
    float *etf_opt  = (float *) calloc(nfm, sizeof(float));
    float *g_opt    = (float *) calloc(nmodes, sizeof(float));
    float *g_fact   = (float *) calloc(nmodes, sizeof(float));
    float *summary  = (float *) calloc((size_t) nmodes * 10u, sizeof(float));
    double *bw_arr  = (double *) calloc(nmodes, sizeof(double));
    double *psd_tot = (double *) calloc(nfreq, sizeof(double));

    if (!psd_wfs || !psd_ol || !psd_dm || !etf_emp || !etf_opt || !g_opt || !g_fact || !summary ||
        !bw_arr || !psd_tot)
    {
        free(wfs_raw);
        free(dm_raw);
        free(ol_raw);
        free(g_curr);
        free(a_curr);
        free(ogain);
        free(hann);
        free(cos_w);
        free(sin_w);
        free(cos_wt);
        free(sin_wt);
        free(psd_wfs);
        free(psd_ol);
        free(psd_dm);
        free(etf_emp);
        free(etf_opt);
        free(g_opt);
        free(g_fact);
        free(summary);
        free(bw_arr);
        free(psd_tot);
        return RETURN_FAILURE;
    }

    const int    d_int      = (int) floor(tau);
    const double d_frac     = tau - (double) d_int;
    const double min_gm_lin = pow(10.0, p_gm_db / 20.0);
    const double min_pm_rad = p_pm_deg * (M_PI / 180.0);
    const double gmax       = (p_gmax > 0.05) ? p_gmax : 1.0;

    double  sum_var_ol   = 0.0;
    double  sum_var_curr = 0.0;
    double  sum_var_opt  = 0.0;
    double  sum_ogain    = 0.0;
    int64_t cnt_over     = 0;
    int64_t cnt_noise    = 0;

#pragma omp parallel reduction(+ : sum_var_ol, sum_var_curr, sum_var_opt, sum_ogain, cnt_over, \
                                   cnt_noise)
    {
        float *ts_wfs  = (float *) malloc(nt * sizeof(float));
        float *ts_dm   = (float *) malloc(nt * sizeof(float));
        float *ts_ol   = (float *) malloc(nt * sizeof(float));
        float *fft_buf = (float *) malloc((size_t) nseg * 2u * sizeof(float));
        float *s_turb  = (float *) malloc(nfreq * sizeof(float));
        double *loc_pt = (double *) calloc(nfreq, sizeof(double));

        uint32_t m;
#pragma omp for schedule(static)
        for (m = 0; m < nmodes; m++)
        {
            double mean_w = 0.0;
            double mean_d = 0.0;
            for (uint32_t t = 0; t < nt; t++)
            {
                float vw = wfs_raw[(size_t) t * nmodes + m];
                float vd = dm_raw[(size_t) t * nmodes + m];
                ts_wfs[t] = vw;
                ts_dm[t]  = vd;
                mean_w += (double) vw;
                mean_d += (double) vd;
            }
            mean_w /= (double) nt;
            mean_d /= (double) nt;

            /* Optional telemetry-step optical gain estimation when selfRM was not supplied */
            if (!has_selfrm && p_est_ogain)
            {
                double num = 0.0;
                double den = 0.0;
                uint32_t t0_s = (uint32_t) (d_int + 3);
                for (uint32_t t = t0_s; t < nt; t++)
                {
                    double d_dm0 = (1.0 - d_frac) * (double) ts_dm[t - d_int] +
                                   d_frac * (double) ts_dm[t - d_int - 1];
                    double d_dm1 = (1.0 - d_frac) * (double) ts_dm[t - d_int - 1] +
                                   d_frac * (double) ts_dm[t - d_int - 2];
                    double step_dm  = d_dm0 - d_dm1;
                    double step_wfs = (double) ts_wfs[t] - (double) ts_wfs[t - 1];
                    num += step_wfs * step_dm;
                    den += step_dm * step_dm;
                }
                if (den > 1.0e-12)
                {
                    double est = num / den;
                    if (est >= 0.10 && est <= 1.40)
                    {
                        ogain[m] = (float) est;
                    }
                }
            }

            double og     = (ogain[m] > 0.05f) ? (double) ogain[m] : 1.0;
            double inv_og = 1.0 / og;
            sum_ogain += og;

            /* Reconstruct or copy pseudo-open-loop sequence in true sky modal units */
            uint32_t t_valid0 = (uint32_t) (d_int + 2);
            if (ol_raw != NULL && !p_recompute_ol)
            {
                for (uint32_t t = 0; t < nt; t++)
                {
                    ts_ol[t] = ol_raw[(size_t) t * nmodes + m];
                }
            }
            else
            {
                for (uint32_t t = 0; t < nt; t++)
                {
                    double dm_del = 0.0;
                    if (t >= t_valid0)
                    {
                        dm_del = (1.0 - d_frac) * (double) ts_dm[t - d_int] +
                                 d_frac * (double) ts_dm[t - d_int - 1];
                    }
                    else
                    {
                        dm_del = (double) ts_dm[0];
                    }
                    ts_ol[t] = (float) (inv_og * (double) ts_wfs[t] - dm_del);
                }
            }

            /* Remove DC means & compute time-domain variances + 2nd-difference noise floor */
            double mean_o = 0.0;
            for (uint32_t t = 0; t < nt; t++)
            {
                mean_o += (double) ts_ol[t];
            }
            mean_o /= (double) nt;

            double var_w = 0.0;
            double var_d = 0.0;
            double var_o = 0.0;
            double d4_w  = 0.0;
            for (uint32_t t = 0; t < nt; t++)
            {
                double w0 = (double) ts_wfs[t] - mean_w;
                double d0 = (double) ts_dm[t] - mean_d;
                double o0 = (double) ts_ol[t] - mean_o;
                ts_wfs[t] = (float) w0;
                ts_dm[t]  = (float) d0;
                ts_ol[t]  = (float) o0;
                var_w += w0 * w0;
                var_d += d0 * d0;
                var_o += o0 * o0;
                if (t >= 4u)
                {
                    double dd = (double) ts_wfs[t] - 4.0 * (double) ts_wfs[t - 1] +
                                6.0 * (double) ts_wfs[t - 2] - 4.0 * (double) ts_wfs[t - 3] +
                                (double) ts_wfs[t - 4];
                    d4_w += dd * dd;
                }
            }
            var_w /= (double) nt;
            var_d /= (double) nt;
            var_o /= (double) nt;
            double noise_var_wfs = (nt > 4u) ? (d4_w / (70.0 * (double) (nt - 4u))) : 0.0;
            if (noise_var_wfs > 0.95 * var_w)
            {
                noise_var_wfs = 0.95 * var_w;
            }
            double noise_var_sky = noise_var_wfs * inv_og * inv_og;
            double sky_var_curr  = var_w * inv_og * inv_og;

            /* Compute Welch PSDs for WFS (sky-referred), OL, and DM */
            const float *seqs[3] = { ts_wfs, ts_ol, ts_dm };
            float       *outs[3] = { psd_wfs, psd_ol, psd_dm };
            double       scs[3]  = { psd_scale * inv_og * inv_og, psd_scale, psd_scale };

            for (int s = 0; s < 3; s++)
            {
                for (uint32_t w = 0; w < nwin; w++)
                {
                    uint32_t off = w * step;
                    for (uint32_t i = 0; i < nseg; i++)
                    {
                        fft_buf[2u * i]      = seqs[s][off + i] * hann[i];
                        fft_buf[2u * i + 1u] = 0.0f;
                    }
                    mlt_fft1d_c2c(fft_buf, nseg, 0);
                    for (uint32_t k = 0; k < nfreq; k++)
                    {
                        double re  = (double) fft_buf[2u * k];
                        double im  = (double) fft_buf[2u * k + 1u];
                        double pwr = (re * re + im * im) * scs[s];
                        if (k > 0u && k < nseg / 2u)
                        {
                            pwr *= 2.0;
                        }
                        outs[s][(size_t) m * nfreq + k] += (float) pwr;
                    }
                }
            }

            /* Empirical ETF, 0-dB rejection bandwidth, vibration peak, & clean turbulence PSD */
            double s_noise_sky = noise_var_sky / (0.5 * fs);
            double bw_hz       = 0.0;
            double vib_hz      = 0.0;
            double vib_pk      = 0.0;

            for (uint32_t k = 0; k < nfreq; k++)
            {
                double pw = (double) psd_wfs[(size_t) m * nfreq + k];
                double po = (double) psd_ol[(size_t) m * nfreq + k];
                double et = (po > 1.0e-30) ? (pw / po) : 1.0;
                etf_emp[(size_t) m * nfreq + k] = (float) et;

                if (k >= 1u)
                {
                    loc_pt[k] += pw;
                    double f_k = (double) k * df;
                    if (f_k >= 5.0 && pw > vib_pk)
                    {
                        vib_pk = pw;
                        vib_hz = f_k;
                    }
                }
                double st = po - s_noise_sky;
                s_turb[k] = (float) ((st > 0.0) ? st : 0.0);
            }
            for (uint32_t k = 1u; k < nfreq; k++)
            {
                if (etf_emp[(size_t) m * nfreq + k] >= 1.0f)
                {
                    double e0   = (double) etf_emp[(size_t) m * nfreq + k - 1u];
                    double e1   = (double) etf_emp[(size_t) m * nfreq + k];
                    double frac = (e1 > e0) ? ((1.0 - e0) / (e1 - e0)) : 0.0;
                    if (frac < 0.0)
                    {
                        frac = 0.0;
                    }
                    if (frac > 1.0)
                    {
                        frac = 1.0;
                    }
                    bw_hz = ((double) (k - 1u) + frac) * df;
                    break;
                }
            }
            bw_arr[m] = bw_hz;

            /* Grid + golden-section search for stability-constrained optimal modal gain */
            double alpha     = (double) a_curr[m];
            double best_g    = 0.0;
            double best_cost = var_o;
            (void) eval_gain_cost(0.0, alpha, og, nfreq, cos_wt, sin_wt, cos_w, sin_w, s_turb,
                                  s_noise_sky, df, min_gm_lin, min_pm_rad, &best_cost);

            const int ngrid    = 48;
            double    g_hi_ok  = 0.0;
            for (int gi = 1; gi <= ngrid; gi++)
            {
                double gc   = gmax * (double) gi / (double) ngrid;
                double cost = 0.0;
                if (!eval_gain_cost(gc, alpha, og, nfreq, cos_wt, sin_wt, cos_w, sin_w, s_turb,
                                    s_noise_sky, df, min_gm_lin, min_pm_rad, &cost))
                {
                    break;
                }
                g_hi_ok = gc;
                if (cost < best_cost)
                {
                    best_cost = cost;
                    best_g    = gc;
                }
            }

            if (g_hi_ok > 0.0)
            {
                double step_g = gmax / (double) ngrid;
                double lo     = (best_g > step_g) ? (best_g - step_g) : 0.0;
                double hi     = (best_g + step_g < g_hi_ok) ? (best_g + step_g) : g_hi_ok;
                const double gr = 0.6180339887498949;
                double x1 = hi - gr * (hi - lo);
                double x2 = lo + gr * (hi - lo);
                double f1 = 1e30, f2 = 1e30;
                (void) eval_gain_cost(x1, alpha, og, nfreq, cos_wt, sin_wt, cos_w, sin_w, s_turb,
                                      s_noise_sky, df, min_gm_lin, min_pm_rad, &f1);
                (void) eval_gain_cost(x2, alpha, og, nfreq, cos_wt, sin_wt, cos_w, sin_w, s_turb,
                                      s_noise_sky, df, min_gm_lin, min_pm_rad, &f2);
                for (int it = 0; it < 14; it++)
                {
                    if (f1 < f2)
                    {
                        hi = x2;
                        x2 = x1;
                        f2 = f1;
                        x1 = hi - gr * (hi - lo);
                        if (!eval_gain_cost(x1, alpha, og, nfreq, cos_wt, sin_wt, cos_w, sin_w,
                                            s_turb, s_noise_sky, df, min_gm_lin, min_pm_rad, &f1))
                        {
                            f1 = 1e30;
                        }
                    }
                    else
                    {
                        lo = x1;
                        x1 = x2;
                        f1 = f2;
                        x2 = lo + gr * (hi - lo);
                        if (!eval_gain_cost(x2, alpha, og, nfreq, cos_wt, sin_wt, cos_w, sin_w,
                                            s_turb, s_noise_sky, df, min_gm_lin, min_pm_rad, &f2))
                        {
                            f2 = 1e30;
                        }
                    }
                }
                if (f1 < best_cost)
                {
                    best_cost = f1;
                    best_g    = x1;
                }
                if (f2 < best_cost)
                {
                    best_cost = f2;
                    best_g    = x2;
                }
            }

            /* Evaluate analytical ETF at best_g & compare against current sky residual variance */
            for (uint32_t k = 0; k < nfreq; k++)
            {
                double dr  = 1.0 - alpha * cos_w[k];
                double di  = alpha * sin_w[k];
                double d2  = dr * dr + di * di;
                double clr = dr + og * best_g * cos_wt[k];
                double cli = di - og * best_g * sin_wt[k];
                double cl2 = clr * clr + cli * cli;
                etf_opt[(size_t) m * nfreq + k] = (float) ((cl2 > 1e-15) ? (d2 / cl2) : 1.0);
            }

            double curr_model_cost = sky_var_curr;
            (void) eval_gain_cost((double) g_curr[m], alpha, og, nfreq, cos_wt, sin_wt, cos_w,
                                  sin_w, s_turb, s_noise_sky, df, 1.001, 0.01, &curr_model_cost);
            double pred_sky_var = sky_var_curr;
            if (curr_model_cost > 1.0e-30 && best_cost < curr_model_cost)
            {
                pred_sky_var = sky_var_curr * (best_cost / curr_model_cost);
            }

            g_opt[m]  = (float) best_g;
            g_fact[m] = (p_loopgain > 1.0e-9) ? (float) (best_g / p_loopgain) : (float) best_g;

            if (sky_var_curr > var_o && var_o > 1.0e-20)
            {
                cnt_over++;
            }
            if (noise_var_wfs > 0.5 * var_w)
            {
                cnt_noise++;
            }

            sum_var_ol += var_o;
            sum_var_curr += sky_var_curr;
            sum_var_opt += pred_sky_var;

            float *row = &summary[(size_t) m * 10u];
            row[0]     = (float) sqrt(var_o);
            row[1]     = (float) sqrt(sky_var_curr);
            row[2]     = (float) sqrt(var_d);
            row[3]     = (float) sqrt(noise_var_sky);
            row[4]     = (float) og;
            row[5]     = g_curr[m];
            row[6]     = g_opt[m];
            row[7]     = (float) sqrt(pred_sky_var);
            row[8]     = (float) bw_hz;
            row[9]     = (float) vib_hz;
        }

#pragma omp critical
        {
            for (uint32_t k = 0; k < nfreq; k++)
            {
                psd_tot[k] += loc_pt[k];
            }
        }

        free(ts_wfs);
        free(ts_dm);
        free(ts_ol);
        free(fft_buf);
        free(s_turb);
        free(loc_pt);
    }

    qsort(bw_arr, nmodes, sizeof(double), cmp_double_asc);
    double med_bw = bw_arr[nmodes / 2u];

    double dom_pk = 0.0;
    double dom_hz = 0.0;
    for (uint32_t k = 1u; k < nfreq; k++)
    {
        double f_k = (double) k * df;
        if (f_k >= 5.0 && psd_tot[k] > dom_pk)
        {
            dom_pk = psd_tot[k];
            dom_hz = f_k;
        }
    }

    o_nmodes       = (int64_t) nmodes;
    o_nframes      = (int64_t) nt;
    o_rms_ol       = sqrt(sum_var_ol);
    o_rms_curr     = sqrt(sum_var_curr);
    o_rms_opt      = sqrt(sum_var_opt);
    o_var_red_pct  = (sum_var_curr > 1e-30)
                         ? (100.0 * (sum_var_curr - sum_var_opt) / sum_var_curr)
                         : 0.0;
    o_mean_ogain   = sum_ogain / (double) nmodes;
    o_tt_ogain     = (nmodes >= 2u) ? (0.5 * ((double) ogain[0] + (double) ogain[1]))
                                    : (double) ogain[0];
    o_median_bw_hz = med_bw;
    o_dom_vib_hz   = dom_hz;
    o_n_overshoot  = cnt_over;
    o_n_noise_dom  = cnt_noise;

    /* Write FITS diagnostic products */
    mlt_fits_kw kws[4] = {
        { "FS_HZ", 'D', fs, 0, "", "Loop sampling rate [Hz]" },
        { "DF_HZ", 'D', df, 0, "", "Welch PSD frequency bin spacing [Hz]" },
        { "LAT_FR", 'D', tau, 0, "", "Total loop latency [frames]" },
        { "NFRAMES", 'L', 0.0, (long) nt, "", "Frames analyzed" },
    };
    long ax_psd[2]  = { (long) nfreq, (long) nmodes };
    long ax_vec[2]  = { (long) nmodes, 1 };
    long ax_sum[2]  = { 10, (long) nmodes };
    char fpath[MLT_PATH_MAX];

    snprintf(fpath, sizeof(fpath), "%s/psd_wfs.fits", p_outdir);
    (void) mlt_write_fits_f32(fpath, psd_wfs, 2, ax_psd, kws, 4);
    snprintf(fpath, sizeof(fpath), "%s/psd_ol.fits", p_outdir);
    (void) mlt_write_fits_f32(fpath, psd_ol, 2, ax_psd, kws, 4);
    snprintf(fpath, sizeof(fpath), "%s/psd_dm.fits", p_outdir);
    (void) mlt_write_fits_f32(fpath, psd_dm, 2, ax_psd, kws, 4);
    snprintf(fpath, sizeof(fpath), "%s/etf_emp.fits", p_outdir);
    (void) mlt_write_fits_f32(fpath, etf_emp, 2, ax_psd, kws, 4);
    snprintf(fpath, sizeof(fpath), "%s/etf_opt.fits", p_outdir);
    (void) mlt_write_fits_f32(fpath, etf_opt, 2, ax_psd, kws, 4);
    snprintf(fpath, sizeof(fpath), "%s/ogain_est.fits", p_outdir);
    (void) mlt_write_fits_f32(fpath, ogain, 2, ax_vec, kws, 4);
    snprintf(fpath, sizeof(fpath), "%s/opt_mgain.fits", p_outdir);
    (void) mlt_write_fits_f32(fpath, g_opt, 2, ax_vec, kws, 4);
    snprintf(fpath, sizeof(fpath), "%s/opt_mgainfact.fits", p_outdir);
    (void) mlt_write_fits_f32(fpath, g_fact, 2, ax_vec, kws, 4);
    snprintf(fpath, sizeof(fpath), "%s/mode_summary.fits", p_outdir);
    (void) mlt_write_fits_f32(fpath, summary, 2, ax_sum, kws, 4);

    if (p_outreport[0] != '\0')
    {
        FILE *fp = fopen(p_outreport, "w");
        if (fp != NULL)
        {
            fprintf(fp, "nmodes=%ld\n", (long) o_nmodes);
            fprintf(fp, "nframes=%ld\n", (long) o_nframes);
            fprintf(fp, "fs_hz=%.4f\n", fs);
            fprintf(fp, "latency_fr=%.4f\n", tau);
            fprintf(fp, "rms_ol=%.6f\n", o_rms_ol);
            fprintf(fp, "rms_curr=%.6f\n", o_rms_curr);
            fprintf(fp, "rms_opt=%.6f\n", o_rms_opt);
            fprintf(fp, "var_red_pct=%.4f\n", o_var_red_pct);
            fprintf(fp, "mean_ogain=%.6f\n", o_mean_ogain);
            fprintf(fp, "tt_ogain=%.6f\n", o_tt_ogain);
            fprintf(fp, "median_bw_hz=%.4f\n", o_median_bw_hz);
            fprintf(fp, "dom_vib_hz=%.4f\n", o_dom_vib_hz);
            fprintf(fp, "n_overshoot=%ld\n", (long) o_n_overshoot);
            fprintf(fp, "n_noise_dom=%ld\n", (long) o_n_noise_dom);
            fclose(fp);
        }
    }

    if (!dcquiet)
    {
        printf("mlt.aoopt: M=%ld N=%ld  rms(OL->curr->opt)=%.4f -> %.4f -> %.4f "
               "(dVar=%.1f%%, og=%.3f, bw=%.1f Hz)\n",
               (long) o_nmodes, (long) o_nframes, o_rms_ol, o_rms_curr, o_rms_opt, o_var_red_pct,
               o_mean_ogain, o_median_bw_hz);
    }

    free(wfs_raw);
    free(dm_raw);
    free(ol_raw);
    free(g_curr);
    free(a_curr);
    free(ogain);
    free(hann);
    free(cos_w);
    free(sin_w);
    free(cos_wt);
    free(sin_wt);
    free(psd_wfs);
    free(psd_ol);
    free(psd_dm);
    free(etf_emp);
    free(etf_opt);
    free(g_opt);
    free(g_fact);
    free(summary);
    free(bw_arr);
    free(psd_tot);
    return RETURN_SUCCESS;
}

static MILK_HOT errno_t aoopt_compute(const MILK_RUNNABLE *self)
{
    errno_t ret = RETURN_SUCCESS;
    DEBUG_TRACE_FSTART();
    MILK_PROCINFO_INIT(self)
    MILK_PROCINFO_LOOPSTART(self)
    ret = aoopt_run();
    MILK_PROCINFO_END(self)
    if (dcfpsptr != NULL)
    {
        fpsresync_modvar_to_fps(dcfpsptr, dcfpsbindings, dcfpsnbindings);
    }
    DEBUG_TRACE_FEXIT();
    return ret;
}

const MILK_RUNNABLE runnable_mlt_aoopt = {
    .kind = MILK_RUNNABLE_FPS,
    .app  = { .fps_name         = "mltaoopt",
              .cmdkey           = "aoopt",
              .description      = "analyze AO modal PSDs, ETFs, optical gains, & optimal gains",
              .description_long = "Compute per-mode Welch PSDs (WFS, pseudo-OL, DM), empirical "
                                  "rejection transfer functions, Pyramid WFS optical gains, and "
                                  "stability-margin-constrained variance-optimal modal gains." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(mlt_aoopt),
    .compute = aoopt_compute,
};

EXPORT_FPSEXEC(milk, "mlt-aoopt", runnable_mlt_aoopt);
