// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    sysid_unit.c
 * @brief   Two-WFS + inter-stage DM linear system identification (`B = Mab A + Mdb D`)
 *
 * Jointly calibrates:
 *   1. Forward model:          B = Mab * A + Mdb * D + b0 + eps_B
 *   2. Virtual upstream model: A = Mba * B - Mda_virt * D + a0 + eps_A
 * using an identical scalar causal AR(p) whitening filter W(z) applied to all three aligned
 * streams (A, B, D) to break closed-loop low-frequency collinearity without biasing Mab or Mdb,
 * followed by block-regularized Schur-complement normal equations with contiguous held-out CV.
 */

// MILK_CMAKE_MANDATE_CFITSIO
// MILK_CMAKE_MANDATE_BLAS
// MILK_CMAKE_MANDATE_LAPACKE

#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "milk_blas_lapacke.h"
#include "milk_runnable.h"
#include "ao_math.h"
#include "tlm_fileset.h"

#define SYSID_MAX_AR_ORDER 64

static char    p_a[FUNCTION_PARAMETER_STRMAXLEN]          = "";
static char    p_b[FUNCTION_PARAMETER_STRMAXLEN]          = "";
static char    p_d[FUNCTION_PARAMETER_STRMAXLEN]          = "";
static char    p_outdir[FUNCTION_PARAMETER_STRMAXLEN]     = ".";
static char    p_outreport[FUNCTION_PARAMETER_STRMAXLEN]  = "";
static char    p_selfrm_cal[FUNCTION_PARAMETER_STRMAXLEN] = "";
static char    p_selfrm_sky[FUNCTION_PARAMETER_STRMAXLEN] = "";
static int64_t p_max_modes_a                              = 0;
static int64_t p_max_modes_b                              = 0;
static int64_t p_max_modes_d                              = 0;
static int64_t p_whiten_order                             = 4;
static double  p_whiten_ridge                             = 0.02;
static int     p_whiten_normch                            = 1;
static double  p_train_frac                               = 0.70;
static double  p_reg_a                                    = -1.0;
static double  p_reg_d                                    = -1.0;
static double  p_reg_b                                    = -1.0;
static double  p_lag_ab_ms                                = 0.0;
static double  p_lag_db_ms                                = 0.0;
static double  p_lag_ab_fr                                = 0.0;
static double  p_lag_db_fr                                = 0.0;
static double  p_fs_hz                                    = 1000.0;

static int64_t o_nframes       = 0;
static int64_t o_n_train       = 0;
static int64_t o_n_test        = 0;
static int64_t o_dim_a         = 0;
static int64_t o_dim_b         = 0;
static int64_t o_dim_d         = 0;
static double  o_reg_a_opt     = 0.0;
static double  o_reg_d_opt     = 0.0;
static double  o_eta_d_raw     = 0.0;
static double  o_eta_d_white   = 0.0;
static double  o_mdb_diag_raw  = 0.0;
static double  o_mdb_diag_med  = 0.0;
static double  o_mdb_diag_iqr  = 0.0;
static double  o_mdb_offdiag   = 0.0;
static double  o_r2_b_w_med    = 0.0;
static double  o_r2_ol_raw_med = 0.0;
static double  o_r2_a_raw_med  = 0.0;
static double  o_r2_a_w_med    = 0.0;
static double  o_cross_rel_err = 0.0;
static double  o_selfrm_og_med = 0.0;
static double  o_selfrm_err    = 0.0;
static double  o_b0_rms        = 0.0;

#define FPS_PARAMS_SYSID(X)                                                                      \
    X(".a", p_a, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT,                                         \
      "Aligned upstream WFS A matrix (dA, N) image/FITS")                                        \
    X(".b", p_b, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT,                                         \
      "Aligned downstream WFS B matrix (dB, N) image/FITS")                                      \
    X(".d", p_d, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT,                                         \
      "Aligned inter-stage DM D matrix (dD, N) image/FITS")                                      \
    X(".outdir", p_outdir, FPTYPE_DIRNAME, 1, FPFLAG_DEFAULT_INPUT,                              \
      "Output directory for Mab, Mdb, Mda_virt, & diagnostics")                                  \
    X(".outreport", p_outreport, FPTYPE_FILENAME, 0, FPFLAG_DEFAULT_INPUT,                       \
      "Optional key=value summary report")                                                       \
    X(".selfrm_cal", p_selfrm_cal, FPTYPE_STRING, 0, FPFLAG_DEFAULT_INPUT,                       \
      "Optional calibration selfRM cube [M,M,Z] for optical-gain check")                         \
    X(".selfrm_sky", p_selfrm_sky, FPTYPE_STRING, 0, FPFLAG_DEFAULT_INPUT,                       \
      "Optional on-sky selfRM cube [M,M,Z] for optical-gain check")                              \
    X(".max_modes_a", &p_max_modes_a, FPTYPE_INT64, 0, FPFLAG_DEFAULT_INPUT,                     \
      "Max leading channels of A to retain (0=all)")                                             \
    X(".max_modes_b", &p_max_modes_b, FPTYPE_INT64, 0, FPFLAG_DEFAULT_INPUT,                     \
      "Max leading channels of B to retain (0=all)")                                             \
    X(".max_modes_d", &p_max_modes_d, FPTYPE_INT64, 0, FPFLAG_DEFAULT_INPUT,                     \
      "Max leading channels of D to retain (0=all)")                                             \
    X(".whiten_order", &p_whiten_order, FPTYPE_INT64, 0, FPFLAG_DEFAULT_INPUT,                   \
      "Causal AR(p) whitening order W(z) (0=off)")                                               \
    X(".whiten_ridge", &p_whiten_ridge, FPTYPE_FLOAT64, 0, FPFLAG_DEFAULT_INPUT,                 \
      "Ridge on normalized AR autocorrelation")                                                  \
    X(".whiten_normch", &p_whiten_normch, FPTYPE_ONOFF, 0, FPFLAG_DEFAULT_INPUT,                 \
      "Unit-variance channel pooling in AR(p) fit")                                              \
    X(".train_frac", &p_train_frac, FPTYPE_FLOAT64, 0, FPFLAG_DEFAULT_INPUT,                     \
      "Contiguous temporal training fraction [0.2..0.95]")                                       \
    X(".reg_a", &p_reg_a, FPTYPE_FLOAT64, 0, FPFLAG_DEFAULT_INPUT,                               \
      "Relative diagonal ridge for A (<0 = auto CV)")                                            \
    X(".reg_d", &p_reg_d, FPTYPE_FLOAT64, 0, FPFLAG_DEFAULT_INPUT,                               \
      "Relative diagonal ridge for D (<0 = auto CV)")                                            \
    X(".reg_b", &p_reg_b, FPTYPE_FLOAT64, 0, FPFLAG_DEFAULT_INPUT,                               \
      "Relative diagonal ridge for B in virtual A fit (<0 = match reg_a)")                       \
    X(".lag_ab_ms", &p_lag_ab_ms, FPTYPE_FLOAT64, 0, FPFLAG_DEFAULT_INPUT,                       \
      "Recorded A->B alignment lag [ms] for metadata")                                           \
    X(".lag_db_ms", &p_lag_db_ms, FPTYPE_FLOAT64, 0, FPFLAG_DEFAULT_INPUT,                       \
      "Recorded D->B alignment lag [ms] for metadata")                                           \
    X(".lag_ab_fr", &p_lag_ab_fr, FPTYPE_FLOAT64, 0, FPFLAG_DEFAULT_INPUT,                       \
      "Recorded A->B alignment lag [B-frames] for metadata")                                     \
    X(".lag_db_fr", &p_lag_db_fr, FPTYPE_FLOAT64, 0, FPFLAG_DEFAULT_INPUT,                       \
      "Recorded D->B alignment lag [B-frames] for metadata")                                     \
    X(".fs_hz", &p_fs_hz, FPTYPE_FLOAT64, 0, FPFLAG_DEFAULT_INPUT,                               \
      "Effective common sampling rate [Hz] for metadata")                                        \
    X(".nframes", &o_nframes, FPTYPE_INT64, 0, FPFLAG_DEFAULT_OUTPUT, "Total aligned frames")    \
    X(".n_train", &o_n_train, FPTYPE_INT64, 0, FPFLAG_DEFAULT_OUTPUT, "Training split frames")   \
    X(".n_test", &o_n_test, FPTYPE_INT64, 0, FPFLAG_DEFAULT_OUTPUT, "Held-out test frames")      \
    X(".dim_a", &o_dim_a, FPTYPE_INT64, 0, FPFLAG_DEFAULT_OUTPUT, "Active A channels")           \
    X(".dim_b", &o_dim_b, FPTYPE_INT64, 0, FPFLAG_DEFAULT_OUTPUT, "Active B channels")           \
    X(".dim_d", &o_dim_d, FPTYPE_INT64, 0, FPFLAG_DEFAULT_OUTPUT, "Active D channels")           \
    X(".reg_a_opt", &o_reg_a_opt, FPTYPE_FLOAT64, 0, FPFLAG_DEFAULT_OUTPUT, "Selected reg_a")    \
    X(".reg_d_opt", &o_reg_d_opt, FPTYPE_FLOAT64, 0, FPFLAG_DEFAULT_OUTPUT, "Selected reg_d")    \
    X(".eta_d_raw", &o_eta_d_raw, FPTYPE_FLOAT64, 0, FPFLAG_DEFAULT_OUTPUT,                      \
      "Unwhitened median conditional excitation eta_D")                                          \
    X(".eta_d_white", &o_eta_d_white, FPTYPE_FLOAT64, 0, FPFLAG_DEFAULT_OUTPUT,                  \
      "Whitened median conditional excitation eta_D")                                            \
    X(".mdb_diag_raw", &o_mdb_diag_raw, FPTYPE_FLOAT64, 0, FPFLAG_DEFAULT_OUTPUT,                \
      "Unwhitened median diag(Mdb)")                                                             \
    X(".mdb_diag_med", &o_mdb_diag_med, FPTYPE_FLOAT64, 0, FPFLAG_DEFAULT_OUTPUT,                \
      "Whitened median diag(Mdb)")                                                               \
    X(".mdb_diag_iqr", &o_mdb_diag_iqr, FPTYPE_FLOAT64, 0, FPFLAG_DEFAULT_OUTPUT,                \
      "Whitened IQR of diag(Mdb)")                                                               \
    X(".mdb_offdiag", &o_mdb_offdiag, FPTYPE_FLOAT64, 0, FPFLAG_DEFAULT_OUTPUT,                  \
      "Off-diagonal RMS of Mdb")                                                                 \
    X(".r2_b_w_med", &o_r2_b_w_med, FPTYPE_FLOAT64, 0, FPFLAG_DEFAULT_OUTPUT,                    \
      "Held-out median whitened B R2")                                                           \
    X(".r2_ol_raw_med", &o_r2_ol_raw_med, FPTYPE_FLOAT64, 0, FPFLAG_DEFAULT_OUTPUT,              \
      "Held-out median OL consistency R2 (B - Mdb D vs Mab A)")                                  \
    X(".r2_a_raw_med", &o_r2_a_raw_med, FPTYPE_FLOAT64, 0, FPFLAG_DEFAULT_OUTPUT,                \
      "Held-out median virtual upstream A raw R2")                                               \
    X(".r2_a_w_med", &o_r2_a_w_med, FPTYPE_FLOAT64, 0, FPFLAG_DEFAULT_OUTPUT,                    \
      "Held-out median virtual upstream A whitened R2")                                          \
    X(".cross_rel_err", &o_cross_rel_err, FPTYPE_FLOAT64, 0, FPFLAG_DEFAULT_OUTPUT,              \
      "Relative ||Mab Mda_virt - Mdb||_F / ||Mdb||_F")                                           \
    X(".selfrm_og_med", &o_selfrm_og_med, FPTYPE_FLOAT64, 0, FPFLAG_DEFAULT_OUTPUT,              \
      "Median selfRM optical gain (0 if unavailable)")                                           \
    X(".selfrm_err", &o_selfrm_err, FPTYPE_FLOAT64, 0, FPFLAG_DEFAULT_OUTPUT,                    \
      "Relative |mdb_diag_med - selfrm_og_med| / selfrm_og_med")                                 \
    X(".b0_rms", &o_b0_rms, FPTYPE_FLOAT64, 0, FPFLAG_DEFAULT_OUTPUT,                            \
      "RMS of static differential NCPA offset b0")

MILK_PARAMS_DEFINE(mlt_sysid, FPS_PARAMS_SYSID);

static int cmp_double_asc(const void *a, const void *b)
{
    double va = *(const double *) a;
    double vb = *(const double *) b;
    return (va > vb) - (va < vb);
}

static double quantile_sorted(const double *sorted, size_t n, double q)
{
    if (n == 0)
    {
        return 0.0;
    }
    if (n == 1)
    {
        return sorted[0];
    }
    double pos = q * (double) (n - 1);
    size_t i0  = (size_t) floor(pos);
    size_t i1  = (i0 + 1 < n) ? (i0 + 1) : (n - 1);
    double f   = pos - (double) i0;
    return (1.0 - f) * sorted[i0] + f * sorted[i1];
}

static void stats_median_iqr(const double *arr, size_t n, double *med_out, double *iqr_out)
{
    *med_out = 0.0;
    if (iqr_out != NULL)
    {
        *iqr_out = 0.0;
    }
    if (n == 0)
    {
        return;
    }
    double *tmp = (double *) malloc(n * sizeof(double));
    if (tmp == NULL)
    {
        return;
    }
    size_t nf = 0;
    for (size_t i = 0; i < n; i++)
    {
        if (isfinite(arr[i]))
        {
            tmp[nf++] = arr[i];
        }
    }
    if (nf > 0)
    {
        qsort(tmp, nf, sizeof(double), cmp_double_asc);
        *med_out = quantile_sorted(tmp, nf, 0.5);
        if (iqr_out != NULL)
        {
            *iqr_out = quantile_sorted(tmp, nf, 0.75) - quantile_sorted(tmp, nf, 0.25);
        }
    }
    free(tmp);
}

static errno_t load_stream_mat_f64(const char *spec,
                                   int64_t     max_modes,
                                   double    **mat_out,
                                   size_t     *nframes_out,
                                   size_t     *dim_out)
{
    float   *raw = NULL;
    uint32_t nx  = 0;
    uint32_t ny  = 0;
    uint32_t nz  = 0;
    if (mlt_load_array_f32(spec, NULL, &raw, &nx, &ny, &nz) != RETURN_SUCCESS)
    {
        return RETURN_FAILURE;
    }

    size_t dim_in = 0;
    size_t nfr    = 0;
    if (nz > 1u)
    {
        dim_in = (size_t) nx * (size_t) ny;
        nfr    = (size_t) nz;
    }
    else
    {
        dim_in = (size_t) nx;
        nfr    = (size_t) ny;
    }
    if (dim_in == 0 || nfr < 16)
    {
        free(raw);
        fprintf(stderr, "ERROR: sysid stream '%s' has invalid shape (%zu x %zu)\n", spec, dim_in,
                nfr);
        return RETURN_FAILURE;
    }

    size_t dim = dim_in;
    if (max_modes > 0 && (size_t) max_modes < dim)
    {
        dim = (size_t) max_modes;
    }

    double *mat = (double *) malloc(nfr * dim * sizeof(double));
    if (mat == NULL)
    {
        free(raw);
        return RETURN_FAILURE;
    }

    size_t t;
#pragma omp parallel for schedule(static) if (nfr * dim > 16384)
    for (t = 0; t < nfr; t++)
    {
        const float *src = raw + t * dim_in;
        double      *dst = mat + t * dim;
        for (size_t c = 0; c < dim; c++)
        {
            double v = (double) src[c];
            dst[c]   = isfinite(v) ? v : 0.0;
        }
    }

    free(raw);
    *mat_out     = mat;
    *nframes_out = nfr;
    *dim_out     = dim;
    return RETURN_SUCCESS;
}

static void compute_col_means(const double *X, size_t n, size_t d, double *mu_out)
{
    memset(mu_out, 0, d * sizeof(double));
    if (n == 0)
    {
        return;
    }
    for (size_t t = 0; t < n; t++)
    {
        const double *row = X + t * d;
        for (size_t c = 0; c < d; c++)
        {
            mu_out[c] += row[c];
        }
    }
    double inv_n = 1.0 / (double) n;
    for (size_t c = 0; c < d; c++)
    {
        mu_out[c] *= inv_n;
    }
}

static void demean_slice(const double *X, size_t n, size_t d, const double *mu, double *out)
{
    size_t t;
#pragma omp parallel for schedule(static) if (n * d > 16384)
    for (t = 0; t < n; t++)
    {
        const double *src = X + t * d;
        double       *dst = out + t * d;
        for (size_t c = 0; c < d; c++)
        {
            dst[c] = src[c] - mu[c];
        }
    }
}

static void fit_pooled_ar_taps(const double *Xc,
                               size_t        n,
                               size_t        d,
                               int           order,
                               double        ridge,
                               int           normch,
                               double       *taps_out)
{
    taps_out[0] = 1.0;
    for (int m = 1; m <= order; m++)
    {
        taps_out[m] = 0.0;
    }
    if (order <= 0 || n <= (size_t) (2 * order))
    {
        return;
    }

    double r[SYSID_MAX_AR_ORDER + 1];
    memset(r, 0, sizeof(r));

    for (size_t c = 0; c < d; c++)
    {
        double scale = 1.0;
        if (normch)
        {
            double e0 = 0.0;
            for (size_t t = 0; t < n; t++)
            {
                double v = Xc[t * d + c];
                e0 += v * v;
            }
            if (e0 <= 1e-24)
            {
                continue;
            }
            scale = 1.0 / e0;
        }
        for (int m = 0; m <= order; m++)
        {
            double acc = 0.0;
            for (size_t t = (size_t) m; t < n; t++)
            {
                acc += Xc[t * d + c] * Xc[(t - (size_t) m) * d + c];
            }
            r[m] += acc * scale;
        }
    }

    if (r[0] <= 1e-30)
    {
        return;
    }
    double r0 = r[0];
    for (int m = 0; m <= order; m++)
    {
        r[m] /= r0;
    }
    r[0] = 1.0 + ((ridge > 1e-12) ? ridge : 1e-12);

    double a_cur[SYSID_MAX_AR_ORDER + 1];
    double a_nxt[SYSID_MAX_AR_ORDER + 1];
    memset(a_cur, 0, sizeof(a_cur));
    memset(a_nxt, 0, sizeof(a_nxt));

    double err = r[0];
    for (int m = 1; m <= order; m++)
    {
        double num = r[m];
        for (int j = 1; j < m; j++)
        {
            num -= a_cur[j] * r[m - j];
        }
        double km = (err > 1e-18) ? (num / err) : 0.0;
        a_nxt[m]  = km;
        for (int j = 1; j < m; j++)
        {
            a_nxt[j] = a_cur[j] - km * a_cur[m - j];
        }
        memcpy(a_cur, a_nxt, (size_t) (m + 1) * sizeof(double));
        err *= (1.0 - km * km);
        if (err <= 1e-18)
        {
            break;
        }
    }

    taps_out[0] = 1.0;
    for (int m = 1; m <= order; m++)
    {
        taps_out[m] = -a_cur[m];
    }
}

/* Apply causal FIR taps[0..order] to centered Xc (n x d), writing (n - order) x d valid rows. */
static void apply_fir_valid(const double *Xc,
                            size_t        n,
                            size_t        d,
                            const double *taps,
                            int           order,
                            double       *out)
{
    size_t n_eff = (n > (size_t) order) ? (n - (size_t) order) : 0;
    size_t i;
#pragma omp parallel for schedule(static) if (n_eff * d > 16384)
    for (i = 0; i < n_eff; i++)
    {
        size_t  t   = i + (size_t) order;
        double *dst = out + i * d;
        for (size_t c = 0; c < d; c++)
        {
            double acc = 0.0;
            for (int m = 0; m <= order; m++)
            {
                acc += taps[m] * Xc[(t - (size_t) m) * d + c];
            }
            dst[c] = acc;
        }
    }
}

static void add_diag_ridge(const double *S, size_t d, double reg, double *S_out)
{
    memcpy(S_out, S, d * d * sizeof(double));
    double tr = 0.0;
    for (size_t i = 0; i < d; i++)
    {
        tr += S[i * d + i];
    }
    double avg   = (d > 0) ? (tr / (double) d) : 1.0;
    double floor = 1e-12 * ((avg > 0.0) ? avg : 1.0);
    if (floor < 1e-18)
    {
        floor = 1e-18;
    }
    double r = (reg > 1e-12) ? reg : 1e-12;
    for (size_t i = 0; i < d; i++)
    {
        double di = S[i * d + i];
        S_out[i * d + i] += r * ((di > floor) ? di : floor);
    }
}

static int solve_spd_robust(const double *S_in, size_t d, size_t nrhs, double *rhs_inout)
{
    double *A = (double *) malloc(d * d * sizeof(double));
    if (A == NULL)
    {
        return -1;
    }
    memcpy(A, S_in, d * d * sizeof(double));
    lapack_int info = LAPACKE_dposv(LAPACK_ROW_MAJOR, 'U', (lapack_int) d, (lapack_int) nrhs, A,
                                    (lapack_int) d, rhs_inout, (lapack_int) nrhs);
    if (info != 0)
    {
        double tr = 0.0;
        for (size_t i = 0; i < d; i++)
        {
            tr += S_in[i * d + i];
        }
        double jitter = 1e-6 * ((d > 0 && tr > 0.0) ? (tr / (double) d) : 1.0);
        memcpy(A, S_in, d * d * sizeof(double));
        for (size_t i = 0; i < d; i++)
        {
            A[i * d + i] += jitter;
        }
        info = LAPACKE_dposv(LAPACK_ROW_MAJOR, 'U', (lapack_int) d, (lapack_int) nrhs, A,
                             (lapack_int) d, rhs_inout, (lapack_int) nrhs);
    }
    free(A);
    return (int) info;
}

static void transpose_f64(const double *src, size_t rows, size_t cols, double *dst)
{
    for (size_t r = 0; r < rows; r++)
    {
        for (size_t c = 0; c < cols; c++)
        {
            dst[c * rows + r] = src[r * cols + c];
        }
    }
}

/*
 * Solve block-regularized normal equations for Y = M1 * X1 + M2 * X2 using Schur complement:
 *   M1: (dY x d1), M2: (dY x d2)
 *   eta2_out (optional, length d2): conditional excitation diag(Schur) / diag(S22)
 */
static int solve_block_schur(const double *S11,
                             const double *S22,
                             const double *S12,
                             const double *SY1,
                             const double *SY2,
                             size_t        d1,
                             size_t        d2,
                             size_t        dY,
                             double        reg1,
                             double        reg2,
                             double       *M1_out,
                             double       *M2_out,
                             double       *eta2_out)
{
    double *S11_r = (double *) malloc(d1 * d1 * sizeof(double));
    double *S22_r = (double *) malloc(d2 * d2 * sizeof(double));
    double *G12   = (double *) malloc(d1 * d2 * sizeof(double));
    double *Sch   = (double *) malloc(d2 * d2 * sizeof(double));
    double *rhs2  = (double *) malloc(d2 * dY * sizeof(double));
    double *rhs1  = (double *) malloc(d1 * dY * sizeof(double));
    double *tmp2Y = (double *) malloc(dY * d2 * sizeof(double));
    double *tmp1Y = (double *) malloc(dY * d1 * sizeof(double));

    if (!S11_r || !S22_r || !G12 || !Sch || !rhs2 || !rhs1 || !tmp2Y || !tmp1Y)
    {
        free(S11_r);
        free(S22_r);
        free(G12);
        free(Sch);
        free(rhs2);
        free(rhs1);
        free(tmp2Y);
        free(tmp1Y);
        return -1;
    }

    add_diag_ridge(S11, d1, reg1, S11_r);
    add_diag_ridge(S22, d2, reg2, S22_r);

    /* G12 = S11_r^{-1} * S12  (d1 x d2) */
    memcpy(G12, S12, d1 * d2 * sizeof(double));
    int rc = solve_spd_robust(S11_r, d1, d2, G12);
    if (rc != 0)
    {
        goto cleanup;
    }

    /* Sch = S22_r - S12^T * G12  (d2 x d2) */
    memcpy(Sch, S22_r, d2 * d2 * sizeof(double));
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, (int) d2, (int) d2, (int) d1, -1.0, S12,
                (int) d2, G12, (int) d2, 1.0, Sch, (int) d2);

    if (eta2_out != NULL)
    {
        for (size_t j = 0; j < d2; j++)
        {
            double s2_jj = S22[j * d2 + j];
            double sc_jj = Sch[j * d2 + j];
            double ratio = (s2_jj > 1e-18) ? (sc_jj / s2_jj) : 0.0;
            if (ratio < 0.0)
            {
                ratio = 0.0;
            }
            if (ratio > 1.0)
            {
                ratio = 1.0;
            }
            eta2_out[j] = ratio;
        }
    }

    /* tmp2Y = SY2 - SY1 * G12  (dY x d2); solve Sch * M2^T = tmp2Y^T */
    memcpy(tmp2Y, SY2, dY * d2 * sizeof(double));
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int) dY, (int) d2, (int) d1, -1.0, SY1,
                (int) d1, G12, (int) d2, 1.0, tmp2Y, (int) d2);
    transpose_f64(tmp2Y, dY, d2, rhs2);
    rc = solve_spd_robust(Sch, d2, dY, rhs2);
    if (rc != 0)
    {
        goto cleanup;
    }
    transpose_f64(rhs2, d2, dY, M2_out);

    /* tmp1Y = SY1 - M2 * S12^T  (dY x d1); solve S11_r * M1^T = tmp1Y^T */
    memcpy(tmp1Y, SY1, dY * d1 * sizeof(double));
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, (int) dY, (int) d1, (int) d2, -1.0, M2_out,
                (int) d2, S12, (int) d2, 1.0, tmp1Y, (int) d1);
    transpose_f64(tmp1Y, dY, d1, rhs1);
    rc = solve_spd_robust(S11_r, d1, dY, rhs1);
    if (rc != 0)
    {
        goto cleanup;
    }
    transpose_f64(rhs1, d1, dY, M1_out);

cleanup:
    free(S11_r);
    free(S22_r);
    free(G12);
    free(Sch);
    free(rhs2);
    free(rhs1);
    free(tmp2Y);
    free(tmp1Y);
    return rc;
}

static void accum_covs(const double *X1,
                       const double *X2,
                       const double *Y,
                       size_t        n,
                       size_t        d1,
                       size_t        d2,
                       size_t        dY,
                       double       *S11,
                       double       *S22,
                       double       *SYY,
                       double       *S12,
                       double       *SY1,
                       double       *SY2)
{
    double inv_n = (n > 1) ? (1.0 / (double) (n - 1)) : 1.0;
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, (int) d1, (int) d1, (int) n, inv_n, X1,
                (int) d1, X1, (int) d1, 0.0, S11, (int) d1);
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, (int) d2, (int) d2, (int) n, inv_n, X2,
                (int) d2, X2, (int) d2, 0.0, S22, (int) d2);
    if (SYY != NULL)
    {
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, (int) dY, (int) dY, (int) n, inv_n, Y,
                    (int) dY, Y, (int) dY, 0.0, SYY, (int) dY);
    }
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, (int) d1, (int) d2, (int) n, inv_n, X1,
                (int) d1, X2, (int) d2, 0.0, S12, (int) d2);
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, (int) dY, (int) d1, (int) n, inv_n, Y,
                (int) dY, X1, (int) d1, 0.0, SY1, (int) d1);
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, (int) dY, (int) d2, (int) n, inv_n, Y,
                (int) dY, X2, (int) d2, 0.0, SY2, (int) d2);
}

/* Compute per-channel R2 of Y vs (X1 * M1^T + X2 * M2^T) and return median R2. */
static double eval_r2_two(const double *X1,
                          const double *X2,
                          const double *Y,
                          const double *M1,
                          const double *M2,
                          size_t        n,
                          size_t        d1,
                          size_t        d2,
                          size_t        dY,
                          double       *r2_per_ch_out)
{
    double *pred = (double *) malloc(n * dY * sizeof(double));
    if (pred == NULL)
    {
        return -1e9;
    }
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, (int) n, (int) dY, (int) d1, 1.0, X1,
                (int) d1, M1, (int) d1, 0.0, pred, (int) dY);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, (int) n, (int) dY, (int) d2, 1.0, X2,
                (int) d2, M2, (int) d2, 1.0, pred, (int) dY);

    double *r2_loc = (r2_per_ch_out != NULL) ? r2_per_ch_out
                                             : (double *) malloc(dY * sizeof(double));
    if (r2_loc == NULL)
    {
        free(pred);
        return -1e9;
    }

    for (size_t c = 0; c < dY; c++)
    {
        double sse = 0.0;
        double sst = 0.0;
        for (size_t t = 0; t < n; t++)
        {
            double yv = Y[t * dY + c];
            double dv = yv - pred[t * dY + c];
            sse += dv * dv;
            sst += yv * yv;
        }
        r2_loc[c] = (sst > 1e-24) ? (1.0 - sse / sst) : 0.0;
    }

    double med = 0.0;
    stats_median_iqr(r2_loc, dY, &med, NULL);
    if (r2_per_ch_out == NULL)
    {
        free(r2_loc);
    }
    free(pred);
    return med;
}

/* Compute per-channel OL consistency R2: (B - D * Mdb^T) vs (A * Mab^T). */
static double eval_r2_ol(const double *A,
                         const double *B,
                         const double *D,
                         const double *Mab,
                         const double *Mdb,
                         size_t        n,
                         size_t        dA,
                         size_t        dB,
                         size_t        dD,
                         double       *r2_per_ch_out)
{
    double *ol   = (double *) malloc(n * dB * sizeof(double));
    double *aprd = (double *) malloc(n * dB * sizeof(double));
    if (ol == NULL || aprd == NULL)
    {
        free(ol);
        free(aprd);
        return 0.0;
    }
    memcpy(ol, B, n * dB * sizeof(double));
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, (int) n, (int) dB, (int) dD, -1.0, D,
                (int) dD, Mdb, (int) dD, 1.0, ol, (int) dB);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, (int) n, (int) dB, (int) dA, 1.0, A,
                (int) dA, Mab, (int) dA, 0.0, aprd, (int) dB);

    for (size_t c = 0; c < dB; c++)
    {
        double sse = 0.0;
        double sst = 0.0;
        for (size_t t = 0; t < n; t++)
        {
            double ov = ol[t * dB + c];
            double dv = ov - aprd[t * dB + c];
            sse += dv * dv;
            sst += ov * ov;
        }
        r2_per_ch_out[c] = (sst > 1e-24) ? (1.0 - sse / sst) : 0.0;
    }
    double med = 0.0;
    stats_median_iqr(r2_per_ch_out, dB, &med, NULL);
    free(ol);
    free(aprd);
    return med;
}

static void write_f64_as_fits(const char        *path,
                              const double      *data,
                              int                naxis,
                              const long        *naxes,
                              const mlt_fits_kw *kws,
                              size_t             nkws)
{
    size_t ntot = 1;
    for (int i = 0; i < naxis; i++)
    {
        ntot *= (size_t) naxes[i];
    }
    float *buf = (float *) malloc(ntot * sizeof(float));
    if (buf == NULL)
    {
        return;
    }
    for (size_t i = 0; i < ntot; i++)
    {
        buf[i] = (float) data[i];
    }
    (void) mlt_write_fits_f32(path, buf, naxis, naxes, kws, nkws);
    free(buf);
}

static int load_selfrm_og(const char *cal_path,
                          const char *sky_path,
                          size_t      nmodes,
                          double     *og_out)
{
    if (cal_path == NULL || cal_path[0] == '\0' || sky_path == NULL || sky_path[0] == '\0')
    {
        return 0;
    }
    float   *cal = NULL;
    float   *sky = NULL;
    uint32_t cx  = 0;
    uint32_t cy  = 0;
    uint32_t cz  = 0;
    uint32_t sx  = 0;
    uint32_t sy  = 0;
    uint32_t sz  = 0;
    if (mlt_load_array_f32(cal_path, NULL, &cal, &cx, &cy, &cz) != RETURN_SUCCESS)
    {
        return 0;
    }
    if (mlt_load_array_f32(sky_path, NULL, &sky, &sx, &sy, &sz) != RETURN_SUCCESS)
    {
        free(cal);
        return 0;
    }

    size_t mlim = nmodes;
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

    int nvalid = 0;
    for (size_t m = 0; m < nmodes; m++)
    {
        og_out[m] = 0.0;
    }
    for (size_t m = 0; m < mlim; m++)
    {
        double pk_c = 0.0;
        double pk_s = 0.0;
        for (uint32_t z = 0; z < cz; z++)
        {
            double v = fabs((double) cal[(size_t) z * cx * cy + m * cx + m]);
            if (v > pk_c)
            {
                pk_c = v;
            }
        }
        for (uint32_t z = 0; z < sz; z++)
        {
            double v = fabs((double) sky[(size_t) z * sx * sy + m * sx + m]);
            if (v > pk_s)
            {
                pk_s = v;
            }
        }
        if (pk_c > 1e-4 && pk_s > 1e-5)
        {
            og_out[m] = pk_s / pk_c;
            nvalid++;
        }
    }
    free(cal);
    free(sky);
    return (nvalid > 0) ? 1 : 0;
}

static errno_t sysid_run(void)
{
    double *A  = NULL;
    double *B  = NULL;
    double *D  = NULL;
    size_t  nA = 0;
    size_t  nB = 0;
    size_t  nD = 0;
    size_t  dA = 0;
    size_t  dB = 0;
    size_t  dD = 0;

    if (load_stream_mat_f64(p_a, p_max_modes_a, &A, &nA, &dA) != RETURN_SUCCESS ||
        load_stream_mat_f64(p_b, p_max_modes_b, &B, &nB, &dB) != RETURN_SUCCESS ||
        load_stream_mat_f64(p_d, p_max_modes_d, &D, &nD, &dD) != RETURN_SUCCESS)
    {
        free(A);
        free(B);
        free(D);
        return RETURN_FAILURE;
    }

    size_t N = nA;
    if (nB < N)
    {
        N = nB;
    }
    if (nD < N)
    {
        N = nD;
    }
    if (N < 32)
    {
        fprintf(stderr, "ERROR: sysid requires >= 32 aligned frames, got %zu\n", N);
        free(A);
        free(B);
        free(D);
        return RETURN_FAILURE;
    }

    double tf = p_train_frac;
    if (tf < 0.2)
    {
        tf = 0.2;
    }
    if (tf > 0.95)
    {
        tf = 0.95;
    }
    size_t n_tr = (size_t) floor((double) N * tf);
    if (n_tr < 16)
    {
        n_tr = 16;
    }
    if (n_tr > N - 12)
    {
        n_tr = N - 12;
    }
    size_t n_te = N - n_tr;

    int w_ord = (int) p_whiten_order;
    if (w_ord < 0)
    {
        w_ord = 0;
    }
    if (w_ord > SYSID_MAX_AR_ORDER)
    {
        w_ord = SYSID_MAX_AR_ORDER;
    }
    if ((size_t) w_ord >= n_te / 3)
    {
        w_ord = (int) (n_te / 4);
    }

    double *muA_tr = (double *) calloc(dA, sizeof(double));
    double *muB_tr = (double *) calloc(dB, sizeof(double));
    double *muD_tr = (double *) calloc(dD, sizeof(double));
    double *muA_fl = (double *) calloc(dA, sizeof(double));
    double *muB_fl = (double *) calloc(dB, sizeof(double));
    double *muD_fl = (double *) calloc(dD, sizeof(double));

    double *Ac_tr = (double *) malloc(n_tr * dA * sizeof(double));
    double *Bc_tr = (double *) malloc(n_tr * dB * sizeof(double));
    double *Dc_tr = (double *) malloc(n_tr * dD * sizeof(double));
    double *Ac_te = (double *) malloc(n_te * dA * sizeof(double));
    double *Bc_te = (double *) malloc(n_te * dB * sizeof(double));
    double *Dc_te = (double *) malloc(n_te * dD * sizeof(double));
    double *Ac_fl = (double *) malloc(N * dA * sizeof(double));
    double *Bc_fl = (double *) malloc(N * dB * sizeof(double));
    double *Dc_fl = (double *) malloc(N * dD * sizeof(double));

    size_t n_tr_w = n_tr - (size_t) w_ord;
    size_t n_te_w = n_te - (size_t) w_ord;
    size_t n_fl_w = N - (size_t) w_ord;

    double *Aw_tr = (double *) malloc(n_tr_w * dA * sizeof(double));
    double *Bw_tr = (double *) malloc(n_tr_w * dB * sizeof(double));
    double *Dw_tr = (double *) malloc(n_tr_w * dD * sizeof(double));
    double *Aw_te = (double *) malloc(n_te_w * dA * sizeof(double));
    double *Bw_te = (double *) malloc(n_te_w * dB * sizeof(double));
    double *Dw_te = (double *) malloc(n_te_w * dD * sizeof(double));
    double *Aw_fl = (double *) malloc(n_fl_w * dA * sizeof(double));
    double *Bw_fl = (double *) malloc(n_fl_w * dB * sizeof(double));
    double *Dw_fl = (double *) malloc(n_fl_w * dD * sizeof(double));

    double *Saa = (double *) malloc(dA * dA * sizeof(double));
    double *Sdd = (double *) malloc(dD * dD * sizeof(double));
    double *Sbb = (double *) malloc(dB * dB * sizeof(double));
    double *Sad = (double *) malloc(dA * dD * sizeof(double));
    double *Sba = (double *) malloc(dB * dA * sizeof(double));
    double *Sbd = (double *) malloc(dB * dD * sizeof(double));

    size_t  d_diag       = (dB < dD) ? dB : dD;
    double *Mab_tr       = (double *) calloc(dB * dA, sizeof(double));
    double *Mdb_tr       = (double *) calloc(dB * dD, sizeof(double));
    double *Mab_raw      = (double *) calloc(dB * dA, sizeof(double));
    double *Mdb_raw      = (double *) calloc(dB * dD, sizeof(double));
    double *Mab_fl       = (double *) calloc(dB * dA, sizeof(double));
    double *Mdb_fl       = (double *) calloc(dB * dD, sizeof(double));
    double *Mba_tr       = (double *) calloc(dA * dB, sizeof(double));
    double *Mda_tr       = (double *) calloc(dA * dD, sizeof(double));
    double *Mba_fl       = (double *) calloc(dA * dB, sizeof(double));
    double *Mda_fl       = (double *) calloc(dA * dD, sizeof(double));
    double *Mda_virt     = (double *) calloc(dA * dD, sizeof(double));
    double *eta_d_raw    = (double *) calloc(dD, sizeof(double));
    double *eta_d_white  = (double *) calloc(dD, sizeof(double));
    double *mdb_diag_raw = (double *) calloc(d_diag, sizeof(double));
    double *mdb_diag     = (double *) calloc(d_diag, sizeof(double));
    double *r2_b_w       = (double *) calloc(dB, sizeof(double));
    double *r2_ol_raw    = (double *) calloc(dB, sizeof(double));
    double *r2_a_raw     = (double *) calloc(dA, sizeof(double));
    double *r2_a_w       = (double *) calloc(dA, sizeof(double));
    double *b0           = (double *) calloc(dB, sizeof(double));
    double *a0           = (double *) calloc(dA, sizeof(double));
    double *og_selfrm    = (double *) calloc(d_diag, sizeof(double));

    if (!muA_tr || !muB_tr || !muD_tr || !muA_fl || !muB_fl || !muD_fl || !Ac_tr || !Bc_tr ||
        !Dc_tr || !Ac_te || !Bc_te || !Dc_te || !Ac_fl || !Bc_fl || !Dc_fl || !Aw_tr || !Bw_tr ||
        !Dw_tr || !Aw_te || !Bw_te || !Dw_te || !Aw_fl || !Bw_fl || !Dw_fl || !Saa || !Sdd ||
        !Sbb || !Sad || !Sba || !Sbd || !Mab_tr || !Mdb_tr || !Mab_raw || !Mdb_raw || !Mab_fl ||
        !Mdb_fl || !Mba_tr || !Mda_tr || !Mba_fl || !Mda_fl || !Mda_virt || !eta_d_raw ||
        !eta_d_white || !mdb_diag_raw || !mdb_diag || !r2_b_w || !r2_ol_raw || !r2_a_raw ||
        !r2_a_w || !b0 || !a0 || !og_selfrm)
    {
        fprintf(stderr, "ERROR: sysid memory allocation failed\n");
        free(A);
        free(B);
        free(D);
        free(muA_tr);
        free(muB_tr);
        free(muD_tr);
        free(muA_fl);
        free(muB_fl);
        free(muD_fl);
        free(Ac_tr);
        free(Bc_tr);
        free(Dc_tr);
        free(Ac_te);
        free(Bc_te);
        free(Dc_te);
        free(Ac_fl);
        free(Bc_fl);
        free(Dc_fl);
        free(Aw_tr);
        free(Bw_tr);
        free(Dw_tr);
        free(Aw_te);
        free(Bw_te);
        free(Dw_te);
        free(Aw_fl);
        free(Bw_fl);
        free(Dw_fl);
        free(Saa);
        free(Sdd);
        free(Sbb);
        free(Sad);
        free(Sba);
        free(Sbd);
        free(Mab_tr);
        free(Mdb_tr);
        free(Mab_raw);
        free(Mdb_raw);
        free(Mab_fl);
        free(Mdb_fl);
        free(Mba_tr);
        free(Mda_tr);
        free(Mba_fl);
        free(Mda_fl);
        free(Mda_virt);
        free(eta_d_raw);
        free(eta_d_white);
        free(mdb_diag_raw);
        free(mdb_diag);
        free(r2_b_w);
        free(r2_ol_raw);
        free(r2_a_raw);
        free(r2_a_w);
        free(b0);
        free(a0);
        free(og_selfrm);
        return RETURN_FAILURE;
    }

    compute_col_means(A, n_tr, dA, muA_tr);
    compute_col_means(B, n_tr, dB, muB_tr);
    compute_col_means(D, n_tr, dD, muD_tr);
    compute_col_means(A, N, dA, muA_fl);
    compute_col_means(B, N, dB, muB_fl);
    compute_col_means(D, N, dD, muD_fl);

    demean_slice(A, n_tr, dA, muA_tr, Ac_tr);
    demean_slice(B, n_tr, dB, muB_tr, Bc_tr);
    demean_slice(D, n_tr, dD, muD_tr, Dc_tr);
    demean_slice(A + n_tr * dA, n_te, dA, muA_tr, Ac_te);
    demean_slice(B + n_tr * dB, n_te, dB, muB_tr, Bc_te);
    demean_slice(D + n_tr * dD, n_te, dD, muD_tr, Dc_te);
    demean_slice(A, N, dA, muA_fl, Ac_fl);
    demean_slice(B, N, dB, muB_fl, Bc_fl);
    demean_slice(D, N, dD, muD_fl, Dc_fl);

    /* 1. Unwhitened baseline for direct diagnostic comparison against W(z) whitening. */
    accum_covs(Ac_tr, Dc_tr, Bc_tr, n_tr, dA, dD, dB, Saa, Sdd, NULL, Sad, Sba, Sbd);
    double reg_a_raw = (p_reg_a > 0.0) ? p_reg_a : 1e-2;
    double reg_d_raw = (p_reg_d > 0.0) ? p_reg_d : 1e-3;
    (void) solve_block_schur(Saa, Sdd, Sad, Sba, Sbd, dA, dD, dB, reg_a_raw, reg_d_raw, Mab_raw,
                             Mdb_raw, eta_d_raw);
    for (size_t i = 0; i < d_diag; i++)
    {
        mdb_diag_raw[i] = Mdb_raw[i * dD + i];
    }
    stats_median_iqr(eta_d_raw, dD, &o_eta_d_raw, NULL);
    stats_median_iqr(mdb_diag_raw, d_diag, &o_mdb_diag_raw, NULL);

    /* 2. Fit scalar causal AR(p) whitening filter W(z) on A_tr and apply identically to A,B,D. */
    double taps[SYSID_MAX_AR_ORDER + 1];
    fit_pooled_ar_taps(Ac_tr, n_tr, dA, w_ord, p_whiten_ridge, p_whiten_normch, taps);
    apply_fir_valid(Ac_tr, n_tr, dA, taps, w_ord, Aw_tr);
    apply_fir_valid(Bc_tr, n_tr, dB, taps, w_ord, Bw_tr);
    apply_fir_valid(Dc_tr, n_tr, dD, taps, w_ord, Dw_tr);
    apply_fir_valid(Ac_te, n_te, dA, taps, w_ord, Aw_te);
    apply_fir_valid(Bc_te, n_te, dB, taps, w_ord, Bw_te);
    apply_fir_valid(Dc_te, n_te, dD, taps, w_ord, Dw_te);
    apply_fir_valid(Ac_fl, N, dA, taps, w_ord, Aw_fl);
    apply_fir_valid(Bc_fl, N, dB, taps, w_ord, Bw_fl);
    apply_fir_valid(Dc_fl, N, dD, taps, w_ord, Dw_fl);

    /* 3. Whitened training covariances & contiguous held-out CV for (reg_a, reg_d). */
    accum_covs(Aw_tr, Dw_tr, Bw_tr, n_tr_w, dA, dD, dB, Saa, Sdd, Sbb, Sad, Sba, Sbd);

    double best_ra = (p_reg_a > 0.0) ? p_reg_a : 1e-2;
    double best_rd = (p_reg_d > 0.0) ? p_reg_d : 1e-3;
    if (p_reg_a < 0.0 || p_reg_d < 0.0)
    {
        static const double ra_grid[5] = { 1e-3, 3e-3, 1e-2, 3e-2, 1e-1 };
        static const double rd_grid[4] = { 1e-4, 1e-3, 3e-3, 1e-2 };
        int                 na_g       = (p_reg_a < 0.0) ? 5 : 1;
        int                 nd_g       = (p_reg_d < 0.0) ? 4 : 1;
        double              best_score = -1e9;

        for (int ia = 0; ia < na_g; ia++)
        {
            double ra = (p_reg_a < 0.0) ? ra_grid[ia] : p_reg_a;
            for (int id = 0; id < nd_g; id++)
            {
                double rd = (p_reg_d < 0.0) ? rd_grid[id] : p_reg_d;
                if (solve_block_schur(Saa, Sdd, Sad, Sba, Sbd, dA, dD, dB, ra, rd, Mab_tr, Mdb_tr,
                                      NULL) == 0)
                {
                    double sc = eval_r2_two(Aw_te, Dw_te, Bw_te, Mab_tr, Mdb_tr, n_te_w, dA, dD, dB,
                                            NULL);
                    if (sc > best_score)
                    {
                        best_score = sc;
                        best_ra    = ra;
                        best_rd    = rd;
                    }
                }
            }
        }
    }
    o_reg_a_opt = best_ra;
    o_reg_d_opt = best_rd;

    /* 4. Evaluate held-out forward metrics using strictly training-fit matrices:
     *    - Step 4a: Whitened Schur solve gives unbiased Mdb_tr (breaking closed-loop collinearity).
     *    - Step 4b: Subtract Mdb_tr * D from unwhitened B to form DM-free B_OL = B - Mdb * D,
     *               then refine Mab on full-SNR B_OL -> A (avoiding high-freq WFS-A noise
     *               attenuation) while evaluating held-out R2 on both B_w and B_OL. */
    (void) solve_block_schur(Saa, Sdd, Sad, Sba, Sbd, dA, dD, dB, best_ra, best_rd, Mab_tr, Mdb_tr,
                             NULL);

    double *Bol_tr  = (double *) malloc(n_tr * dB * sizeof(double));
    double *Saa_raw = (double *) malloc(dA * dA * sizeof(double));
    double *Sdd_raw = (double *) malloc(dD * dD * sizeof(double));
    double *Sbb_raw = (double *) malloc(dB * dB * sizeof(double));
    double *Sad_raw = (double *) malloc(dA * dD * sizeof(double));
    double *Sba_raw = (double *) malloc(dB * dA * sizeof(double));
    double *Sbd_raw = (double *) malloc(dB * dD * sizeof(double));
    double *Sola    = (double *) malloc(dB * dA * sizeof(double));
    double *Saa_reg = (double *) malloc(dA * dA * sizeof(double));
    double *rhs_ab  = (double *) malloc(dA * dB * sizeof(double));
    double *Mab_pol = (double *) calloc(dB * dA, sizeof(double));

    double best_ra_pol = best_ra;
    if (Bol_tr && Saa_raw && Sdd_raw && Sbb_raw && Sad_raw && Sba_raw && Sbd_raw && Sola &&
        Saa_reg && rhs_ab && Mab_pol)
    {
        accum_covs(Ac_tr, Dc_tr, Bc_tr, n_tr, dA, dD, dB, Saa_raw, Sdd_raw, Sbb_raw, Sad_raw,
                   Sba_raw, Sbd_raw);
        memcpy(Bol_tr, Bc_tr, n_tr * dB * sizeof(double));
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, (int) n_tr, (int) dB, (int) dD, -1.0,
                    Dc_tr, (int) dD, Mdb_tr, (int) dD, 1.0, Bol_tr, (int) dB);
        double inv_ntr = (n_tr > 1) ? (1.0 / (double) (n_tr - 1)) : 1.0;
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, (int) dB, (int) dA, (int) n_tr,
                    inv_ntr, Bol_tr, (int) dB, Ac_tr, (int) dA, 0.0, Sola, (int) dA);

        static const double rap_grid[5] = { 1e-3, 3e-3, 1e-2, 3e-2, 1e-1 };
        int                 nap_g       = (p_reg_a < 0.0) ? 5 : 1;
        double              best_pol_r2 = -1e9;
        for (int ia = 0; ia < nap_g; ia++)
        {
            double rap = (p_reg_a < 0.0) ? rap_grid[ia] : p_reg_a;
            add_diag_ridge(Saa_raw, dA, rap, Saa_reg);
            transpose_f64(Sola, dB, dA, rhs_ab);
            if (solve_spd_robust(Saa_reg, dA, dB, rhs_ab) == 0)
            {
                transpose_f64(rhs_ab, dA, dB, Mab_pol);
                double r2p = eval_r2_ol(Ac_te, Bc_te, Dc_te, Mab_pol, Mdb_tr, n_te, dA, dB, dD,
                                        r2_ol_raw);
                if (r2p > best_pol_r2)
                {
                    best_pol_r2 = r2p;
                    best_ra_pol = rap;
                }
            }
        }
        add_diag_ridge(Saa_raw, dA, best_ra_pol, Saa_reg);
        transpose_f64(Sola, dB, dA, rhs_ab);
        if (solve_spd_robust(Saa_reg, dA, dB, rhs_ab) == 0)
        {
            transpose_f64(rhs_ab, dA, dB, Mab_pol);
        }
    }

    /* Evaluate held-out whitened B R2 using whitened fit, and held-out OL R2 using refined Mab. */
    o_r2_b_w_med = eval_r2_two(Aw_te, Dw_te, Bw_te, Mab_tr, Mdb_tr, n_te_w, dA, dD, dB, r2_b_w);
    o_r2_ol_raw_med = eval_r2_ol(Ac_te, Bc_te, Dc_te, (Mab_pol != NULL) ? Mab_pol : Mab_tr, Mdb_tr,
                                 n_te, dA, dB, dD, r2_ol_raw);

    /* 5. Virtual upstream fit: A = Mba * B + Mda * D (B and D are naturally decoupled). */
    size_t  d_ov_a  = (dA < dD) ? dA : dD;
    double  best_rb = (p_reg_b > 0.0) ? p_reg_b : best_ra_pol;
    double *Sab     = (double *) malloc(dA * dB * sizeof(double));
    double *Sbd_tr  = (double *) malloc(dB * dD * sizeof(double));
    if (Sab != NULL && Sbd_tr != NULL && Sbb_raw != NULL && Sdd_raw != NULL && Sbd_raw != NULL &&
        Sba_raw != NULL && Sad_raw != NULL)
    {
        transpose_f64(Sba_raw, dB, dA, Sab);
        memcpy(Sbd_tr, Sbd_raw, dB * dD * sizeof(double));
        (void) solve_block_schur(Sbb_raw, Sdd_raw, Sbd_tr, Sab, Sad_raw, dB, dD, dA, best_rb,
                                 best_rd, Mba_tr, Mda_tr, NULL);
        (void) eval_r2_two(Bw_te, Dw_te, Aw_te, Mba_tr, Mda_tr, n_te_w, dB, dD, dA, r2_a_w);
        (void) eval_r2_two(Bc_te, Dc_te, Ac_te, Mba_tr, Mda_tr, n_te, dB, dD, dA, r2_a_raw);
        stats_median_iqr(r2_a_w, d_ov_a, &o_r2_a_w_med, NULL);
        stats_median_iqr(r2_a_raw, d_ov_a, &o_r2_a_raw_med, NULL);
    }

    /* 6. Final full-window high-SNR fit for both forward and virtual upstream models. */
    accum_covs(Aw_fl, Dw_fl, Bw_fl, n_fl_w, dA, dD, dB, Saa, Sdd, Sbb, Sad, Sba, Sbd);
    (void) solve_block_schur(Saa, Sdd, Sad, Sba, Sbd, dA, dD, dB, best_ra, best_rd, Mab_fl, Mdb_fl,
                             eta_d_white);
    if (Saa_raw && Sdd_raw && Sbb_raw && Sad_raw && Sba_raw && Sbd_raw && Sola && Saa_reg &&
        rhs_ab && Sab && Sbd_tr)
    {
        accum_covs(Ac_fl, Dc_fl, Bc_fl, N, dA, dD, dB, Saa_raw, Sdd_raw, Sbb_raw, Sad_raw, Sba_raw,
                   Sbd_raw);
        double *Bol_fl = (double *) malloc(N * dB * sizeof(double));
        if (Bol_fl != NULL)
        {
            memcpy(Bol_fl, Bc_fl, N * dB * sizeof(double));
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, (int) N, (int) dB, (int) dD, -1.0,
                        Dc_fl, (int) dD, Mdb_fl, (int) dD, 1.0, Bol_fl, (int) dB);
            double inv_n = (N > 1) ? (1.0 / (double) (N - 1)) : 1.0;
            cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, (int) dB, (int) dA, (int) N, inv_n,
                        Bol_fl, (int) dB, Ac_fl, (int) dA, 0.0, Sola, (int) dA);
            add_diag_ridge(Saa_raw, dA, best_ra_pol, Saa_reg);
            transpose_f64(Sola, dB, dA, rhs_ab);
            if (solve_spd_robust(Saa_reg, dA, dB, rhs_ab) == 0)
            {
                transpose_f64(rhs_ab, dA, dB, Mab_fl);
            }
            free(Bol_fl);
        }
        transpose_f64(Sba_raw, dB, dA, Sab);
        memcpy(Sbd_tr, Sbd_raw, dB * dD * sizeof(double));
        (void) solve_block_schur(Sbb_raw, Sdd_raw, Sbd_tr, Sab, Sad_raw, dB, dD, dA, best_rb,
                                 best_rd, Mba_fl, Mda_fl, NULL);
    }
    free(Bol_tr);
    free(Saa_raw);
    free(Sdd_raw);
    free(Sbb_raw);
    free(Sad_raw);
    free(Sba_raw);
    free(Sbd_raw);
    free(Sola);
    free(Saa_reg);
    free(rhs_ab);
    free(Mab_pol);
    free(Sab);
    free(Sbd_tr);

    for (size_t i = 0; i < dA * dD; i++)
    {
        Mda_virt[i] = -Mda_fl[i];
    }

    /* Static differential NCPA intercepts b0 and a0 from full-window means. */
    memcpy(b0, muB_fl, dB * sizeof(double));
    cblas_dgemv(CblasRowMajor, CblasNoTrans, (int) dB, (int) dA, -1.0, Mab_fl, (int) dA, muA_fl, 1,
                1.0, b0, 1);
    cblas_dgemv(CblasRowMajor, CblasNoTrans, (int) dB, (int) dD, -1.0, Mdb_fl, (int) dD, muD_fl, 1,
                1.0, b0, 1);
    double b0_s2 = 0.0;
    for (size_t i = 0; i < dB; i++)
    {
        b0_s2 += b0[i] * b0[i];
    }
    o_b0_rms = sqrt(b0_s2 / (double) dB);

    memcpy(a0, muA_fl, dA * sizeof(double));
    cblas_dgemv(CblasRowMajor, CblasNoTrans, (int) dA, (int) dB, -1.0, Mba_fl, (int) dB, muB_fl, 1,
                1.0, a0, 1);
    cblas_dgemv(CblasRowMajor, CblasNoTrans, (int) dA, (int) dD, -1.0, Mda_fl, (int) dD, muD_fl, 1,
                1.0, a0, 1);

    /* Diagnostics on Mdb and algebraic cross-consistency Mab * Mda_virt vs Mdb. */
    double off_s2 = 0.0;
    size_t off_n  = 0;
    for (size_t r = 0; r < dB; r++)
    {
        for (size_t c = 0; c < dD; c++)
        {
            double v = Mdb_fl[r * dD + c];
            if (r == c && r < d_diag)
            {
                mdb_diag[r] = v;
            }
            else
            {
                off_s2 += v * v;
                off_n++;
            }
        }
    }
    o_mdb_offdiag = (off_n > 0) ? sqrt(off_s2 / (double) off_n) : 0.0;
    stats_median_iqr(eta_d_white, dD, &o_eta_d_white, NULL);
    stats_median_iqr(mdb_diag, d_diag, &o_mdb_diag_med, &o_mdb_diag_iqr);

    double *Mdb_cross = (double *) malloc(dB * dD * sizeof(double));
    if (Mdb_cross != NULL)
    {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int) dB, (int) dD, (int) dA, 1.0,
                    Mab_fl, (int) dA, Mda_virt, (int) dD, 0.0, Mdb_cross, (int) dD);
        double num = 0.0;
        double den = 0.0;
        for (size_t i = 0; i < dB * dD; i++)
        {
            double diff = Mdb_cross[i] - Mdb_fl[i];
            num += diff * diff;
            den += Mdb_fl[i] * Mdb_fl[i];
        }
        o_cross_rel_err = (den > 1e-24) ? sqrt(num / den) : 0.0;
        free(Mdb_cross);
    }

    int has_selfrm = load_selfrm_og(p_selfrm_cal, p_selfrm_sky, d_diag, og_selfrm);
    o_selfrm_og_med = 0.0;
    o_selfrm_err    = 0.0;
    if (has_selfrm)
    {
        stats_median_iqr(og_selfrm, d_diag, &o_selfrm_og_med, NULL);
        if (o_selfrm_og_med > 1e-6)
        {
            o_selfrm_err = fabs(o_mdb_diag_med - o_selfrm_og_med) / o_selfrm_og_med;
        }
    }

    o_nframes = (int64_t) N;
    o_n_train = (int64_t) n_tr;
    o_n_test  = (int64_t) n_te;
    o_dim_a   = (int64_t) dA;
    o_dim_b   = (int64_t) dB;
    o_dim_d   = (int64_t) dD;

    /* Write FITS matrices, diagnostic vectors, key=value report, and JSON summary. */
    mlt_fits_kw kws[6] = {
        { "NFRAMES", 'L', 0.0, (long) N, "", "Aligned frames" },
        { "WORDER", 'L', 0.0, (long) w_ord, "", "Causal AR whitening order" },
        { "REGA", 'D', best_ra, 0, "", "Selected A relative ridge" },
        { "REGD", 'D', best_rd, 0, "", "Selected D relative ridge" },
        { "LAGABMS", 'D', p_lag_ab_ms, 0, "", "A->B lag [ms]" },
        { "LAGDBMS", 'D', p_lag_db_ms, 0, "", "D->B lag [ms]" },
    };

    char fpath[MLT_PATH_MAX];
    long ax_ab[2]   = { (long) dA, (long) dB };
    long ax_db[2]   = { (long) dD, (long) dB };
    long ax_ba[2]   = { (long) dB, (long) dA };
    long ax_da[2]   = { (long) dD, (long) dA };
    long ax_va[2]   = { (long) dA, 1L };
    long ax_vb[2]   = { (long) dB, 1L };
    long ax_vd[2]   = { (long) dD, 1L };
    long ax_vdiag[2] = { (long) d_diag, 1L };
    long ax_taps[2] = { (long) (w_ord + 1), 1L };

    snprintf(fpath, sizeof(fpath), "%s/Mab.fits", p_outdir);
    write_f64_as_fits(fpath, Mab_fl, 2, ax_ab, kws, 6);
    snprintf(fpath, sizeof(fpath), "%s/Mdb.fits", p_outdir);
    write_f64_as_fits(fpath, Mdb_fl, 2, ax_db, kws, 6);
    snprintf(fpath, sizeof(fpath), "%s/Mba.fits", p_outdir);
    write_f64_as_fits(fpath, Mba_fl, 2, ax_ba, kws, 6);
    snprintf(fpath, sizeof(fpath), "%s/Mda_virt.fits", p_outdir);
    write_f64_as_fits(fpath, Mda_virt, 2, ax_da, kws, 6);
    snprintf(fpath, sizeof(fpath), "%s/b0.fits", p_outdir);
    write_f64_as_fits(fpath, b0, 2, ax_vb, kws, 6);
    snprintf(fpath, sizeof(fpath), "%s/a0.fits", p_outdir);
    write_f64_as_fits(fpath, a0, 2, ax_va, kws, 6);
    snprintf(fpath, sizeof(fpath), "%s/ar_taps.fits", p_outdir);
    write_f64_as_fits(fpath, taps, 2, ax_taps, kws, 6);
    snprintf(fpath, sizeof(fpath), "%s/eta_d.fits", p_outdir);
    write_f64_as_fits(fpath, eta_d_white, 2, ax_vd, kws, 6);
    snprintf(fpath, sizeof(fpath), "%s/eta_d_raw.fits", p_outdir);
    write_f64_as_fits(fpath, eta_d_raw, 2, ax_vd, kws, 6);
    snprintf(fpath, sizeof(fpath), "%s/mdb_diag.fits", p_outdir);
    write_f64_as_fits(fpath, mdb_diag, 2, ax_vdiag, kws, 6);
    snprintf(fpath, sizeof(fpath), "%s/mdb_diag_raw.fits", p_outdir);
    write_f64_as_fits(fpath, mdb_diag_raw, 2, ax_vdiag, kws, 6);
    snprintf(fpath, sizeof(fpath), "%s/r2_b_w.fits", p_outdir);
    write_f64_as_fits(fpath, r2_b_w, 2, ax_vb, kws, 6);
    snprintf(fpath, sizeof(fpath), "%s/r2_ol_raw.fits", p_outdir);
    write_f64_as_fits(fpath, r2_ol_raw, 2, ax_vb, kws, 6);
    snprintf(fpath, sizeof(fpath), "%s/r2_a_raw.fits", p_outdir);
    write_f64_as_fits(fpath, r2_a_raw, 2, ax_va, kws, 6);
    snprintf(fpath, sizeof(fpath), "%s/r2_a_w.fits", p_outdir);
    write_f64_as_fits(fpath, r2_a_w, 2, ax_va, kws, 6);
    if (has_selfrm)
    {
        snprintf(fpath, sizeof(fpath), "%s/og_selfrm.fits", p_outdir);
        write_f64_as_fits(fpath, og_selfrm, 2, ax_vdiag, kws, 6);
    }

    /* Write JSON summary inside outdir. */
    snprintf(fpath, sizeof(fpath), "%s/sysid_summary.json", p_outdir);
    FILE *jfp = fopen(fpath, "w");
    if (jfp != NULL)
    {
        fprintf(jfp, "{\n");
        fprintf(jfp, "  \"enabled\": true,\n");
        fprintf(jfp, "  \"nframes\": %zu,\n", N);
        fprintf(jfp, "  \"n_train\": %zu,\n", n_tr);
        fprintf(jfp, "  \"n_test\": %zu,\n", n_te);
        fprintf(jfp, "  \"dim_a\": %zu,\n", dA);
        fprintf(jfp, "  \"dim_b\": %zu,\n", dB);
        fprintf(jfp, "  \"dim_d\": %zu,\n", dD);
        fprintf(jfp, "  \"whiten_order\": %d,\n", w_ord);
        fprintf(jfp, "  \"whiten_ridge\": %.6g,\n", p_whiten_ridge);
        fprintf(jfp, "  \"ar_taps\": [");
        for (int m = 0; m <= w_ord; m++)
        {
            fprintf(jfp, "%s%.6f", (m > 0) ? ", " : "", taps[m]);
        }
        fprintf(jfp, "],\n");
        fprintf(jfp, "  \"fs_hz\": %.4f,\n", p_fs_hz);
        fprintf(jfp, "  \"lag_ab_ms\": %.4f,\n", p_lag_ab_ms);
        fprintf(jfp, "  \"lag_db_ms\": %.4f,\n", p_lag_db_ms);
        fprintf(jfp, "  \"lag_ab_fr\": %.4f,\n", p_lag_ab_fr);
        fprintf(jfp, "  \"lag_db_fr\": %.4f,\n", p_lag_db_fr);
        fprintf(jfp, "  \"reg_a_opt\": %.6g,\n", o_reg_a_opt);
        fprintf(jfp, "  \"reg_d_opt\": %.6g,\n", o_reg_d_opt);
        fprintf(jfp, "  \"eta_d_raw_med\": %.6f,\n", o_eta_d_raw);
        fprintf(jfp, "  \"eta_d_white_med\": %.6f,\n", o_eta_d_white);
        fprintf(jfp, "  \"mdb_diag_raw_med\": %.6f,\n", o_mdb_diag_raw);
        fprintf(jfp, "  \"mdb_diag_med\": %.6f,\n", o_mdb_diag_med);
        fprintf(jfp, "  \"mdb_diag_iqr\": %.6f,\n", o_mdb_diag_iqr);
        fprintf(jfp, "  \"mdb_offdiag_rms\": %.6f,\n", o_mdb_offdiag);
        fprintf(jfp, "  \"r2_b_w_med\": %.6f,\n", o_r2_b_w_med);
        fprintf(jfp, "  \"r2_ol_raw_med\": %.6f,\n", o_r2_ol_raw_med);
        fprintf(jfp, "  \"r2_a_raw_med\": %.6f,\n", o_r2_a_raw_med);
        fprintf(jfp, "  \"r2_a_w_med\": %.6f,\n", o_r2_a_w_med);
        fprintf(jfp, "  \"cross_rel_err\": %.6f,\n", o_cross_rel_err);
        fprintf(jfp, "  \"has_selfrm\": %s,\n", has_selfrm ? "true" : "false");
        fprintf(jfp, "  \"selfrm_og_med\": %.6f,\n", o_selfrm_og_med);
        fprintf(jfp, "  \"selfrm_rel_err\": %.6f,\n", o_selfrm_err);
        fprintf(jfp, "  \"b0_rms\": %.8g\n", o_b0_rms);
        fprintf(jfp, "}\n");
        fclose(jfp);
    }

    if (p_outreport[0] != '\0')
    {
        FILE *fp = fopen(p_outreport, "w");
        if (fp != NULL)
        {
            fprintf(fp, "nframes=%zu\n", N);
            fprintf(fp, "n_train=%zu\n", n_tr);
            fprintf(fp, "n_test=%zu\n", n_te);
            fprintf(fp, "dim_a=%zu\n", dA);
            fprintf(fp, "dim_b=%zu\n", dB);
            fprintf(fp, "dim_d=%zu\n", dD);
            fprintf(fp, "whiten_order=%d\n", w_ord);
            fprintf(fp, "whiten_ridge=%.6g\n", p_whiten_ridge);
            fprintf(fp, "fs_hz=%.4f\n", p_fs_hz);
            fprintf(fp, "lag_ab_ms=%.4f\n", p_lag_ab_ms);
            fprintf(fp, "lag_db_ms=%.4f\n", p_lag_db_ms);
            fprintf(fp, "reg_a_opt=%.6g\n", o_reg_a_opt);
            fprintf(fp, "reg_d_opt=%.6g\n", o_reg_d_opt);
            fprintf(fp, "eta_d_raw=%.6f\n", o_eta_d_raw);
            fprintf(fp, "eta_d_white=%.6f\n", o_eta_d_white);
            fprintf(fp, "mdb_diag_raw=%.6f\n", o_mdb_diag_raw);
            fprintf(fp, "mdb_diag_med=%.6f\n", o_mdb_diag_med);
            fprintf(fp, "mdb_diag_iqr=%.6f\n", o_mdb_diag_iqr);
            fprintf(fp, "mdb_offdiag=%.6f\n", o_mdb_offdiag);
            fprintf(fp, "r2_b_w_med=%.6f\n", o_r2_b_w_med);
            fprintf(fp, "r2_ol_raw_med=%.6f\n", o_r2_ol_raw_med);
            fprintf(fp, "r2_a_raw_med=%.6f\n", o_r2_a_raw_med);
            fprintf(fp, "r2_a_w_med=%.6f\n", o_r2_a_w_med);
            fprintf(fp, "cross_rel_err=%.6f\n", o_cross_rel_err);
            fprintf(fp, "has_selfrm=%d\n", has_selfrm);
            fprintf(fp, "selfrm_og_med=%.6f\n", o_selfrm_og_med);
            fprintf(fp, "selfrm_err=%.6f\n", o_selfrm_err);
            fprintf(fp, "b0_rms=%.8g\n", o_b0_rms);
            fclose(fp);
        }
    }

    if (!dcquiet)
    {
        printf("mlt.sysid: N=%zu (dA=%zu dB=%zu dD=%zu) AR(%d) eta_D=%.3f->%.3f "
               "diag(Mdb)=%.3f->%.3f (off=%.4f) R2(B_w=%.3f OL=%.3f A=%.3f)\n",
               N, dA, dB, dD, w_ord, o_eta_d_raw, o_eta_d_white, o_mdb_diag_raw, o_mdb_diag_med,
               o_mdb_offdiag, o_r2_b_w_med, o_r2_ol_raw_med, o_r2_a_raw_med);
    }

    free(A);
    free(B);
    free(D);
    free(muA_tr);
    free(muB_tr);
    free(muD_tr);
    free(muA_fl);
    free(muB_fl);
    free(muD_fl);
    free(Ac_tr);
    free(Bc_tr);
    free(Dc_tr);
    free(Ac_te);
    free(Bc_te);
    free(Dc_te);
    free(Ac_fl);
    free(Bc_fl);
    free(Dc_fl);
    free(Aw_tr);
    free(Bw_tr);
    free(Dw_tr);
    free(Aw_te);
    free(Bw_te);
    free(Dw_te);
    free(Aw_fl);
    free(Bw_fl);
    free(Dw_fl);
    free(Saa);
    free(Sdd);
    free(Sbb);
    free(Sad);
    free(Sba);
    free(Sbd);
    free(Mab_tr);
    free(Mdb_tr);
    free(Mab_raw);
    free(Mdb_raw);
    free(Mab_fl);
    free(Mdb_fl);
    free(Mba_tr);
    free(Mda_tr);
    free(Mba_fl);
    free(Mda_fl);
    free(Mda_virt);
    free(eta_d_raw);
    free(eta_d_white);
    free(mdb_diag_raw);
    free(mdb_diag);
    free(r2_b_w);
    free(r2_ol_raw);
    free(r2_a_raw);
    free(r2_a_w);
    free(b0);
    free(a0);
    free(og_selfrm);
    return RETURN_SUCCESS;
}

static MILK_HOT errno_t sysid_compute(const MILK_RUNNABLE *self)
{
    errno_t ret = RETURN_SUCCESS;
    DEBUG_TRACE_FSTART();
    MILK_PROCINFO_INIT(self)
    MILK_PROCINFO_LOOPSTART(self)
    ret = sysid_run();
    MILK_PROCINFO_END(self)
    if (dcfpsptr != NULL)
    {
        fpsresync_modvar_to_fps(dcfpsptr, dcfpsbindings, dcfpsnbindings);
    }
    DEBUG_TRACE_FEXIT();
    return ret;
}

const MILK_RUNNABLE runnable_mlt_sysid = {
    .kind = MILK_RUNNABLE_FPS,
    .app  = { .fps_name         = "mltsysid",
              .cmdkey           = "sysid",
              .description      = "calibrate Mab, Mdb & virtual Mda_virt from 2-WFS+1-DM telemetry",
              .description_long = "Solve B = Mab A + Mdb D + b0 and virtual upstream "
                                  "A = Mba B - Mda_virt D + a0 via common causal AR(p) whitening "
                                  "and block-regularized Schur-complement normal equations with "
                                  "contiguous temporal cross-validation." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(mlt_sysid),
    .compute = sysid_compute,
};

EXPORT_FPSEXEC(milk, "mlt-sysid", runnable_mlt_sysid);
