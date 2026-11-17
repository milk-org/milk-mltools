// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    mlt_calc_ext.c
 * @brief   High-speed time-series and mask operators for `milk.calc` (`libmltools_calc.so`)
 *
 * Exports:
 *   - mlt_xcov(A, B, maxlag=50, norm=1)
 *   - Xw, Yw = mlt_arwhiten(X, Y, order=10, ridge=1e-5)
 *   - mlt_arfilt1(X, order=10, ridge=1e-5)
 *   - pos, val, fwhm = mlt_peak1d(curve, x0=0.0, dx=1.0)
 *   - mlt_mask(cube, mask)
 *   - mlt_unmask(mat, mask)
 */

#include <math.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>

#include "milk_calc_ext.h"

/* =========================================================================
 * 1. mlt_xcov(A, B, maxlag=50, norm=1)
 * ========================================================================= */

#define MLT_XCOV_PARAMS(X)                                                                   \
    X(".maxlag", maxlag, FPTYPE_INT64, 50, FPFLAG_DEFAULT_INPUT, "Maximum sample lag (>=1)") \
    X(".norm", norm, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                \
      "1=normalized Pearson RMS, 0=raw covariance RMS")

#define MLT_XCOV_PORTS(X)                                                              \
    X(IN, A, MILK_PORT_IMAGE, 0, "Modal matrix A (ka, N) or 1D vector (N)")            \
    X(IN, B, MILK_PORT_IMAGE, 0, "Modal matrix B (kb, N) or 1D vector (N)")            \
    X(OUT, xc, MILK_PORT_IMAGE, 0, "1D RMS cross-correlation curve of len 2*maxlag+1")

MILK_CALC_OP_DECLARE(mlt_xcov, MLT_XCOV_PARAMS, MLT_XCOV_PORTS);

static void get_mat_dims(const IMAGE *im, uint32_t *k_out, size_t *n_out)
{
    if (im->md[0].naxis == 1)
    {
        *k_out = 1;
        *n_out = im->md[0].size[0];
    }
    else
    {
        *k_out = im->md[0].size[0];
        *n_out = im->md[0].size[1];
    }
}

static int mlt_xcov_init(milk_calc_op_ctx_t *ctx, const void *P, void **state)
{
    (void) state;
    const struct mlt_xcov_par *par = (const struct mlt_xcov_par *) P;
    const IMAGE               *imA = MILK_IN(ctx, A);
    const IMAGE               *imB = MILK_IN(ctx, B);
    if (imA == NULL || imB == NULL)
    {
        return -1;
    }

    uint32_t ka = 0;
    uint32_t kb = 0;
    size_t   na = 0;
    size_t   nb = 0;
    get_mat_dims(imA, &ka, &na);
    get_mat_dims(imB, &kb, &nb);

    if (ka == 0 || kb == 0 || na != nb || na < 3 || par->maxlag < 1)
    {
        return -1;
    }

    uint32_t maxlag = (uint32_t) par->maxlag;
    if ((size_t) maxlag >= na - 1)
    {
        maxlag = (uint32_t) (na - 2);
    }
    milk_op_set_out_shape(ctx, xc, 2 * maxlag + 1, 0);
    return 0;
}

static int mlt_xcov_exec(milk_calc_op_ctx_t *ctx, const void *P, void *state)
{
    (void) state;
    const struct mlt_xcov_par *par = (const struct mlt_xcov_par *) P;
    const IMAGE               *imA = MILK_IN(ctx, A);
    const IMAGE               *imB = MILK_IN(ctx, B);
    IMAGE                     *out = MILK_OUT(ctx, xc);
    if (imA == NULL || imB == NULL || out == NULL)
    {
        return -1;
    }

    uint32_t ka = 0;
    uint32_t kb = 0;
    size_t   na = 0;
    size_t   nb = 0;
    get_mat_dims(imA, &ka, &na);
    get_mat_dims(imB, &kb, &nb);
    if (na != nb || na < 3)
    {
        return -1;
    }

    int64_t maxlag = par->maxlag;
    if (maxlag >= (int64_t) na - 1)
    {
        maxlag = (int64_t) na - 2;
    }
    uint32_t out_n = out->md[0].size[0];
    if (out_n != (uint32_t) (2 * maxlag + 1))
    {
        maxlag = ((int64_t) out_n - 1) / 2;
    }

    size_t  n    = na;
    double *abuf = (double *) malloc((size_t) ka * n * sizeof(double));
    double *bbuf = (double *) malloc((size_t) kb * n * sizeof(double));
    double *apf  = (double *) malloc((size_t) ka * (n + 1) * sizeof(double));
    double *bpf  = (double *) malloc((size_t) kb * (n + 1) * sizeof(double));
    if (abuf == NULL || bbuf == NULL || apf == NULL || bpf == NULL)
    {
        free(abuf);
        free(bbuf);
        free(apf);
        free(bpf);
        return -1;
    }

    const float *srcA = imA->array.F;
    const float *srcB = imB->array.F;
    uint32_t     c;

#pragma omp parallel for schedule(static) if ((size_t) ka * n > 16384)
    for (c = 0; c < ka; c++)
    {
        double  sum = 0.0;
        double *row = abuf + (size_t) c * n;
        for (size_t t = 0; t < n; t++)
        {
            double v = (double) srcA[t * (size_t) ka + c];
            row[t]   = v;
            sum += v;
        }
        double  mean = sum / (double) n;
        double *pf   = apf + (size_t) c * (n + 1);
        double  acc  = 0.0;
        pf[0]        = 0.0;
        for (size_t t = 0; t < n; t++)
        {
            double z = row[t] - mean;
            row[t]   = z;
            acc += z * z;
            pf[t + 1] = acc;
        }
    }

#pragma omp parallel for schedule(static) if ((size_t) kb * n > 16384)
    for (c = 0; c < kb; c++)
    {
        double  sum = 0.0;
        double *row = bbuf + (size_t) c * n;
        for (size_t t = 0; t < n; t++)
        {
            double v = (double) srcB[t * (size_t) kb + c];
            row[t]   = v;
            sum += v;
        }
        double  mean = sum / (double) n;
        double *pf   = bpf + (size_t) c * (n + 1);
        double  acc  = 0.0;
        pf[0]        = 0.0;
        for (size_t t = 0; t < n; t++)
        {
            double z = row[t] - mean;
            row[t]   = z;
            acc += z * z;
            pf[t + 1] = acc;
        }
    }

    float  *dst     = out->array.F;
    int     do_norm = (par->norm != 0);
    double  denom   = do_norm ? (double) ((ka < kb) ? ka : kb) : ((double) ka * (double) kb);
    double  inv_den = 1.0 / denom;
    int64_t nlags   = 2 * maxlag + 1;
    int64_t li;

#pragma omp parallel for schedule(dynamic, 2) if (nlags > 4 && (size_t) ka * kb * n > 8192)
    for (li = 0; li < nlags; li++)
    {
        int64_t lag = li - maxlag;
        size_t  t0  = (lag < 0) ? (size_t) (-lag) : 0;
        size_t  t1  = (lag > 0) ? (n - (size_t) lag) : n;
        if (t1 <= t0 + 1)
        {
            dst[li] = 0.0f;
            continue;
        }

        size_t b0   = (size_t) ((int64_t) t0 + lag);
        size_t b1   = (size_t) ((int64_t) t1 + lag);
        size_t n_ov = t1 - t0;
        double len  = (double) n_ov;
        double s2   = 0.0;

        for (uint32_t ca = 0; ca < ka; ca++)
        {
            const double *arow = abuf + (size_t) ca * n + t0;
            const double *ap   = apf + (size_t) ca * (n + 1);
            double        ea   = ap[t1] - ap[t0];
            for (uint32_t cb = 0; cb < kb; cb++)
            {
                const double *brow = bbuf + (size_t) cb * n + b0;
                const double *bp   = bpf + (size_t) cb * (n + 1);
                double        eb   = bp[b1] - bp[b0];
                double        dot  = 0.0;
                for (size_t k = 0; k < n_ov; k++)
                {
                    dot += arow[k] * brow[k];
                }
                double val;
                if (do_norm)
                {
                    double den = sqrt(ea * eb);
                    val        = (den > 1e-30) ? (dot / den) : 0.0;
                }
                else
                {
                    val = dot / len;
                }
                s2 += val * val;
            }
        }
        dst[li] = (float) sqrt(s2 * inv_den);
    }

    free(abuf);
    free(bbuf);
    free(apf);
    free(bpf);
    return 0;
}

static void mlt_xcov_free(void *state)
{
    (void) state;
}

/* =========================================================================
 * 2. Shared causal AR(p) Levinson-Durbin whitening helpers
 * ========================================================================= */

static void fit_pooled_ar(const IMAGE *imX,
                          int          order,
                          double       ridge,
                          double      *a_out,
                          double     **z_out)
{
    uint32_t k = 0;
    size_t   n = 0;
    get_mat_dims(imX, &k, &n);

    int p = order;
    if (p < 0)
    {
        p = 0;
    }
    if ((size_t) p >= n / 2)
    {
        p = (int) (n / 4);
    }
    memset(a_out, 0, (size_t) (order + 1) * sizeof(double));

    double *zbuf = (double *) malloc((size_t) k * n * sizeof(double));
    if (zbuf == NULL)
    {
        *z_out = NULL;
        return;
    }

    const float *src = imX->array.F;
    uint32_t     c;
#pragma omp parallel for schedule(static) if ((size_t) k * n > 16384)
    for (c = 0; c < k; c++)
    {
        double  sum = 0.0;
        double *row = zbuf + (size_t) c * n;
        for (size_t t = 0; t < n; t++)
        {
            double v = (double) src[t * (size_t) k + c];
            row[t]   = v;
            sum += v;
        }
        double mean = sum / (double) n;
        for (size_t t = 0; t < n; t++)
        {
            row[t] -= mean;
        }
    }
    *z_out = zbuf;

    if (p <= 0)
    {
        return;
    }

    double *r = (double *) calloc((size_t) (p + 1), sizeof(double));
    if (r == NULL)
    {
        return;
    }

    for (uint32_t ci = 0; ci < k; ci++)
    {
        const double *row = zbuf + (size_t) ci * n;
        for (int m = 0; m <= p; m++)
        {
            double acc = 0.0;
            for (size_t t = (size_t) m; t < n; t++)
            {
                acc += row[t] * row[t - (size_t) m];
            }
            r[m] += acc;
        }
    }

    if (r[0] <= 1e-30)
    {
        free(r);
        return;
    }
    double r0 = r[0];
    for (int m = 0; m <= p; m++)
    {
        r[m] /= r0;
    }
    r[0] = 1.0 + ((ridge > 1e-12) ? ridge : 1e-12);

    /* Levinson-Durbin recursion for Toeplitz Yule-Walker system. */
    double *a_cur = (double *) calloc((size_t) (p + 1), sizeof(double));
    double *a_nxt = (double *) calloc((size_t) (p + 1), sizeof(double));
    if (a_cur != NULL && a_nxt != NULL)
    {
        double err = r[0];
        for (int m = 1; m <= p; m++)
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
        memcpy(a_out, a_cur, (size_t) (p + 1) * sizeof(double));
    }
    free(a_cur);
    free(a_nxt);
    free(r);
}

static void apply_causal_ar_from_z(const double *zbuf,
                                   uint32_t      k,
                                   size_t        n,
                                   const double *a,
                                   int           p,
                                   float        *dst)
{
    uint32_t c;
#pragma omp parallel for schedule(static) if ((size_t) k * n > 16384)
    for (c = 0; c < k; c++)
    {
        const double *row = zbuf + (size_t) c * n;
        for (size_t t = 0; t < n; t++)
        {
            if (t < (size_t) p)
            {
                dst[t * (size_t) k + c] = 0.0f;
            }
            else
            {
                double val = row[t];
                for (int m = 1; m <= p; m++)
                {
                    val -= a[m] * row[t - (size_t) m];
                }
                dst[t * (size_t) k + c] = (float) val;
            }
        }
    }
}

static void apply_causal_ar_image(const IMAGE  *im,
                                  const double *a,
                                  int           p,
                                  float        *dst)
{
    uint32_t     k   = 0;
    size_t       n   = 0;
    const float *src = im->array.F;
    get_mat_dims(im, &k, &n);

    if (p > (int) (n / 2))
    {
        p = (int) (n / 4);
    }

    uint32_t c;
#pragma omp parallel for schedule(static) if ((size_t) k * n > 16384)
    for (c = 0; c < k; c++)
    {
        double sum = 0.0;
        for (size_t t = 0; t < n; t++)
        {
            sum += (double) src[t * (size_t) k + c];
        }
        double mean = sum / (double) n;
        for (size_t t = 0; t < n; t++)
        {
            if (t < (size_t) p)
            {
                dst[t * (size_t) k + c] = 0.0f;
            }
            else
            {
                double val = (double) src[t * (size_t) k + c] - mean;
                for (int m = 1; m <= p; m++)
                {
                    double xm = (double) src[(t - (size_t) m) * (size_t) k + c] - mean;
                    val -= a[m] * xm;
                }
                dst[t * (size_t) k + c] = (float) val;
            }
        }
    }
}

/* =========================================================================
 * 3. Xw, Yw = mlt_arwhiten(X, Y, order=10, ridge=1e-5)
 * ========================================================================= */

#define MLT_ARW_PARAMS(X)                                                                      \
    X(".order", order, FPTYPE_INT64, 10, FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,               \
      "Causal AR whitening order")                                                             \
    X(".ridge", ridge, FPTYPE_FLOAT64, 1e-5, FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,           \
      "Diagonal ridge loading")

#define MLT_ARW_PORTS(M)                                                                  \
    M(IN, X, MILK_PORT_IMAGE, 0, "Reference matrix X (kx, N)")                            \
    M(IN, Y, MILK_PORT_IMAGE, 0, "Partner matrix Y (ky, Ny)")                             \
    M(OUT, Xw, MILK_PORT_IMAGE, 0, "Whitened X")                                          \
    M(OUT, Yw, MILK_PORT_IMAGE, 0, "Whitened Y filtered with identical A(z)")

MILK_CALC_OP_DECLARE(mlt_arwhiten, MLT_ARW_PARAMS, MLT_ARW_PORTS);

static int mlt_arwhiten_init(milk_calc_op_ctx_t *ctx, const void *P, void **state)
{
    (void) P;
    (void) state;
    const IMAGE *imX = MILK_IN(ctx, X);
    const IMAGE *imY = MILK_IN(ctx, Y);
    if (imX == NULL || imY == NULL)
    {
        return -1;
    }

    if (imX->md[0].naxis == 1)
    {
        milk_op_set_out_shape(ctx, Xw, imX->md[0].size[0], 0);
    }
    else
    {
        milk_op_set_out_shape(ctx, Xw, imX->md[0].size[0], imX->md[0].size[1]);
    }

    if (imY->md[0].naxis == 1)
    {
        milk_op_set_out_shape(ctx, Yw, imY->md[0].size[0], 0);
    }
    else
    {
        milk_op_set_out_shape(ctx, Yw, imY->md[0].size[0], imY->md[0].size[1]);
    }
    return 0;
}

static int mlt_arwhiten_exec(milk_calc_op_ctx_t *ctx, const void *P, void *state)
{
    (void) state;
    const struct mlt_arwhiten_par *par = (const struct mlt_arwhiten_par *) P;
    const IMAGE                   *imX = MILK_IN(ctx, X);
    const IMAGE                   *imY = MILK_IN(ctx, Y);
    IMAGE                         *owX = MILK_OUT(ctx, Xw);
    IMAGE                         *owY = MILK_OUT(ctx, Yw);
    if (imX == NULL || imY == NULL || owX == NULL || owY == NULL)
    {
        return -1;
    }

    int order = (int) par->order;
    if (order < 0)
    {
        order = 0;
    }
    if (order > 128)
    {
        order = 128;
    }

    uint32_t kx = 0;
    size_t   nx = 0;
    get_mat_dims(imX, &kx, &nx);
    int p = order;
    if ((size_t) p >= nx / 2)
    {
        p = (int) (nx / 4);
    }

    double  a[129];
    double *zX = NULL;
    fit_pooled_ar(imX, order, par->ridge, a, &zX);
    if (zX == NULL)
    {
        return -1;
    }
    apply_causal_ar_from_z(zX, kx, nx, a, p, owX->array.F);
    free(zX);

    apply_causal_ar_image(imY, a, p, owY->array.F);
    return 0;
}

static void mlt_arwhiten_free(void *state)
{
    (void) state;
}

/* =========================================================================
 * 4. mlt_arfilt1(X, order=10, ridge=1e-5) — Tier 1c single-input filter
 * ========================================================================= */

#define MLT_AR1_PARAMS(X)                                                                      \
    X(".order", order, FPTYPE_INT64, 10, FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,               \
      "Causal AR whitening order")                                                             \
    X(".ridge", ridge, FPTYPE_FLOAT64, 1e-5, FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,           \
      "Diagonal ridge loading")

MILK_CALC_PARAMS(mlt_arfilt1, MLT_AR1_PARAMS);

static int mlt_arfilt1_img(IMAGE *out, const IMAGE *in, const struct mlt_arfilt1_par *P)
{
    if (out == NULL || in == NULL)
    {
        return -1;
    }
    int order = (int) P->order;
    if (order < 0)
    {
        order = 0;
    }
    if (order > 128)
    {
        order = 128;
    }
    uint32_t kx = 0;
    size_t   nx = 0;
    get_mat_dims(in, &kx, &nx);
    int p = order;
    if ((size_t) p >= nx / 2)
    {
        p = (int) (nx / 4);
    }

    double  a[129];
    double *zX = NULL;
    fit_pooled_ar(in, order, P->ridge, a, &zX);
    if (zX == NULL)
    {
        return -1;
    }
    apply_causal_ar_from_z(zX, kx, nx, a, p, out->array.F);
    free(zX);
    return 0;
}

/* =========================================================================
 * 5. pos, val, fwhm = mlt_peak1d(curve, x0=0.0, dx=1.0)
 * ========================================================================= */

#define MLT_PEAK1D_PARAMS(X)                                                            \
    X(".x0", x0, FPTYPE_FLOAT64, 0.0, FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,           \
      "Physical coordinate at sample 0")                                                \
    X(".dx", dx, FPTYPE_FLOAT64, 1.0, FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,           \
      "Coordinate step per sample")

#define MLT_PEAK1D_PORTS(X)                                                 \
    X(IN, curve, MILK_PORT_IMAGE, 0, "1D curve to find peak on")            \
    X(OUT, pos, MILK_PORT_SCALAR, 0, "Parabolic-interpolated peak pos")     \
    X(OUT, val, MILK_PORT_SCALAR, 0, "Interpolated peak value")             \
    X(OUT, fwhm, MILK_PORT_SCALAR, 0, "Full width at half maximum")

MILK_CALC_OP_DECLARE(mlt_peak1d, MLT_PEAK1D_PARAMS, MLT_PEAK1D_PORTS);

static int mlt_peak1d_init(milk_calc_op_ctx_t *ctx, const void *P, void **state)
{
    (void) P;
    (void) state;
    const IMAGE *im = MILK_IN(ctx, curve);
    if (im == NULL || im->md[0].nelement == 0)
    {
        return -1;
    }
    return 0;
}

static int cmp_double(const void *a, const void *b)
{
    double va = *(const double *) a;
    double vb = *(const double *) b;
    return (va > vb) - (va < vb);
}

static int mlt_peak1d_exec(milk_calc_op_ctx_t *ctx, const void *P, void *state)
{
    (void) state;
    const struct mlt_peak1d_par *par = (const struct mlt_peak1d_par *) P;
    const IMAGE                 *im  = MILK_IN(ctx, curve);
    if (im == NULL || im->md[0].nelement == 0)
    {
        return -1;
    }

    size_t       n   = (size_t) im->md[0].nelement;
    const float *y   = im->array.F;
    size_t       idx = 0;
    double       maxv = -INFINITY;
    int          any  = 0;

    for (size_t i = 0; i < n; i++)
    {
        double v = (double) y[i];
        if (isfinite(v) && (!any || v > maxv))
        {
            maxv = v;
            idx  = i;
            any  = 1;
        }
    }
    if (!any)
    {
        MILK_OUT_SCALAR(ctx, pos)  = par->x0;
        MILK_OUT_SCALAR(ctx, val)  = 0.0;
        MILK_OUT_SCALAR(ctx, fwhm) = 0.0;
        return 0;
    }

    double delta    = 0.0;
    double peak_val = maxv;
    if (idx > 0 && idx + 1 < n)
    {
        double ym    = (double) y[idx - 1];
        double y0    = (double) y[idx];
        double yp    = (double) y[idx + 1];
        double denom = ym - 2.0 * y0 + yp;
        if (isfinite(denom) && denom < -1e-30)
        {
            delta = 0.5 * (ym - yp) / denom;
            if (delta < -0.5)
            {
                delta = -0.5;
            }
            if (delta > 0.5)
            {
                delta = 0.5;
            }
            peak_val = y0 - 0.25 * (ym - yp) * delta;
        }
    }

    double baseline = 0.0;
    double *tmp     = (double *) malloc(n * sizeof(double));
    if (tmp != NULL)
    {
        size_t nf = 0;
        for (size_t i = 0; i < n; i++)
        {
            if (isfinite(y[i]))
            {
                tmp[nf++] = (double) y[i];
            }
        }
        if (nf > 0)
        {
            qsort(tmp, nf, sizeof(double), cmp_double);
            baseline = tmp[nf / 2];
        }
        free(tmp);
    }

    double half = baseline + 0.5 * (peak_val - baseline);
    double x_l  = (double) idx;
    for (size_t i = idx; i > 0; i--)
    {
        if ((double) y[i - 1] <= half)
        {
            double d = (double) y[i] - (double) y[i - 1];
            double f = (d > 1e-30) ? ((half - (double) y[i - 1]) / d) : 0.0;
            x_l      = (double) (i - 1) + f;
            break;
        }
        if (i == 1)
        {
            x_l = 0.0;
        }
    }

    double x_r = (double) idx;
    for (size_t i = idx; i + 1 < n; i++)
    {
        if ((double) y[i + 1] <= half)
        {
            double d = (double) y[i] - (double) y[i + 1];
            double f = (d > 1e-30) ? (((double) y[i] - half) / d) : 0.0;
            x_r      = (double) i + f;
            break;
        }
        if (i + 2 == n)
        {
            x_r = (double) (n - 1);
        }
    }

    MILK_OUT_SCALAR(ctx, pos)  = par->x0 + ((double) idx + delta) * par->dx;
    MILK_OUT_SCALAR(ctx, val)  = peak_val;
    MILK_OUT_SCALAR(ctx, fwhm) = fabs(x_r - x_l) * fabs(par->dx);
    return 0;
}

static void mlt_peak1d_free(void *state)
{
    (void) state;
}

/* =========================================================================
 * 6. mlt_mask(cube, mask) and mlt_unmask(mat, mask)
 * ========================================================================= */

#define MLT_EMPTY_PARAMS(X)

#define MLT_MASK_PORTS(X)                                                     \
    X(IN, cube, MILK_PORT_IMAGE, 0, "2D (nx, ny) or 3D (nx, ny, K) input")    \
    X(IN, mask, MILK_PORT_IMAGE, 0, "2D (nx, ny) mask (> 0.5 selects pixel)") \
    X(OUT, mat, MILK_PORT_IMAGE, 0, "Packed 2D (P, K) matrix")

MILK_CALC_OP_DECLARE(mlt_mask, MLT_EMPTY_PARAMS, MLT_MASK_PORTS);

static int mlt_mask_init(milk_calc_op_ctx_t *ctx, const void *P, void **state)
{
    (void) P;
    (void) state;
    const IMAGE *cube = MILK_IN(ctx, cube);
    const IMAGE *mask = MILK_IN(ctx, mask);
    if (cube == NULL || mask == NULL)
    {
        return -1;
    }
    if (cube->md[0].naxis < 2 || mask->md[0].naxis < 2 ||
        cube->md[0].size[0] != mask->md[0].size[0] || cube->md[0].size[1] != mask->md[0].size[1])
    {
        return -1;
    }

    size_t   np = (size_t) mask->md[0].size[0] * (size_t) mask->md[0].size[1];
    uint32_t p  = 0;
    for (size_t i = 0; i < np; i++)
    {
        if (mask->array.F[i] > 0.5f)
        {
            p++;
        }
    }
    if (p == 0)
    {
        return -1;
    }
    uint32_t k = (cube->md[0].naxis >= 3) ? cube->md[0].size[2] : 1;
    milk_op_set_out_shape(ctx, mat, p, k);
    return 0;
}

static int mlt_mask_exec(milk_calc_op_ctx_t *ctx, const void *P, void *state)
{
    (void) P;
    (void) state;
    const IMAGE *cube = MILK_IN(ctx, cube);
    const IMAGE *mask = MILK_IN(ctx, mask);
    IMAGE       *mat  = MILK_OUT(ctx, mat);
    if (cube == NULL || mask == NULL || mat == NULL)
    {
        return -1;
    }

    size_t    np     = (size_t) mask->md[0].size[0] * (size_t) mask->md[0].size[1];
    uint32_t  p_out  = mat->md[0].size[0];
    uint32_t  k_out  = (mat->md[0].naxis >= 2) ? mat->md[0].size[1] : 1;
    uint32_t *idx    = (uint32_t *) malloc(p_out * sizeof(uint32_t));
    if (idx == NULL)
    {
        return -1;
    }

    uint32_t cnt = 0;
    for (size_t i = 0; i < np && cnt < p_out; i++)
    {
        if (mask->array.F[i] > 0.5f)
        {
            idx[cnt++] = (uint32_t) i;
        }
    }

    const float *src = cube->array.F;
    float       *dst = mat->array.F;
    uint32_t     f;
#pragma omp parallel for schedule(static) if ((size_t) k_out * p_out > 16384)
    for (f = 0; f < k_out; f++)
    {
        const float *s = src + (size_t) f * np;
        float       *d = dst + (size_t) f * p_out;
        for (uint32_t i = 0; i < cnt; i++)
        {
            d[i] = s[idx[i]];
        }
        for (uint32_t i = cnt; i < p_out; i++)
        {
            d[i] = 0.0f;
        }
    }

    free(idx);
    return 0;
}

static void mlt_mask_free(void *state)
{
    (void) state;
}

#define MLT_UNMASK_PORTS(X)                                                    \
    X(IN, mat, MILK_PORT_IMAGE, 0, "Packed 2D (P, K) or 1D (P) input")         \
    X(IN, mask, MILK_PORT_IMAGE, 0, "2D (nx, ny) mask (> 0.5 selects pixel)")  \
    X(OUT, cube, MILK_PORT_IMAGE, 0, "Unpacked 2D (nx, ny) or 3D (nx, ny, K)")

MILK_CALC_OP_DECLARE(mlt_unmask, MLT_EMPTY_PARAMS, MLT_UNMASK_PORTS);

static int mlt_unmask_init(milk_calc_op_ctx_t *ctx, const void *P, void **state)
{
    (void) P;
    (void) state;
    const IMAGE *mat  = MILK_IN(ctx, mat);
    const IMAGE *mask = MILK_IN(ctx, mask);
    if (mat == NULL || mask == NULL || mask->md[0].naxis < 2)
    {
        return -1;
    }

    uint32_t nx = mask->md[0].size[0];
    uint32_t ny = mask->md[0].size[1];
    uint32_t k  = (mat->md[0].naxis >= 2) ? mat->md[0].size[1] : 1;
    milk_op_set_out_shape3d(ctx, cube, nx, ny, k);
    return 0;
}

static int mlt_unmask_exec(milk_calc_op_ctx_t *ctx, const void *P, void *state)
{
    (void) P;
    (void) state;
    const IMAGE *mat  = MILK_IN(ctx, mat);
    const IMAGE *mask = MILK_IN(ctx, mask);
    IMAGE       *cube = MILK_OUT(ctx, cube);
    if (mat == NULL || mask == NULL || cube == NULL)
    {
        return -1;
    }

    size_t    np    = (size_t) mask->md[0].size[0] * (size_t) mask->md[0].size[1];
    uint32_t  p_in  = mat->md[0].size[0];
    uint32_t  k_in  = (mat->md[0].naxis >= 2) ? mat->md[0].size[1] : 1;
    uint32_t *idx   = (uint32_t *) malloc(p_in * sizeof(uint32_t));
    if (idx == NULL)
    {
        return -1;
    }

    uint32_t cnt = 0;
    for (size_t i = 0; i < np && cnt < p_in; i++)
    {
        if (mask->array.F[i] > 0.5f)
        {
            idx[cnt++] = (uint32_t) i;
        }
    }

    const float *src = mat->array.F;
    float       *dst = cube->array.F;
    memset(dst, 0, np * (size_t) k_in * sizeof(float));

    uint32_t f;
#pragma omp parallel for schedule(static) if ((size_t) k_in * np > 16384)
    for (f = 0; f < k_in; f++)
    {
        const float *s = src + (size_t) f * p_in;
        float       *d = dst + (size_t) f * np;
        for (uint32_t i = 0; i < cnt; i++)
        {
            d[idx[i]] = s[i];
        }
    }

    free(idx);
    return 0;
}

static void mlt_unmask_free(void *state)
{
    (void) state;
}

MILK_CALC_EXT_EXPORT(MILK_CALC_REG_OP(mlt_xcov,
                                      mlt_xcov,
                                      mlt_xcov_init,
                                      mlt_xcov_exec,
                                      mlt_xcov_free),
                     MILK_CALC_REG_OP(mlt_arwhiten,
                                      mlt_arwhiten,
                                      mlt_arwhiten_init,
                                      mlt_arwhiten_exec,
                                      mlt_arwhiten_free),
                     MILK_CALC_REG_IMG(mlt_arfilt1, mlt_arfilt1_img, mlt_arfilt1),
                     MILK_CALC_REG_OP(mlt_peak1d,
                                      mlt_peak1d,
                                      mlt_peak1d_init,
                                      mlt_peak1d_exec,
                                      mlt_peak1d_free),
                     MILK_CALC_REG_OP(mlt_mask,
                                      mlt_mask,
                                      mlt_mask_init,
                                      mlt_mask_exec,
                                      mlt_mask_free),
                     MILK_CALC_REG_OP(mlt_unmask,
                                      mlt_unmask,
                                      mlt_unmask_init,
                                      mlt_unmask_exec,
                                      mlt_unmask_free));
