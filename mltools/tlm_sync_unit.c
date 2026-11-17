// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    tlm_sync_unit.c
 * @brief   Resample telemetry stream B onto reference timeline A at arbitrary lag
 */

#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "milk_runnable.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "tlm_fileset.h"

static char   p_a_t[FUNCTION_PARAMETER_STRMAXLEN]  = "";
static char   p_b[FUNCTION_PARAMETER_STRMAXLEN]    = "";
static char   p_b_t[FUNCTION_PARAMETER_STRMAXLEN]  = "";
static char   p_out[FUNCTION_PARAMETER_STRMAXLEN]  = "@L:bsync";
static double p_lag                                = 0.0;
static char   p_mode[FUNCTION_PARAMETER_STRMAXLEN] = "linear";

static int64_t o_i0     = -1;
static int64_t o_i1     = -1;
static int64_t o_nvalid = 0;

#define FPS_PARAMS_TLMSYNC(X)                                                                   \
    X(".a_t", p_a_t, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "Ref timing image (<A>_t) or .txt") \
    X(".b", p_b, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "Source image B (P x Nb)")        \
    X(".b_t", p_b_t, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "Src timing image (<B>_t) or .txt") \
    X(".out", p_out, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "Output synced image")         \
    X(".lag", &p_lag, FPTYPE_FLOAT64, 0, FPFLAG_DEFAULT_INPUT,                                  \
      "Lag tau [s]: samples B at t_a + lag")                                                    \
    X(".mode", p_mode, FPTYPE_STRING_NOT_STREAM, 0, FPFLAG_DEFAULT_INPUT,                       \
      "Mode: linear|zoh|nearest|window")                                                        \
    X(".i0", &o_i0, FPTYPE_INT64, 0, FPFLAG_DEFAULT_OUTPUT, "First valid A frame index")        \
    X(".i1", &o_i1, FPTYPE_INT64, 0, FPFLAG_DEFAULT_OUTPUT, "Last valid A frame index")         \
    X(".nvalid", &o_nvalid, FPTYPE_INT64, 0, FPFLAG_DEFAULT_OUTPUT, "Valid overlapping frames")

MILK_PARAMS_DEFINE(mlt_tlmsync, FPS_PARAMS_TLMSYNC);

typedef enum
{
    MLT_SYNC_LINEAR  = 0,
    MLT_SYNC_ZOH     = 1,
    MLT_SYNC_NEAREST = 2,
    MLT_SYNC_WINDOW  = 3
} mlt_sync_mode;

static size_t lower_bound_double(const double *arr, size_t n, double val)
{
    size_t lo = 0;
    size_t hi = n;
    while (lo < hi)
    {
        size_t mid = lo + (hi - lo) / 2;
        if (arr[mid] < val)
        {
            lo = mid + 1;
        }
        else
        {
            hi = mid;
        }
    }
    return lo;
}

static errno_t tlmsync_run(void)
{
    mlt_sync_mode mode = MLT_SYNC_LINEAR;
    if (strcmp(p_mode, "linear") == 0 || p_mode[0] == '\0')
    {
        mode = MLT_SYNC_LINEAR;
    }
    else if (strcmp(p_mode, "zoh") == 0)
    {
        mode = MLT_SYNC_ZOH;
    }
    else if (strcmp(p_mode, "nearest") == 0)
    {
        mode = MLT_SYNC_NEAREST;
    }
    else if (strcmp(p_mode, "window") == 0)
    {
        mode = MLT_SYNC_WINDOW;
    }
    else
    {
        fprintf(stderr, "ERROR: unknown sync mode '%s' (use linear|zoh|nearest|window)\n", p_mode);
        return RETURN_FAILURE;
    }

    double *ta      = NULL;
    double *tb      = NULL;
    size_t  na      = 0;
    size_t  nb      = 0;
    int64_t ep_a    = 0;
    int64_t ep_b    = 0;

    if (mlt_load_timing_arg(p_a_t, &ta, NULL, &na, &ep_a) != RETURN_SUCCESS)
    {
        return RETURN_FAILURE;
    }
    if (mlt_load_timing_arg(p_b_t, &tb, NULL, &nb, &ep_b) != RETURN_SUCCESS)
    {
        free(ta);
        return RETURN_FAILURE;
    }

    IMGID bimg = imgid_make_from_name(p_b);
    resolveIMGID(&bimg, ERRMODE_NULL, dcimg, dcnimg);
    if (bimg.ID == -1)
    {
        read_sharedmem_image(bimg.name, dcimg, dcnimg);
        resolveIMGID(&bimg, ERRMODE_NULL, dcimg, dcnimg);
    }
    if (bimg.ID == -1)
    {
        fprintf(stderr, "ERROR: cannot resolve source image '%s'\n", p_b);
        imgid_free(&bimg);
        free(ta);
        free(tb);
        return RETURN_FAILURE;
    }

    IMAGE   *bim     = bimg.im;
    uint8_t  b_naxis = bim->md[0].naxis;
    uint32_t b_nx    = bim->md[0].size[0];
    uint32_t b_ny    = (b_naxis >= 2) ? bim->md[0].size[1] : 1;
    uint32_t b_nz    = (b_naxis >= 3) ? bim->md[0].size[2] : 1;
    uint32_t p_frame = 0;
    size_t   nb_img  = 0;

    if (b_naxis == 2)
    {
        p_frame = b_nx;
        nb_img  = b_ny;
    }
    else if (b_naxis == 3)
    {
        p_frame = b_nx * b_ny;
        nb_img  = b_nz;
    }
    else
    {
        fprintf(stderr, "ERROR: source image '%s' must be 2D or 3D, got naxis=%d\n", p_b,
                (int) b_naxis);
        imgid_free(&bimg);
        free(ta);
        free(tb);
        return RETURN_FAILURE;
    }

    if (nb_img != nb)
    {
        fprintf(stderr, "ERROR: source image frames (%zu) != timing rows (%zu)\n", nb_img, nb);
        imgid_free(&bimg);
        free(ta);
        free(tb);
        return RETURN_FAILURE;
    }

    if (bim->md[0].datatype != _DATATYPE_FLOAT)
    {
        fprintf(stderr, "ERROR: source image '%s' must be FLOAT\n", p_b);
        imgid_free(&bimg);
        free(ta);
        free(tb);
        return RETURN_FAILURE;
    }

    IMGID probe_out = imgid_make_from_name(p_out);
    if (probe_out.name[0] != '\0')
    {
        delete_image_ID(probe_out.name, DELETE_IMAGE_ERRMODE_IGNORE);
    }
    imgid_free(&probe_out);

    uint32_t dims[3] = { 1, 1, 1 };
    if (b_naxis == 2)
    {
        dims[0] = b_nx;
        dims[1] = (uint32_t) na;
    }
    else
    {
        dims[0] = b_nx;
        dims[1] = b_ny;
        dims[2] = (uint32_t) na;
    }

    imageID out_id = -1;
    if (create_image_ID(p_out, (long) b_naxis, dims, _DATATYPE_FLOAT, 0, IMGID_NB_KEYWO_MAX, 0,
                        &out_id) != RETURN_SUCCESS ||
        out_id < 0)
    {
        imgid_free(&bimg);
        free(ta);
        free(tb);
        return RETURN_FAILURE;
    }

    if (fps_cli_is_standalone() && dcimg[out_id].md[0].shared == 0)
    {
        fprintf(stderr,
                "tlmsync: '%s' would be lost when this command exits: name it @S:%s\n",
                p_out, dcimg[out_id].md[0].name);
        delete_image_ID(dcimg[out_id].md[0].name, DELETE_IMAGE_ERRMODE_IGNORE);
        imgid_free(&bimg);
        free(ta);
        free(tb);
        return RETURN_FAILURE;
    }

    double epoch_off = (double) (ep_a - ep_b);
    double shift     = epoch_off + p_lag;
    double tb_min    = tb[0];
    double tb_max    = tb[nb - 1];
    double tol       = 0.0;
    if (mode == MLT_SYNC_NEAREST && nb >= 2)
    {
        tol = 0.5 * (tb[nb - 1] - tb[0]) / (double) (nb - 1);
    }

    int64_t i0 = -1;
    int64_t i1 = -1;
    for (size_t i = 0; i < na; i++)
    {
        double tq = ta[i] + shift;
        if (tq >= tb_min - tol && tq <= tb_max + tol)
        {
            if (i0 < 0)
            {
                i0 = (int64_t) i;
            }
            i1 = (int64_t) i;
        }
    }

    const float *b_arr   = bim->array.F;
    float       *out_arr = dcimg[out_id].array.F;
    size_t       pf      = (size_t) p_frame;

    if (i0 >= 0 && i1 >= i0)
    {
        size_t s0 = (size_t) i0;
        size_t s1 = (size_t) i1;
        size_t i;
#pragma omp parallel for schedule(static) if (na > 64 && pf * na > 16384)
        for (i = 0; i < na; i++)
        {
            float *dst = out_arr + i * pf;
            if (i < s0 || i > s1)
            {
                memset(dst, 0, pf * sizeof(float));
                continue;
            }

            if (mode == MLT_SYNC_WINDOW && na >= 2 && nb >= 2)
            {
                double t_c  = ta[i] + shift;
                double d_lo = (i > 0) ? 0.5 * (ta[i] - ta[i - 1]) : 0.5 * (ta[1] - ta[0]);
                double d_hi = (i + 1 < na) ? 0.5 * (ta[i + 1] - ta[i])
                                           : 0.5 * (ta[na - 1] - ta[na - 2]);
                double w_lo = t_c - d_lo;
                double w_hi = t_c + d_hi;
                if (w_hi <= w_lo)
                {
                    w_hi = w_lo + 1e-12;
                }

                size_t kstart = lower_bound_double(tb, nb, w_lo);
                if (kstart > 0)
                {
                    kstart--;
                }
                memset(dst, 0, pf * sizeof(float));
                double wsum = 0.0;
                for (size_t k = kstart; k < nb; k++)
                {
                    double b_lo = (k > 0) ? 0.5 * (tb[k - 1] + tb[k])
                                          : tb[0] - 0.5 * (tb[1] - tb[0]);
                    double b_hi = (k + 1 < nb) ? 0.5 * (tb[k] + tb[k + 1])
                                               : tb[nb - 1] + 0.5 * (tb[nb - 1] - tb[nb - 2]);
                    if (b_lo >= w_hi)
                    {
                        break;
                    }
                    double ov_lo = (b_lo > w_lo) ? b_lo : w_lo;
                    double ov_hi = (b_hi < w_hi) ? b_hi : w_hi;
                    double w     = ov_hi - ov_lo;
                    if (w > 0.0)
                    {
                        float        wf  = (float) w;
                        const float *src = b_arr + k * pf;
#pragma omp simd
                        for (size_t p = 0; p < pf; p++)
                        {
                            dst[p] += wf * src[p];
                        }
                        wsum += w;
                    }
                }
                if (wsum > 0.0)
                {
                    float invw = (float) (1.0 / wsum);
#pragma omp simd
                    for (size_t p = 0; p < pf; p++)
                    {
                        dst[p] *= invw;
                    }
                }
                continue;
            }

            double tq = ta[i] + shift;
            if (tq <= tb[0])
            {
                memcpy(dst, b_arr, pf * sizeof(float));
                continue;
            }
            if (tq >= tb[nb - 1])
            {
                memcpy(dst, b_arr + (nb - 1) * pf, pf * sizeof(float));
                continue;
            }

            size_t k1 = lower_bound_double(tb, nb, tq);
            if (k1 == 0)
            {
                k1 = 1;
            }
            if (k1 >= nb)
            {
                k1 = nb - 1;
            }
            size_t k0 = k1 - 1;

            if (mode == MLT_SYNC_ZOH)
            {
                size_t       k   = (tb[k1] <= tq) ? k1 : k0;
                const float *src = b_arr + k * pf;
                memcpy(dst, src, pf * sizeof(float));
            }
            else if (mode == MLT_SYNC_NEAREST)
            {
                size_t       k   = ((tq - tb[k0]) <= (tb[k1] - tq)) ? k0 : k1;
                const float *src = b_arr + k * pf;
                memcpy(dst, src, pf * sizeof(float));
            }
            else
            {
                double dt = tb[k1] - tb[k0];
                float  a  = (dt > 1e-15) ? (float) ((tq - tb[k0]) / dt) : 0.0f;
                float  om = 1.0f - a;
                const float *s0p = b_arr + k0 * pf;
                const float *s1p = b_arr + k1 * pf;
#pragma omp simd
                for (size_t p = 0; p < pf; p++)
                {
                    dst[p] = om * s0p[p] + a * s1p[p];
                }
            }
        }
    }
    else
    {
        memset(out_arr, 0, na * pf * sizeof(float));
    }

    o_i0     = i0;
    o_i1     = i1;
    o_nvalid = (i0 >= 0 && i1 >= i0) ? (i1 - i0 + 1) : 0;

    const char *out_bare = dcimg[out_id].md[0].name;
    image_write_keyword_L(out_bare, "SYNCI0", (long) o_i0, "First valid ref frame");
    image_write_keyword_L(out_bare, "SYNCI1", (long) o_i1, "Last valid ref frame");
    image_write_keyword_L(out_bare, "SYNCNV", (long) o_nvalid, "Valid synced frames");
    image_write_keyword_D(out_bare, "SYNCLAG", p_lag, "Applied lag [s]");

    if (!dcquiet)
    {
        printf("mlt.tlmsync: %s -> %s (%s, lag=%.6fs, valid=[%lld..%lld], nvalid=%lld)\n",
               bimg.name, out_bare, p_mode, p_lag, (long long) o_i0, (long long) o_i1,
               (long long) o_nvalid);
    }

    imgid_free(&bimg);
    free(ta);
    free(tb);
    return RETURN_SUCCESS;
}

static MILK_HOT errno_t tlmsync_compute(const MILK_RUNNABLE *self)
{
    errno_t ret = RETURN_SUCCESS;
    DEBUG_TRACE_FSTART();
    MILK_PROCINFO_INIT(self)
    MILK_PROCINFO_LOOPSTART(self)

    ret = tlmsync_run();
    if (ret == RETURN_SUCCESS && dcfpsptr != NULL)
    {
        fpsresync_modvar_to_fps(dcfpsptr, dcfpsbindings, dcfpsnbindings);
    }

    MILK_PROCINFO_END(self)
    DEBUG_TRACE_FEXIT();
    return ret;
}

const MILK_RUNNABLE runnable_mlt_tlmsync = {
    .kind = MILK_RUNNABLE_FPS,
    .app  = { .fps_name         = "mlttlmsync",
              .cmdkey           = "tlmsync",
              .description      = "resample telemetry stream B onto A timestamps at given lag",
              .description_long = "Resample 2D (P, Nb) or 3D (nx, ny, Nb) stream B onto "
                                  "reference timeline A at query times t_a[i] + lag using linear, "
                                  "zero-order-hold, nearest, or exposure-window interpolation." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(mlt_tlmsync),
    .compute = tlmsync_compute,
};

EXPORT_FPSEXEC(milk, "mlt-tlmsync", runnable_mlt_tlmsync);
