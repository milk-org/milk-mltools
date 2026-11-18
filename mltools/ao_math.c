// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    ao_math.c
 * @brief   Shared FITS/SHM array loaders, radix-2 FFTs, pupil/DM stencils, & radial PSF metrics
 */

// MILK_CMAKE_MANDATE_CFITSIO

#ifndef _GNU_SOURCE
#    define _GNU_SOURCE
#endif

#include <errno.h>
#include <fitsio.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "milk_runnable.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "ao_math.h"
#include "tlm_fileset.h"

#ifndef M_PI
#    define M_PI 3.14159265358979323846
#endif

static errno_t ensure_parent_dirs(const char *filepath)
{
    char   tmp[MLT_PATH_MAX];
    int    n = snprintf(tmp, sizeof(tmp), "%s", filepath);
    char  *slash;
    char  *p;

    if (n < 0 || (size_t) n >= sizeof(tmp))
    {
        return RETURN_FAILURE;
    }
    slash = strrchr(tmp, '/');
    if (slash == NULL || slash == tmp)
    {
        return RETURN_SUCCESS;
    }
    *slash = '\0';

    for (p = tmp + 1; *p != '\0'; p++)
    {
        if (*p == '/')
        {
            *p = '\0';
            if (mkdir(tmp, 0775) != 0 && errno != EEXIST)
            {
                return RETURN_FAILURE;
            }
            *p = '/';
        }
    }
    if (mkdir(tmp, 0775) != 0 && errno != EEXIST)
    {
        return RETURN_FAILURE;
    }
    return RETURN_SUCCESS;
}

static errno_t read_fits_cube_f32(const char *path,
                                  float     **buf_out,
                                  uint32_t   *nx_out,
                                  uint32_t   *ny_out,
                                  uint32_t   *nz_out)
{
    fitsfile *fptr     = NULL;
    int       status   = 0;
    int       naxis    = 0;
    long      naxes[4] = { 1, 1, 1, 1 };
    long      fpixel[4] = { 1, 1, 1, 1 };

    fits_open_image(&fptr, path, READONLY, &status);
    if (status != 0)
    {
        char msg[FLEN_ERRMSG];
        fits_get_errstatus(status, msg);
        fprintf(stderr, "ERROR: CFITSIO open '%s': %s\n", path, msg);
        return RETURN_FAILURE;
    }
    fits_get_img_dim(fptr, &naxis, &status);
    if (status == 0 && naxis >= 1)
    {
        fits_get_img_size(fptr, (naxis > 4) ? 4 : naxis, naxes, &status);
    }
    if (status != 0 || naxis < 1 || naxes[0] <= 0)
    {
        int cstat = 0;
        fits_close_file(fptr, &cstat);
        fprintf(stderr, "ERROR: invalid FITS image '%s' (naxis=%d)\n", path, naxis);
        return RETURN_FAILURE;
    }

    uint32_t nx   = (uint32_t) naxes[0];
    uint32_t ny   = (naxis >= 2 && naxes[1] > 0) ? (uint32_t) naxes[1] : 1u;
    uint32_t nz   = (naxis >= 3 && naxes[2] > 0) ? (uint32_t) naxes[2] : 1u;
    size_t   ntot = (size_t) nx * ny * nz;
    float   *buf  = (float *) malloc(ntot * sizeof(float));
    if (buf == NULL)
    {
        int cstat = 0;
        fits_close_file(fptr, &cstat);
        return RETURN_FAILURE;
    }

    int anynul = 0;
    fits_read_pix(fptr, TFLOAT, fpixel, (LONGLONG) ntot, NULL, buf, &anynul, &status);
    int cstat = 0;
    fits_close_file(fptr, &cstat);
    if (status != 0)
    {
        char msg[FLEN_ERRMSG];
        fits_get_errstatus(status, msg);
        fprintf(stderr, "ERROR: CFITSIO read '%s': %s\n", path, msg);
        free(buf);
        return RETURN_FAILURE;
    }

    *buf_out = buf;
    *nx_out  = nx;
    *ny_out  = ny;
    *nz_out  = nz;
    return RETURN_SUCCESS;
}

errno_t mlt_load_array_f32(const char *spec,
                           const char *stream_opt,
                           float     **buf_out,
                           uint32_t   *nx_out,
                           uint32_t   *ny_out,
                           uint32_t   *nz_out)
{
    *buf_out = NULL;
    *nx_out  = 0;
    *ny_out  = 0;
    *nz_out  = 0;

    if (spec == NULL || spec[0] == '\0' || strcmp(spec, "none") == 0)
    {
        return RETURN_FAILURE;
    }

    /* 1. Check if @spec is a directory containing telemetry cubes */
    struct stat st;
    if (stat(spec, &st) == 0 && S_ISDIR(st.st_mode))
    {
        MLT_FILESET fs;
        if (mlt_fileset_scan(spec, stream_opt, "auto", &fs) != RETURN_SUCCESS)
        {
            return RETURN_FAILURE;
        }
        if (fs.ncubes == 1)
        {
            errno_t rc = read_fits_cube_f32(fs.entries[0].fits_path, buf_out, nx_out, ny_out,
                                            nz_out);
            mlt_fileset_free(&fs);
            return rc;
        }

        uint32_t c_nx     = 0;
        uint32_t c_ny     = 0;
        uint32_t total_nz = 0;
        float  **cubes    = (float **) calloc(fs.ncubes, sizeof(float *));
        uint32_t *czs     = (uint32_t *) calloc(fs.ncubes, sizeof(uint32_t));
        if (cubes == NULL || czs == NULL)
        {
            free(cubes);
            free(czs);
            mlt_fileset_free(&fs);
            return RETURN_FAILURE;
        }

        for (size_t c = 0; c < fs.ncubes; c++)
        {
            uint32_t x = 0;
            uint32_t y = 0;
            uint32_t z = 0;
            if (read_fits_cube_f32(fs.entries[c].fits_path, &cubes[c], &x, &y, &z) !=
                RETURN_SUCCESS)
            {
                for (size_t j = 0; j < c; j++)
                {
                    free(cubes[j]);
                }
                free(cubes);
                free(czs);
                mlt_fileset_free(&fs);
                return RETURN_FAILURE;
            }
            if (c == 0)
            {
                c_nx = x;
                c_ny = y;
            }
            else if (x != c_nx || y != c_ny)
            {
                for (size_t j = 0; j <= c; j++)
                {
                    free(cubes[j]);
                }
                free(cubes);
                free(czs);
                mlt_fileset_free(&fs);
                fprintf(stderr, "ERROR: mismatched cube spatial dims in '%s'\n", spec);
                return RETURN_FAILURE;
            }
            czs[c] = z;
            total_nz += z;
        }

        size_t fpix = (size_t) c_nx * c_ny;
        float *cat  = (float *) malloc(fpix * total_nz * sizeof(float));
        if (cat == NULL)
        {
            for (size_t j = 0; j < fs.ncubes; j++)
            {
                free(cubes[j]);
            }
            free(cubes);
            free(czs);
            mlt_fileset_free(&fs);
            return RETURN_FAILURE;
        }

        size_t zoff = 0;
        for (size_t c = 0; c < fs.ncubes; c++)
        {
            memcpy(cat + zoff * fpix, cubes[c], fpix * czs[c] * sizeof(float));
            zoff += czs[c];
            free(cubes[c]);
        }
        free(cubes);
        free(czs);
        mlt_fileset_free(&fs);

        *buf_out = cat;
        *nx_out  = c_nx;
        *ny_out  = c_ny;
        *nz_out  = total_nz;
        return RETURN_SUCCESS;
    }

    /* 2. Check if @spec is a readable file on disk (FITS / .fits.fz) */
    if (stat(spec, &st) == 0 && S_ISREG(st.st_mode))
    {
        return read_fits_cube_f32(spec, buf_out, nx_out, ny_out, nz_out);
    }

    /* 3. Resolve in-memory or SHM MILK image */
    IMGID img = imgid_make_from_name(spec);
    resolveIMGID(&img, ERRMODE_NULL, dcimg, dcnimg);
    if (img.ID == -1 && img.name[0] != '\0')
    {
        read_sharedmem_image(img.name, dcimg, dcnimg);
        resolveIMGID(&img, ERRMODE_NULL, dcimg, dcnimg);
    }
    if (img.ID == -1 || img.im == NULL)
    {
        imgid_free(&img);
        fprintf(stderr, "ERROR: cannot resolve array/FITS '%s'\n", spec);
        return RETURN_FAILURE;
    }

    IMAGE   *im    = img.im;
    uint8_t  naxis = im->md[0].naxis;
    uint32_t nx    = im->md[0].size[0];
    uint32_t ny    = (naxis >= 2) ? im->md[0].size[1] : 1u;
    uint32_t nz    = (naxis >= 3) ? im->md[0].size[2] : 1u;
    size_t   ntot  = (size_t) nx * ny * nz;
    float   *buf   = (float *) malloc(ntot * sizeof(float));
    if (buf == NULL)
    {
        imgid_free(&img);
        return RETURN_FAILURE;
    }

    switch (im->md[0].datatype)
    {
    case _DATATYPE_FLOAT:
        memcpy(buf, im->array.F, ntot * sizeof(float));
        break;
    case _DATATYPE_DOUBLE:
        for (size_t i = 0; i < ntot; i++)
        {
            buf[i] = (float) im->array.D[i];
        }
        break;
    case _DATATYPE_UINT16:
        for (size_t i = 0; i < ntot; i++)
        {
            buf[i] = (float) im->array.UI16[i];
        }
        break;
    case _DATATYPE_INT16:
        for (size_t i = 0; i < ntot; i++)
        {
            buf[i] = (float) im->array.SI16[i];
        }
        break;
    case _DATATYPE_INT32:
        for (size_t i = 0; i < ntot; i++)
        {
            buf[i] = (float) im->array.SI32[i];
        }
        break;
    case _DATATYPE_UINT32:
        for (size_t i = 0; i < ntot; i++)
        {
            buf[i] = (float) im->array.UI32[i];
        }
        break;
    default:
        free(buf);
        imgid_free(&img);
        fprintf(stderr, "ERROR: unsupported MILK datatype %d for '%s'\n", (int) im->md[0].datatype,
                spec);
        return RETURN_FAILURE;
    }

    imgid_free(&img);
    *buf_out = buf;
    *nx_out  = nx;
    *ny_out  = ny;
    *nz_out  = nz;
    return RETURN_SUCCESS;
}

errno_t mlt_write_fits_f32(const char        *path,
                           const float       *data,
                           int                naxis,
                           const long        *naxes,
                           const mlt_fits_kw *kws,
                           size_t             nkws)
{
    if (path == NULL || data == NULL || naxis < 1 || naxis > 4)
    {
        return RETURN_FAILURE;
    }
    if (ensure_parent_dirs(path) != RETURN_SUCCESS)
    {
        fprintf(stderr, "ERROR: cannot create parent directory for '%s'\n", path);
        return RETURN_FAILURE;
    }

    char bang[MLT_PATH_MAX + 4];
    int  nb = snprintf(bang, sizeof(bang), "!%s", (path[0] == '!') ? (path + 1) : path);
    if (nb < 0 || (size_t) nb >= sizeof(bang))
    {
        return RETURN_FAILURE;
    }

    LONGLONG ntot = 1;
    for (int a = 0; a < naxis; a++)
    {
        ntot *= (LONGLONG) naxes[a];
    }

    fitsfile *fptr   = NULL;
    int       status = 0;
    long      ax_mut[4] = { 1, 1, 1, 1 };
    long      fpixel[4] = { 1, 1, 1, 1 };
    for (int a = 0; a < naxis; a++)
    {
        ax_mut[a] = naxes[a];
    }

    fits_create_file(&fptr, bang, &status);
    fits_create_img(fptr, FLOAT_IMG, naxis, ax_mut, &status);
    for (size_t k = 0; k < nkws && status == 0; k++)
    {
        if (kws[k].type == 'D')
        {
            double v = kws[k].val_d;
            fits_update_key(fptr, TDOUBLE, kws[k].key, &v, (char *) kws[k].comment, &status);
        }
        else if (kws[k].type == 'L')
        {
            long v = kws[k].val_l;
            fits_update_key(fptr, TLONG, kws[k].key, &v, (char *) kws[k].comment, &status);
        }
        else if (kws[k].type == 'S')
        {
            fits_update_key(fptr, TSTRING, kws[k].key, (void *) kws[k].val_s,
                            (char *) kws[k].comment, &status);
        }
    }
    if (status == 0)
    {
        fits_write_pix(fptr, TFLOAT, fpixel, ntot, (void *) data, &status);
    }

    int cstat = 0;
    if (fptr != NULL)
    {
        fits_close_file(fptr, &cstat);
    }
    if (status != 0 || cstat != 0)
    {
        char msg[FLEN_ERRMSG];
        fits_get_errstatus(status != 0 ? status : cstat, msg);
        fprintf(stderr, "ERROR: CFITSIO write '%s': %s\n", path, msg);
        return RETURN_FAILURE;
    }
    return RETURN_SUCCESS;
}

uint32_t mlt_next_pow2_u32(uint32_t v)
{
    uint32_t p = 2u;
    while (p < v && p < (1u << 30))
    {
        p <<= 1u;
    }
    return p;
}

void mlt_fft1d_c2c(float *re_im, uint32_t n, int inverse)
{
    if (n < 2u)
    {
        return;
    }

    /* Bit-reversal permutation */
    uint32_t j = 0u;
    for (uint32_t i = 1u; i < n; i++)
    {
        uint32_t bit = n >> 1u;
        while ((j & bit) != 0u)
        {
            j ^= bit;
            bit >>= 1u;
        }
        j ^= bit;
        if (i < j)
        {
            float tr          = re_im[2u * i];
            float ti          = re_im[2u * i + 1u];
            re_im[2u * i]     = re_im[2u * j];
            re_im[2u * i + 1u] = re_im[2u * j + 1u];
            re_im[2u * j]     = tr;
            re_im[2u * j + 1u] = ti;
        }
    }

    const double sign = inverse ? 1.0 : -1.0;
    for (uint32_t len = 2u; len <= n; len <<= 1u)
    {
        uint32_t half = len >> 1u;
        double   ang  = sign * 2.0 * M_PI / (double) len;
        double   wlr  = cos(ang);
        double   wli  = sin(ang);

        for (uint32_t i = 0u; i < n; i += len)
        {
            double wr = 1.0;
            double wi = 0.0;
            for (uint32_t k = 0u; k < half; k++)
            {
                uint32_t u  = i + k;
                uint32_t v  = u + half;
                float    vr = (float) (re_im[2u * v] * wr - re_im[2u * v + 1u] * wi);
                float    vi = (float) (re_im[2u * v] * wi + re_im[2u * v + 1u] * wr);
                float    ur = re_im[2u * u];
                float    ui = re_im[2u * u + 1u];

                re_im[2u * u]      = ur + vr;
                re_im[2u * u + 1u] = ui + vi;
                re_im[2u * v]      = ur - vr;
                re_im[2u * v + 1u] = ui - vi;

                double nwr = wr * wlr - wi * wli;
                wi         = wr * wli + wi * wlr;
                wr         = nwr;
            }
        }
    }

    if (inverse)
    {
        float inv_n = 1.0f / (float) n;
        for (uint32_t i = 0u; i < 2u * n; i++)
        {
            re_im[i] *= inv_n;
        }
    }
}

void mlt_fft2d_c2c(float *re_im, uint32_t nx, uint32_t ny, int inverse)
{
    uint32_t y;
#pragma omp parallel for schedule(static) if (ny >= 32)
    for (y = 0; y < ny; y++)
    {
        mlt_fft1d_c2c(&re_im[(size_t) y * nx * 2u], nx, inverse);
    }

#pragma omp parallel if (nx >= 32)
    {
        float *col = (float *) malloc((size_t) ny * 2u * sizeof(float));
        if (col != NULL)
        {
            uint32_t x;
#pragma omp for schedule(static)
            for (x = 0; x < nx; x++)
            {
                for (uint32_t r = 0; r < ny; r++)
                {
                    size_t idx     = ((size_t) r * nx + x) * 2u;
                    col[2u * r]     = re_im[idx];
                    col[2u * r + 1u] = re_im[idx + 1u];
                }
                mlt_fft1d_c2c(col, ny, inverse);
                for (uint32_t r = 0; r < ny; r++)
                {
                    size_t idx      = ((size_t) r * nx + x) * 2u;
                    re_im[idx]      = col[2u * r];
                    re_im[idx + 1u] = col[2u * r + 1u];
                }
            }
            free(col);
        }
    }
}

void mlt_make_pupil(uint32_t nwf,
                    float    pup_diam,
                    float    cobs,
                    int      nspider,
                    float    spider_w,
                    float    spider_deg,
                    float   *pup_out)
{
    const float r_out  = 0.5f * pup_diam * (float) nwf;
    const float r_in   = cobs * r_out;
    const float r_out2 = r_out * r_out;
    const float r_in2  = r_in * r_in;
    const float half_w = 0.5f * spider_w * r_out * 2.0f;
    const float ctr    = 0.5f * (float) (nwf - 1u);
    const float deg0   = spider_deg * (float) (M_PI / 180.0);

    float sp_cos[8];
    float sp_sin[8];
    int   ns = (nspider > 8) ? 8 : ((nspider > 0) ? nspider : 0);
    for (int s = 0; s < ns; s++)
    {
        float ang = deg0 + (float) s * (float) (2.0 * M_PI) / (float) ns;
        sp_cos[s] = cosf(ang);
        sp_sin[s] = sinf(ang);
    }

    uint32_t jy;
#pragma omp parallel for schedule(static) if (nwf >= 64)
    for (jy = 0; jy < nwf; jy++)
    {
        for (uint32_t ix = 0; ix < nwf; ix++)
        {
            int hits = 0;
            for (int sy = 0; sy < 4; sy++)
            {
                float y = ((float) jy - ctr) + ((float) sy - 1.5f) * 0.25f;
                for (int sx = 0; sx < 4; sx++)
                {
                    float x  = ((float) ix - ctr) + ((float) sx - 1.5f) * 0.25f;
                    float r2 = x * x + y * y;
                    if (r2 > r_out2 || r2 < r_in2)
                    {
                        continue;
                    }
                    int blocked = 0;
                    for (int s = 0; s < ns; s++)
                    {
                        float along = x * sp_cos[s] + y * sp_sin[s];
                        float perp  = fabsf(-x * sp_sin[s] + y * sp_cos[s]);
                        if (along >= 0.0f && perp <= half_w)
                        {
                            blocked = 1;
                            break;
                        }
                    }
                    if (!blocked)
                    {
                        hits++;
                    }
                }
            }
            pup_out[(size_t) jy * nwf + ix] = (float) hits * (1.0f / 16.0f);
        }
    }
}

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
                                     float         **u_act_out)
{
    *u_act_out = NULL;
    if (modes_dm == NULL || act_idx == NULL || nmodes == 0 || n_act == 0)
    {
        return RETURN_FAILURE;
    }

    float *u_act = (float *) calloc((size_t) n_act * nmodes, sizeof(float));
    if (u_act == NULL)
    {
        return RETURN_FAILURE;
    }

    /* Direct pupil-grid basis (`nact_x == nwf && nact_y == nwf`) */
    if (nact_x == nwf && nact_y == nwf)
    {
        size_t npix = (size_t) nwf * nwf;
        uint32_t p;
#pragma omp parallel for schedule(static)
        for (p = 0; p < n_act; p++)
        {
            uint32_t pix = act_idx[p];
            float   *dst = &u_act[(size_t) p * nmodes];
            for (uint32_t m = 0; m < nmodes; m++)
            {
                dst[m] = opd_scale * modes_dm[(size_t) m * npix + pix];
            }
        }
        *u_act_out = u_act;
        return RETURN_SUCCESS;
    }

    const float diam  = (dm_diam > 0.05f && dm_diam <= 1.25f) ? dm_diam : 0.98f;
    const float span  = diam * (float) nwf;
    const float ctr   = 0.5f * (float) (nwf - 1u);
    const float px    = (nact_x > 1u) ? (span / (float) (nact_x - 1u)) : span;
    const float py    = (nact_y > 1u) ? (span / (float) (nact_y - 1u)) : span;
    const float x0    = ctr - 0.5f * (float) (nact_x - 1u) * px;
    const float y0    = ctr - 0.5f * (float) (nact_y - 1u) * py;
    float       coupl = (if_coupl < 0.005f) ? 0.005f : ((if_coupl > 0.45f) ? 0.45f : if_coupl);
    const float alpha = -logf(coupl);
    const size_t nact2 = (size_t) nact_x * nact_y;

    uint32_t p;
#pragma omp parallel for schedule(static)
    for (p = 0; p < n_act; p++)
    {
        uint32_t pix = act_idx[p];
        uint32_t ix  = pix % nwf;
        uint32_t jy  = pix / nwf;
        float    fx  = ((float) ix - x0) / px;
        float    fy  = ((float) jy - y0) / py;
        int      ax0 = (int) floorf(fx) - 1;
        int      ay0 = (int) floorf(fy) - 1;

        float lat_sum = 0.0f;
        for (int dy = 0; dy < 4; dy++)
        {
            float ry = fy - (float) (ay0 + dy);
            for (int dx = 0; dx < 4; dx++)
            {
                float rx = fx - (float) (ax0 + dx);
                lat_sum += expf(-alpha * (rx * rx + ry * ry));
            }
        }
        float w_scale = (lat_sum > 1e-8f) ? (opd_scale / lat_sum) : opd_scale;

        int   t_idx[MLT_DM_STENCIL];
        float t_w[MLT_DM_STENCIL];
        int   ntap = 0;
        for (int dy = 0; dy < 4; dy++)
        {
            int ay = ay0 + dy;
            if (ay < 0 || ay >= (int) nact_y)
            {
                continue;
            }
            float ry = fy - (float) ay;
            for (int dx = 0; dx < 4; dx++)
            {
                int ax = ax0 + dx;
                if (ax < 0 || ax >= (int) nact_x)
                {
                    continue;
                }
                float rx      = fx - (float) ax;
                t_idx[ntap]   = ay * (int) nact_x + ax;
                t_w[ntap]     = w_scale * expf(-alpha * (rx * rx + ry * ry));
                ntap++;
            }
        }

        float *dst = &u_act[(size_t) p * nmodes];
        for (uint32_t m = 0; m < nmodes; m++)
        {
            const float *mslice = &modes_dm[(size_t) m * nact2];
            float        acc    = 0.0f;
            for (int t = 0; t < ntap; t++)
            {
                acc += t_w[t] * mslice[t_idx[t]];
            }
            dst[m] = acc;
        }
    }

    *u_act_out = u_act;
    return RETURN_SUCCESS;
}

void mlt_psf_metrics(const float *img,
                     uint32_t     ncam,
                     float        ee_rad_pix,
                     double      *peak_out,
                     double      *fwhm_pix_out,
                     double      *ee_out,
                     float       *rad_prof_out,
                     uint32_t     nbins)
{
    const uint32_t c0   = ncam / 2u;
    size_t         pk_i = (size_t) c0 * ncam + c0;
    double         peak = -1e30;
    double         tot  = 0.0;

    /* Locate peak within central box `[c0-16 .. c0+16]` to track slight tip/tilt offsets */
    int32_t rbox = (ncam > 40u) ? 16 : (int32_t) (ncam / 4u);
    for (uint32_t y = 0; y < ncam; y++)
    {
        for (uint32_t x = 0; x < ncam; x++)
        {
            double v = (double) img[(size_t) y * ncam + x];
            tot += v;
            if (abs((int) x - (int) c0) <= rbox && abs((int) y - (int) c0) <= rbox && v > peak)
            {
                peak = v;
                pk_i = (size_t) y * ncam + x;
            }
        }
    }
    double cx = (double) (pk_i % ncam);
    double cy = (double) (pk_i / ncam);

    double ee_sum = 0.0;
    double r_ee2  = (double) ee_rad_pix * (double) ee_rad_pix;
    double *bsum  = (nbins > 0) ? (double *) calloc(nbins, sizeof(double)) : NULL;
    size_t *bcnt  = (nbins > 0) ? (size_t *) calloc(nbins, sizeof(size_t)) : NULL;

    for (uint32_t y = 0; y < ncam; y++)
    {
        double dy = (double) y - cy;
        for (uint32_t x = 0; x < ncam; x++)
        {
            double dx = (double) x - cx;
            double r2 = dx * dx + dy * dy;
            double v  = (double) img[(size_t) y * ncam + x];
            if (r2 <= r_ee2)
            {
                ee_sum += v;
            }
            if (bsum != NULL)
            {
                uint32_t b = (uint32_t) floor(sqrt(r2));
                if (b < nbins)
                {
                    bsum[b] += v;
                    bcnt[b]++;
                }
            }
        }
    }

    double fwhm_pix = 0.0;
    if (bsum != NULL)
    {
        double half_pk = 0.5 * peak;
        int    found   = 0;
        for (uint32_t b = 0; b < nbins; b++)
        {
            double avg = (bcnt[b] > 0) ? (bsum[b] / (double) bcnt[b]) : 0.0;
            if (rad_prof_out != NULL)
            {
                rad_prof_out[b] = (float) avg;
            }
            if (!found && b > 0 && avg <= half_pk)
            {
                double prev = (bcnt[b - 1] > 0) ? (bsum[b - 1] / (double) bcnt[b - 1]) : peak;
                double denom = prev - avg;
                double frac  = (denom > 1e-30) ? ((prev - half_pk) / denom) : 0.0;
                double r_hwhm = ((double) (b - 1) + frac);
                fwhm_pix      = 2.0 * r_hwhm;
                found         = 1;
            }
        }
        free(bsum);
        free(bcnt);
    }

    if (peak_out != NULL)
    {
        *peak_out = peak;
    }
    if (fwhm_pix_out != NULL)
    {
        *fwhm_pix_out = fwhm_pix;
    }
    if (ee_out != NULL)
    {
        *ee_out = (tot > 1e-30) ? (ee_sum / tot) : 0.0;
    }
}
