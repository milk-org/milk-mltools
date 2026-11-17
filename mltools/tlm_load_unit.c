// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    tlm_load_unit.c
 * @brief   Parallel loader for telemetry `.fits`/`.fits.fz` cubes + `.txt` timing files
 */

// MILK_CMAKE_MANDATE_CFITSIO

#include <ctype.h>
#include <fitsio.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "milk_runnable.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "tlm_fileset.h"

static char    p_dir[FUNCTION_PARAMETER_STRMAXLEN]   = ".";
static char    p_stream[FUNCTION_PARAMETER_STRMAXLEN] = "";
static char    p_out[FUNCTION_PARAMETER_STRMAXLEN]   = "@L:tlm";
static char    p_tstart[FUNCTION_PARAMETER_STRMAXLEN] = "";
static char    p_tend[FUNCTION_PARAMETER_STRMAXLEN]  = "";
static int64_t p_maxframes                           = 0;
static int64_t p_decim                               = 1;
static char    p_tsrc[FUNCTION_PARAMETER_STRMAXLEN]  = "auto";
static char    p_mask[FUNCTION_PARAMETER_STRMAXLEN]  = "";
static char    p_roi[FUNCTION_PARAMETER_STRMAXLEN]   = "";
static int64_t p_bin                                 = 1;
static int     p_packed                              = 1;
static int64_t p_nthreads                            = 0;

static int64_t o_nframes = 0;
static int64_t o_npix    = 0;
static int64_t o_nx      = 0;
static int64_t o_ny      = 0;

#define FPS_PARAMS_TLMLOAD(X)                                                                  \
    X(".dir", p_dir, FPTYPE_DIRNAME, 1, FPFLAG_DEFAULT_INPUT, "Telemetry dir or FITS file")    \
    X(".stream", p_stream, FPTYPE_STRING_NOT_STREAM, 1, FPFLAG_DEFAULT_INPUT,                  \
      "Stream prefix (auto if empty)")                                                         \
    X(".out", p_out, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "Output image name")          \
    X(".tstart", p_tstart, FPTYPE_STRING_NOT_STREAM, 0, FPFLAG_DEFAULT_INPUT,                  \
      "Start time (HH:MM:SS, +s, rel, unix)")                                                  \
    X(".tend", p_tend, FPTYPE_STRING_NOT_STREAM, 0, FPFLAG_DEFAULT_INPUT,                      \
      "End time (HH:MM:SS, +s, rel, unix)")                                                    \
    X(".maxframes", &p_maxframes, FPTYPE_INT64, 0, FPFLAG_DEFAULT_INPUT,                       \
      "Max frames to load (0=all)")                                                            \
    X(".decim", &p_decim, FPTYPE_INT64, 0, FPFLAG_DEFAULT_INPUT, "Temporal stride (>=1)")      \
    X(".tsrc", p_tsrc, FPTYPE_STRING_NOT_STREAM, 0, FPFLAG_DEFAULT_INPUT,                      \
      "Timestamp source: auto|acq|log")                                                        \
    X(".mask", p_mask, FPTYPE_STRING, 0, FPFLAG_DEFAULT_INPUT, "Optional 2D mask image/FITS")  \
    X(".roi", p_roi, FPTYPE_STRING_NOT_STREAM, 0, FPFLAG_DEFAULT_INPUT,                        \
      "Optional ROI x0:x1,y0:y1")                                                              \
    X(".bin", &p_bin, FPTYPE_INT64, 0, FPFLAG_DEFAULT_INPUT, "Spatial binning factor (>=1)")   \
    X(".packed", &p_packed, FPTYPE_ONOFF, 0, FPFLAG_DEFAULT_INPUT, "1=(P,N) 2D, 0=(nx,ny,N)")  \
    X(".nthreads", &p_nthreads, FPTYPE_INT64, 0, FPFLAG_DEFAULT_INPUT, "Reader threads (0=auto)") \
    X(".nframes", &o_nframes, FPTYPE_INT64, 0, FPFLAG_DEFAULT_OUTPUT, "Loaded frame count")    \
    X(".npix", &o_npix, FPTYPE_INT64, 0, FPFLAG_DEFAULT_OUTPUT, "Output pixels per frame")     \
    X(".nx", &o_nx, FPTYPE_INT64, 0, FPFLAG_DEFAULT_OUTPUT, "Output spatial width")            \
    X(".ny", &o_ny, FPTYPE_INT64, 0, FPFLAG_DEFAULT_OUTPUT, "Output spatial height")

MILK_PARAMS_DEFINE(mlt_tlmload, FPS_PARAMS_TLMLOAD);

typedef struct
{
    uint32_t     cube_idx;
    uint32_t     frame_idx;
    MILK_TLM_ROW row;
} mlt_sel_frame;

typedef struct
{
    uint32_t cube_idx;
    size_t   out_start;
    size_t   count;
    int      contiguous;
} mlt_read_task;

static void report_fits_err(const char *file, int status, const char *ctx)
{
    char errmsg[FLEN_ERRMSG];
    fits_get_errstatus(status, errmsg);
    fprintf(stderr, "ERROR: CFITSIO %s (%s): %s (status=%d)\n", ctx, file, errmsg, status);
}

static errno_t inspect_first_fits(const char *path, uint32_t *nx_out, uint32_t *ny_out)
{
    fitsfile *fptr   = NULL;
    int       status = 0;
    int       naxis  = 0;
    long      naxes[4] = { 1, 1, 1, 1 };

    fits_open_image(&fptr, path, READONLY, &status);
    if (status != 0)
    {
        report_fits_err(path, status, "open");
        return RETURN_FAILURE;
    }
    fits_get_img_dim(fptr, &naxis, &status);
    if (status == 0 && naxis >= 1)
    {
        fits_get_img_size(fptr, (naxis > 4) ? 4 : naxis, naxes, &status);
    }
    int cstat = 0;
    fits_close_file(fptr, &cstat);

    if (status != 0 || naxis < 2 || naxes[0] <= 0 || naxes[1] <= 0)
    {
        fprintf(stderr, "ERROR: invalid FITS dimensions in '%s' (naxis=%d)\n", path, naxis);
        return RETURN_FAILURE;
    }
    *nx_out = (uint32_t) naxes[0];
    *ny_out = (uint32_t) naxes[1];
    return RETURN_SUCCESS;
}

static errno_t parse_roi(const char *spec,
                         uint32_t    orig_nx,
                         uint32_t    orig_ny,
                         uint32_t   *x0_out,
                         uint32_t   *x1_out,
                         uint32_t   *y0_out,
                         uint32_t   *y1_out)
{
    if (spec == NULL || spec[0] == '\0' || strcmp(spec, "none") == 0)
    {
        *x0_out = 0;
        *x1_out = orig_nx - 1;
        *y0_out = 0;
        *y1_out = orig_ny - 1;
        return RETURN_SUCCESS;
    }
    long x0 = 0;
    long x1 = 0;
    long y0 = 0;
    long y1 = 0;
    if (sscanf(spec, "%ld:%ld,%ld:%ld", &x0, &x1, &y0, &y1) != 4 || x0 < 0 || y0 < 0 ||
        x1 < x0 || y1 < y0 || x1 >= (long) orig_nx || y1 >= (long) orig_ny)
    {
        fprintf(stderr, "ERROR: invalid ROI '%s' for %ux%u frame (expected x0:x1,y0:y1)\n", spec,
                orig_nx, orig_ny);
        return RETURN_FAILURE;
    }
    *x0_out = (uint32_t) x0;
    *x1_out = (uint32_t) x1;
    *y0_out = (uint32_t) y0;
    *y1_out = (uint32_t) y1;
    return RETURN_SUCCESS;
}

static errno_t load_mask_active_idx(const char *mask_arg,
                                    uint32_t    orig_nx,
                                    uint32_t    rx0,
                                    uint32_t    ry0,
                                    uint32_t    nx_bin,
                                    uint32_t    ny_bin,
                                    uint32_t    bin,
                                    uint32_t  **active_out,
                                    uint32_t   *p_out)
{
    *active_out = NULL;
    *p_out      = 0;
    if (mask_arg == NULL || mask_arg[0] == '\0' || strcmp(mask_arg, "none") == 0)
    {
        return RETURN_SUCCESS;
    }

    float   *mbuf   = NULL;
    uint32_t m_nx   = 0;
    uint32_t m_ny   = 0;
    int      owned  = 0;

    IMGID mimg = imgid_make_from_name(mask_arg);
    resolveIMGID(&mimg, ERRMODE_NULL, dcimg, dcnimg);
    if (mimg.ID != -1)
    {
        IMAGE *im = mimg.im;
        m_nx      = im->md[0].size[0];
        m_ny      = (im->md[0].naxis >= 2) ? im->md[0].size[1] : 1;
        size_t np = (size_t) m_nx * (size_t) m_ny;
        mbuf      = (float *) malloc(np * sizeof(float));
        if (mbuf == NULL)
        {
            imgid_free(&mimg);
            return RETURN_FAILURE;
        }
        owned = 1;
        for (size_t i = 0; i < np; i++)
        {
            switch (im->md[0].datatype)
            {
                case _DATATYPE_FLOAT:
                    mbuf[i] = im->array.F[i];
                    break;
                case _DATATYPE_DOUBLE:
                    mbuf[i] = (float) im->array.D[i];
                    break;
                case _DATATYPE_UINT8:
                    mbuf[i] = (float) im->array.UI8[i];
                    break;
                case _DATATYPE_INT32:
                    mbuf[i] = (float) im->array.SI32[i];
                    break;
                case _DATATYPE_UINT16:
                    mbuf[i] = (float) im->array.UI16[i];
                    break;
                default:
                    mbuf[i] = 0.0f;
                    break;
            }
        }
        imgid_free(&mimg);
    }
    else
    {
        imgid_free(&mimg);
        fitsfile *fptr   = NULL;
        int       status = 0;
        int       naxis  = 0;
        long      naxes[3] = { 1, 1, 1 };
        fits_open_image(&fptr, mask_arg, READONLY, &status);
        if (status != 0)
        {
            report_fits_err(mask_arg, status, "open mask");
            return RETURN_FAILURE;
        }
        fits_get_img_dim(fptr, &naxis, &status);
        fits_get_img_size(fptr, 3, naxes, &status);
        if (status != 0 || naxis < 2)
        {
            int cstat = 0;
            fits_close_file(fptr, &cstat);
            fprintf(stderr, "ERROR: mask FITS '%s' must be 2D\n", mask_arg);
            return RETURN_FAILURE;
        }
        m_nx      = (uint32_t) naxes[0];
        m_ny      = (uint32_t) naxes[1];
        size_t np = (size_t) m_nx * (size_t) m_ny;
        mbuf      = (float *) malloc(np * sizeof(float));
        if (mbuf == NULL)
        {
            int cstat = 0;
            fits_close_file(fptr, &cstat);
            return RETURN_FAILURE;
        }
        owned            = 1;
        long fpix[2]     = { 1, 1 };
        int  anynul      = 0;
        fits_read_pix(fptr, TFLOAT, fpix, (LONGLONG) np, NULL, mbuf, &anynul, &status);
        int cstat = 0;
        fits_close_file(fptr, &cstat);
        if (status != 0)
        {
            free(mbuf);
            report_fits_err(mask_arg, status, "read mask");
            return RETURN_FAILURE;
        }
    }

    uint32_t  ngrid = nx_bin * ny_bin;
    uint32_t *idx   = (uint32_t *) malloc((size_t) ngrid * sizeof(uint32_t));
    if (idx == NULL)
    {
        if (owned)
        {
            free(mbuf);
        }
        return RETURN_FAILURE;
    }

    uint32_t count = 0;
    if (m_nx == nx_bin && m_ny == ny_bin)
    {
        for (uint32_t p = 0; p < ngrid; p++)
        {
            if (mbuf[p] > 0.5f)
            {
                idx[count++] = p;
            }
        }
    }
    else if (m_nx == orig_nx && (uint32_t) m_ny >= ry0 + ny_bin * bin)
    {
        for (uint32_t by = 0; by < ny_bin; by++)
        {
            for (uint32_t bx = 0; bx < nx_bin; bx++)
            {
                int on = 0;
                for (uint32_t dy = 0; dy < bin && !on; dy++)
                {
                    uint32_t py = ry0 + by * bin + dy;
                    for (uint32_t dx = 0; dx < bin; dx++)
                    {
                        uint32_t px = rx0 + bx * bin + dx;
                        if (mbuf[(size_t) py * m_nx + px] > 0.5f)
                        {
                            on = 1;
                            break;
                        }
                    }
                }
                if (on)
                {
                    idx[count++] = by * nx_bin + bx;
                }
            }
        }
    }
    else
    {
        fprintf(stderr, "ERROR: mask dimensions %ux%u do not match output (%ux%u) or raw (%u)\n",
                m_nx, m_ny, nx_bin, ny_bin, orig_nx);
        free(idx);
        if (owned)
        {
            free(mbuf);
        }
        return RETURN_FAILURE;
    }

    if (owned)
    {
        free(mbuf);
    }
    if (count == 0)
    {
        fprintf(stderr, "ERROR: mask '%s' has zero active pixels (> 0.5)\n", mask_arg);
        free(idx);
        return RETURN_FAILURE;
    }
    *active_out = idx;
    *p_out      = count;
    return RETURN_SUCCESS;
}

static void transform_frame_to_out(const float    *src_roi,
                                   uint32_t        roi_w,
                                   uint32_t        nx_bin,
                                   uint32_t        ny_bin,
                                   uint32_t        bin,
                                   const uint32_t *active_idx,
                                   uint32_t        npix_out,
                                   float          *dst,
                                   float          *bin_tmp)
{
    if (bin == 1 && active_idx == NULL)
    {
        memcpy(dst, src_roi, (size_t) npix_out * sizeof(float));
        return;
    }

    const float *grid = src_roi;
    if (bin > 1)
    {
        float inv_b2 = 1.0f / (float) (bin * bin);
        for (uint32_t by = 0; by < ny_bin; by++)
        {
            for (uint32_t bx = 0; bx < nx_bin; bx++)
            {
                float sum = 0.0f;
                for (uint32_t dy = 0; dy < bin; dy++)
                {
                    const float *row = src_roi + (size_t) (by * bin + dy) * roi_w + bx * bin;
                    for (uint32_t dx = 0; dx < bin; dx++)
                    {
                        sum += row[dx];
                    }
                }
                bin_tmp[(size_t) by * nx_bin + bx] = sum * inv_b2;
            }
        }
        grid = bin_tmp;
    }

    if (active_idx != NULL)
    {
        for (uint32_t p = 0; p < npix_out; p++)
        {
            dst[p] = grid[active_idx[p]];
        }
    }
    else
    {
        memcpy(dst, grid, (size_t) npix_out * sizeof(float));
    }
}

static errno_t tlmload_run(void)
{
    MLT_FILESET fs;
    if (mlt_fileset_scan(p_dir, p_stream, p_tsrc, &fs) != RETURN_SUCCESS)
    {
        return RETURN_FAILURE;
    }

    int64_t tstart_ns = fs.t_first_ns;
    int64_t tend_ns   = fs.t_last_ns;
    if (mlt_parse_time_bound(p_tstart, fs.t_epoch_sec, fs.t_first_ns, fs.t_first_ns, &tstart_ns) !=
            RETURN_SUCCESS ||
        mlt_parse_time_bound(p_tend, fs.t_epoch_sec, tstart_ns, fs.t_last_ns, &tend_ns) !=
            RETURN_SUCCESS)
    {
        mlt_fileset_free(&fs);
        return RETURN_FAILURE;
    }
    if (tend_ns < tstart_ns)
    {
        fprintf(stderr, "ERROR: tend is earlier than tstart\n");
        mlt_fileset_free(&fs);
        return RETURN_FAILURE;
    }

    int64_t decim = (p_decim >= 1) ? p_decim : 1;
    int64_t bin   = (p_bin >= 1) ? p_bin : 1;

    size_t  nc      = fs.ncubes;
    int    *overlap = (int *) calloc(nc, sizeof(int));
    if (overlap == NULL)
    {
        mlt_fileset_free(&fs);
        return RETURN_FAILURE;
    }

    size_t n_ov = 0;
    for (size_t c = 0; c < nc; c++)
    {
        if (fs.entries[c].t_last_ns >= tstart_ns && fs.entries[c].t_first_ns <= tend_ns)
        {
            overlap[c] = 1;
            n_ov++;
        }
    }
    if (n_ov == 0)
    {
        free(overlap);
        mlt_fileset_free(&fs);
        fprintf(stderr, "ERROR: no telemetry cubes overlap requested [tstart, tend]\n");
        return RETURN_FAILURE;
    }

    MILK_TLM_TABLE *tables = (MILK_TLM_TABLE *) calloc(nc, sizeof(MILK_TLM_TABLE));
    int            *ok_arr = (int *) calloc(nc, sizeof(int));
    if (tables == NULL || ok_arr == NULL)
    {
        free(tables);
        free(ok_arr);
        free(overlap);
        mlt_fileset_free(&fs);
        return RETURN_FAILURE;
    }

    size_t ci;
#pragma omp parallel for schedule(dynamic, 2) if (n_ov > 4)
    for (ci = 0; ci < nc; ci++)
    {
        if (overlap[ci])
        {
            if (milk_tlm_read_file(fs.entries[ci].txt_path, &tables[ci]) == RETURN_SUCCESS)
            {
                ok_arr[ci] = 1;
            }
        }
    }

    size_t est_rows = 0;
    for (size_t c = 0; c < nc; c++)
    {
        if (!overlap[c])
        {
            continue;
        }
        if (!ok_arr[c])
        {
            fprintf(stderr, "ERROR: failed reading timing file '%s'\n", fs.entries[c].txt_path);
            for (size_t k = 0; k < nc; k++)
            {
                milk_tlm_free(&tables[k]);
            }
            free(tables);
            free(ok_arr);
            free(overlap);
            mlt_fileset_free(&fs);
            return RETURN_FAILURE;
        }
        est_rows += tables[c].nrows;
    }

    mlt_sel_frame *sel = (mlt_sel_frame *) malloc((est_rows + 1) * sizeof(mlt_sel_frame));
    if (sel == NULL)
    {
        for (size_t k = 0; k < nc; k++)
        {
            milk_tlm_free(&tables[k]);
        }
        free(tables);
        free(ok_arr);
        free(overlap);
        mlt_fileset_free(&fs);
        return RETURN_FAILURE;
    }

    size_t  nsel      = 0;
    int64_t match_idx = 0;
    int64_t ndropped  = 0;
    int64_t prev_cnt0 = 0;
    int     have_prev = 0;
    for (size_t c = 0; c < nc; c++)
    {
        if (!overlap[c])
        {
            continue;
        }
        const MILK_TLM_TABLE *tbl = &tables[c];
        for (size_t r = 0; r < tbl->nrows; r++)
        {
            const MILK_TLM_ROW *row = &tbl->rows[r];
            int64_t             tns = mlt_row_teff_ns(row, fs.use_acq);
            if (tns < tstart_ns || tns > tend_ns)
            {
                continue;
            }
            if (have_prev && (int64_t) row->cnt0 > prev_cnt0 + 1)
            {
                ndropped += (int64_t) row->cnt0 - prev_cnt0 - 1;
            }
            prev_cnt0 = (int64_t) row->cnt0;
            have_prev = 1;

            if ((match_idx % decim) == 0)
            {
                sel[nsel].cube_idx  = (uint32_t) c;
                sel[nsel].frame_idx =
                    (uint32_t) ((row->frame_idx >= 0) ? row->frame_idx : (int64_t) r);
                sel[nsel].row       = *row;
                nsel++;
                if (p_maxframes > 0 && (int64_t) nsel >= p_maxframes)
                {
                    break;
                }
            }
            match_idx++;
        }
        if (p_maxframes > 0 && (int64_t) nsel >= p_maxframes)
        {
            break;
        }
    }

    for (size_t k = 0; k < nc; k++)
    {
        milk_tlm_free(&tables[k]);
    }
    free(tables);
    free(ok_arr);
    free(overlap);

    if (nsel == 0)
    {
        free(sel);
        mlt_fileset_free(&fs);
        fprintf(stderr, "ERROR: 0 frames matched selection in [tstart, tend]\n");
        return RETURN_FAILURE;
    }

    uint32_t orig_nx = 0;
    uint32_t orig_ny = 0;
    if (inspect_first_fits(fs.entries[sel[0].cube_idx].fits_path, &orig_nx, &orig_ny) !=
        RETURN_SUCCESS)
    {
        free(sel);
        mlt_fileset_free(&fs);
        return RETURN_FAILURE;
    }

    uint32_t x0 = 0;
    uint32_t x1 = 0;
    uint32_t y0 = 0;
    uint32_t y1 = 0;
    if (parse_roi(p_roi, orig_nx, orig_ny, &x0, &x1, &y0, &y1) != RETURN_SUCCESS)
    {
        free(sel);
        mlt_fileset_free(&fs);
        return RETURN_FAILURE;
    }

    uint32_t roi_w_raw = x1 - x0 + 1;
    uint32_t roi_h_raw = y1 - y0 + 1;
    if ((uint32_t) bin > roi_w_raw || (uint32_t) bin > roi_h_raw)
    {
        fprintf(stderr, "ERROR: bin=%lld exceeds ROI size %ux%u\n", (long long) bin, roi_w_raw,
                roi_h_raw);
        free(sel);
        mlt_fileset_free(&fs);
        return RETURN_FAILURE;
    }

    uint32_t nx_bin = roi_w_raw / (uint32_t) bin;
    uint32_t ny_bin = roi_h_raw / (uint32_t) bin;
    uint32_t roi_w  = nx_bin * (uint32_t) bin;
    uint32_t roi_h  = ny_bin * (uint32_t) bin;
    uint32_t rx0    = x0;
    uint32_t rx1    = x0 + roi_w - 1;
    uint32_t ry0    = y0;
    uint32_t ry1    = y0 + roi_h - 1;

    uint32_t *active_idx = NULL;
    uint32_t  p_active   = 0;
    if (load_mask_active_idx(p_mask, orig_nx, rx0, ry0, nx_bin, ny_bin, (uint32_t) bin,
                             &active_idx, &p_active) != RETURN_SUCCESS)
    {
        free(sel);
        mlt_fileset_free(&fs);
        return RETURN_FAILURE;
    }

    int      masked    = (active_idx != NULL) ? 1 : 0;
    uint32_t npix_out  = masked ? p_active : (nx_bin * ny_bin);
    int      is_packed = (p_packed != 0 || masked != 0) ? 1 : 0;

    char out_t_name[FUNCTION_PARAMETER_STRMAXLEN + 4];
    snprintf(out_t_name, sizeof(out_t_name), "%s_t", p_out);

    IMGID probe_out = imgid_make_from_name(p_out);
    IMGID probe_t   = imgid_make_from_name(out_t_name);
    if (probe_out.name[0] != '\0')
    {
        delete_image_ID(probe_out.name, DELETE_IMAGE_ERRMODE_IGNORE);
    }
    if (probe_t.name[0] != '\0')
    {
        delete_image_ID(probe_t.name, DELETE_IMAGE_ERRMODE_IGNORE);
    }
    imgid_free(&probe_out);
    imgid_free(&probe_t);

    uint32_t dims_out[3] = { 1, 1, 1 };
    long     naxis_out   = 2;
    if (is_packed)
    {
        dims_out[0] = npix_out;
        dims_out[1] = (uint32_t) nsel;
        naxis_out   = 2;
    }
    else
    {
        dims_out[0] = nx_bin;
        dims_out[1] = ny_bin;
        dims_out[2] = (uint32_t) nsel;
        naxis_out   = 3;
    }

    imageID out_id = -1;
    if (create_image_ID(p_out, naxis_out, dims_out, _DATATYPE_FLOAT, 0, IMGID_NB_KEYWO_MAX, 0,
                        &out_id) != RETURN_SUCCESS ||
        out_id < 0)
    {
        free(active_idx);
        free(sel);
        mlt_fileset_free(&fs);
        return RETURN_FAILURE;
    }

    uint32_t dims_t[2] = { 6, (uint32_t) nsel };
    imageID  t_id      = -1;
    if (create_image_ID(out_t_name, 2, dims_t, _DATATYPE_DOUBLE, 0, IMGID_NB_KEYWO_MAX, 0, &t_id) !=
            RETURN_SUCCESS ||
        t_id < 0)
    {
        delete_image_ID(dcimg[out_id].md[0].name, DELETE_IMAGE_ERRMODE_IGNORE);
        free(active_idx);
        free(sel);
        mlt_fileset_free(&fs);
        return RETURN_FAILURE;
    }

    if (fps_cli_is_standalone() && dcimg[out_id].md[0].shared == 0)
    {
        fprintf(stderr,
                "tlmload: '%s' would be lost when this command exits: name it @S:%s\n",
                p_out, dcimg[out_id].md[0].name);
        delete_image_ID(dcimg[t_id].md[0].name, DELETE_IMAGE_ERRMODE_IGNORE);
        delete_image_ID(dcimg[out_id].md[0].name, DELETE_IMAGE_ERRMODE_IGNORE);
        free(active_idx);
        free(sel);
        mlt_fileset_free(&fs);
        return RETURN_FAILURE;
    }

    double *tarr     = dcimg[t_id].array.D;
    int64_t epoch_ns = fs.t_epoch_sec * 1000000000LL;
    for (size_t i = 0; i < nsel; i++)
    {
        const MILK_TLM_ROW *r = &sel[i].row;
        size_t              b = i * 6;
        tarr[b + 0]           = (double) (mlt_row_teff_ns(r, fs.use_acq) - epoch_ns) * 1e-9;
        tarr[b + 1]           = (double) (r->t_log_ns - epoch_ns) * 1e-9;
        tarr[b + 2]           = (r->t_acq_ns > 0) ? (double) (r->t_acq_ns - epoch_ns) * 1e-9 : 0.0;
        tarr[b + 3]           = (double) r->cnt0;
        tarr[b + 4]           = (double) r->cnt1;
        tarr[b + 5]           = (double) r->main_idx;
    }

    /* Build parallel chunk tasks over selected frames. */
    size_t         max_chunk = 512;
    mlt_read_task *tasks     = (mlt_read_task *) malloc((nsel + 1) * sizeof(mlt_read_task));
    if (tasks == NULL)
    {
        free(active_idx);
        free(sel);
        mlt_fileset_free(&fs);
        return RETURN_FAILURE;
    }
    size_t ntasks = 0;
    size_t idx    = 0;
    while (idx < nsel)
    {
        uint32_t c0     = sel[idx].cube_idx;
        int      contig = 1;
        size_t   len    = 1;
        while (idx + len < nsel && len < max_chunk && sel[idx + len].cube_idx == c0)
        {
            if (sel[idx + len].frame_idx != sel[idx + len - 1].frame_idx + 1)
            {
                contig = 0;
            }
            len++;
        }
        /* Re-split mixed contiguous/non-contiguous cube runs cleanly. */
        if (!contig)
        {
            len    = 1;
            contig = 0;
            while (idx + len < nsel && len < max_chunk && sel[idx + len].cube_idx == c0)
            {
                len++;
            }
        }
        tasks[ntasks].cube_idx   = c0;
        tasks[ntasks].out_start  = idx;
        tasks[ntasks].count      = len;
        tasks[ntasks].contiguous = contig;
        ntasks++;
        idx += len;
    }

    float *out_buf = dcimg[out_id].array.F;
    int    reent   = fits_is_reentrant();
    int    nth     = (p_nthreads > 0) ? (int) p_nthreads : 16;
    if (nth > 32)
    {
        nth = 32;
    }
    if (!reent)
    {
        nth = 1;
    }

    int err_flag = 0;
    size_t ti;
#pragma omp parallel for schedule(dynamic, 1) num_threads(nth) if (nth > 1 && ntasks > 1) \
    reduction(| : err_flag)
    for (ti = 0; ti < ntasks; ti++)
    {
        const mlt_read_task *tk        = &tasks[ti];
        const char          *fits_path = fs.entries[tk->cube_idx].fits_path;
        fitsfile            *fptr      = NULL;
        int                  status    = 0;
        int                  naxis     = 0;
        long                 naxes[4]  = { 1, 1, 1, 1 };

        fits_open_image(&fptr, fits_path, READONLY, &status);
        if (status != 0)
        {
            report_fits_err(fits_path, status, "open");
            err_flag = 1;
            continue;
        }
        fits_get_img_dim(fptr, &naxis, &status);
        fits_get_img_size(fptr, (naxis > 4) ? 4 : naxis, naxes, &status);
        if (status != 0 || naxis < 2 || (uint32_t) naxes[0] != orig_nx ||
            (uint32_t) naxes[1] != orig_ny)
        {
            fprintf(stderr, "ERROR: FITS shape mismatch in '%s' (got %ldx%ld, expected %ux%u)\n",
                    fits_path, naxes[0], naxes[1], orig_nx, orig_ny);
            int cstat = 0;
            fits_close_file(fptr, &cstat);
            err_flag = 1;
            continue;
        }

        uint32_t cube_nz = (naxis >= 3) ? (uint32_t) naxes[2] : 1;
        size_t   roi_np  = (size_t) roi_w * (size_t) roi_h;
        float   *bin_tmp = (bin > 1) ? (float *) malloc((size_t) nx_bin * ny_bin * sizeof(float))
                                     : NULL;

        if (tk->contiguous && sel[tk->out_start].frame_idx + tk->count <= cube_nz)
        {
            long fpix[3] = { (long) rx0 + 1, (long) ry0 + 1,
                             (naxis >= 3) ? (long) sel[tk->out_start].frame_idx + 1 : 1 };
            long lpix[3] = { (long) rx1 + 1, (long) ry1 + 1,
                             (naxis >= 3)
                                 ? (long) (sel[tk->out_start].frame_idx + tk->count - 1) + 1
                                 : 1 };
            long inc[3]  = { 1, 1, 1 };
            int  anynul  = 0;

            if (bin == 1 && !masked)
            {
                float *dst_base = out_buf + tk->out_start * (size_t) npix_out;
                fits_read_subset(fptr, TFLOAT, fpix, lpix, inc, NULL, dst_base, &anynul, &status);
                if (status != 0)
                {
                    report_fits_err(fits_path, status, "read_subset direct");
                    err_flag = 1;
                }
            }
            else
            {
                float *slab = (float *) malloc(roi_np * tk->count * sizeof(float));
                if (slab == NULL)
                {
                    err_flag = 1;
                }
                else
                {
                    fits_read_subset(fptr, TFLOAT, fpix, lpix, inc, NULL, slab, &anynul, &status);
                    if (status != 0)
                    {
                        report_fits_err(fits_path, status, "read_subset slab");
                        err_flag = 1;
                    }
                    else
                    {
                        for (size_t k = 0; k < tk->count; k++)
                        {
                            float *dst = out_buf + (tk->out_start + k) * (size_t) npix_out;
                            transform_frame_to_out(slab + k * roi_np, roi_w, nx_bin, ny_bin,
                                                   (uint32_t) bin, active_idx, npix_out, dst,
                                                   bin_tmp);
                        }
                    }
                    free(slab);
                }
            }
        }
        else
        {
            float *fbuf = (float *) malloc(roi_np * sizeof(float));
            if (fbuf == NULL)
            {
                err_flag = 1;
            }
            else
            {
                for (size_t k = 0; k < tk->count; k++)
                {
                    uint32_t fidx = sel[tk->out_start + k].frame_idx;
                    if (fidx >= cube_nz)
                    {
                        fidx = cube_nz - 1;
                    }
                    long fpix[3] = { (long) rx0 + 1, (long) ry0 + 1,
                                     (naxis >= 3) ? (long) fidx + 1 : 1 };
                    long lpix[3] = { (long) rx1 + 1, (long) ry1 + 1,
                                     (naxis >= 3) ? (long) fidx + 1 : 1 };
                    long inc[3]  = { 1, 1, 1 };
                    int  anynul  = 0;
                    fits_read_subset(fptr, TFLOAT, fpix, lpix, inc, NULL, fbuf, &anynul, &status);
                    if (status != 0)
                    {
                        report_fits_err(fits_path, status, "read_subset frame");
                        err_flag = 1;
                        break;
                    }
                    float *dst = out_buf + (tk->out_start + k) * (size_t) npix_out;
                    transform_frame_to_out(fbuf, roi_w, nx_bin, ny_bin, (uint32_t) bin, active_idx,
                                           npix_out, dst, bin_tmp);
                }
                free(fbuf);
            }
        }

        free(bin_tmp);
        int cstat = 0;
        fits_close_file(fptr, &cstat);
    }

    free(tasks);
    free(active_idx);
    free(sel);

    if (err_flag)
    {
        delete_image_ID(dcimg[t_id].md[0].name, DELETE_IMAGE_ERRMODE_IGNORE);
        delete_image_ID(dcimg[out_id].md[0].name, DELETE_IMAGE_ERRMODE_IGNORE);
        mlt_fileset_free(&fs);
        return RETURN_FAILURE;
    }

    const char *out_bare = dcimg[out_id].md[0].name;
    const char *t_bare   = dcimg[t_id].md[0].name;
    const char *tsrc_str = fs.use_acq ? "acq" : "log";

    const char *names_kw[2] = { out_bare, t_bare };
    for (int i = 0; i < 2; i++)
    {
        image_write_keyword_L(names_kw[i], "ORIGX", (long) orig_nx, "Raw frame width");
        image_write_keyword_L(names_kw[i], "ORIGY", (long) orig_ny, "Raw frame height");
        image_write_keyword_L(names_kw[i], "ROIX0", (long) rx0, "ROI x0 (0-based)");
        image_write_keyword_L(names_kw[i], "ROIY0", (long) ry0, "ROI y0 (0-based)");
        image_write_keyword_L(names_kw[i], "BINFCT", (long) bin, "Spatial bin factor");
        image_write_keyword_L(names_kw[i], "PACKED", (long) is_packed, "1=(P,N) 2D, 0=(nx,ny,N)");
        image_write_keyword_L(names_kw[i], "MASKED", (long) masked, "1 if pupil mask applied");
        image_write_keyword_L(names_kw[i], "DROPPED", (long) ndropped, "Dropped cnt0 frames");
        image_write_keyword_L(names_kw[i], "MLTEPOCH", (long) fs.t_epoch_sec,
                              "Day 00:00:00 UT [s]");
        image_write_keyword_S(names_kw[i], "MLTTSRC", tsrc_str, "Timestamp source");
        image_write_keyword_L(names_kw[i], "NFRAMES", (long) nsel, "Frame count");
        image_write_keyword_L(names_kw[i], "NPIX", (long) npix_out, "Pixels per frame");
        image_write_keyword_L(names_kw[i], "NXOUT", (long) nx_bin, "Spatial width post-ROI/bin");
        image_write_keyword_L(names_kw[i], "NYOUT", (long) ny_bin, "Spatial height post-ROI/bin");
    }

    o_nframes = (int64_t) nsel;
    o_npix    = (int64_t) npix_out;
    o_nx      = (int64_t) nx_bin;
    o_ny      = (int64_t) ny_bin;

    if (!dcquiet)
    {
        printf("mlt.tlmload: loaded %zu frames (%u pix/frame, %ux%u, tsrc=%s) -> %s, %s\n",
               nsel, npix_out, nx_bin, ny_bin, tsrc_str, out_bare, t_bare);
    }

    mlt_fileset_free(&fs);
    return RETURN_SUCCESS;
}

static MILK_HOT errno_t tlmload_compute(const MILK_RUNNABLE *self)
{
    errno_t ret = RETURN_SUCCESS;
    DEBUG_TRACE_FSTART();
    MILK_PROCINFO_INIT(self)
    MILK_PROCINFO_LOOPSTART(self)

    ret = tlmload_run();
    if (ret == RETURN_SUCCESS && dcfpsptr != NULL)
    {
        fpsresync_modvar_to_fps(dcfpsptr, dcfpsbindings, dcfpsnbindings);
    }

    MILK_PROCINFO_END(self)
    DEBUG_TRACE_FEXIT();
    return ret;
}

const MILK_RUNNABLE runnable_mlt_tlmload = {
    .kind = MILK_RUNNABLE_FPS,
    .app  = { .fps_name         = "mlttlmload",
              .cmdkey           = "tlmload",
              .description      = "load telemetry FITS cubes + timing metadata into memory",
              .description_long = "Load .fits/.fits.fz cubes and companion .txt timing files in "
                                  "parallel with time slicing, ROI/bin/mask extraction, and 6xN "
                                  "float64 timing metadata (<out>_t)." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(mlt_tlmload),
    .compute = tlmload_compute,
};

EXPORT_FPSEXEC(milk, "mlt-tlmload", runnable_mlt_tlmload);
