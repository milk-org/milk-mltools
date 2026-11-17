// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    tlm_save_unit.c
 * @brief   Save in-memory telemetry matrix/cube + timing metadata as FITS + `.txt` pair
 */

// MILK_CMAKE_MANDATE_CFITSIO

#include <errno.h>
#include <fitsio.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#include "milk_runnable.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "COREMOD_memory/tlm_timing.h"
#include "tlm_fileset.h"

static char p_in[FUNCTION_PARAMETER_STRMAXLEN]     = "";
static char p_in_t[FUNCTION_PARAMETER_STRMAXLEN]   = "";
static char p_dir[FUNCTION_PARAMETER_STRMAXLEN]    = ".";
static char p_stream[FUNCTION_PARAMETER_STRMAXLEN] = "telemetry";

#define FPS_PARAMS_TLMSAVE(X)                                                                   \
    X(".in", p_in, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "Input image (P x N or nx,ny,N)") \
    X(".in_t", p_in_t, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "Input timing (<in>_t) or .txt") \
    X(".dir", p_dir, FPTYPE_DIRNAME, 1, FPFLAG_DEFAULT_INPUT, "Output directory")               \
    X(".stream", p_stream, FPTYPE_STRING_NOT_STREAM, 1, FPFLAG_DEFAULT_INPUT,                   \
      "Output stream prefix")

MILK_PARAMS_DEFINE(mlt_tlmsave, FPS_PARAMS_TLMSAVE);

static void format_ut_iso(int64_t ns, char *hms_buf, size_t hms_cap, char *iso_buf, size_t iso_cap)
{
    time_t    sec  = (time_t) (ns / 1000000000LL);
    long long nsec = (long long) (ns % 1000000000LL);
    if (nsec < 0)
    {
        sec -= 1;
        nsec += 1000000000LL;
    }
    struct tm tm_utc;
    gmtime_r(&sec, &tm_utc);
    snprintf(hms_buf, hms_cap, "%02d:%02d:%02d.%09lld", tm_utc.tm_hour, tm_utc.tm_min,
             tm_utc.tm_sec, nsec);
    snprintf(iso_buf, iso_cap, "%04d-%02d-%02dT%02d:%02d:%02d.%09lld", tm_utc.tm_year + 1900,
             tm_utc.tm_mon + 1, tm_utc.tm_mday, tm_utc.tm_hour, tm_utc.tm_min, tm_utc.tm_sec,
             nsec);
}

static errno_t mkdir_if_needed(const char *dir)
{
    if (mkdir(dir, 0775) != 0 && errno != EEXIST)
    {
        fprintf(stderr, "ERROR: cannot create directory '%s': %s\n", dir, strerror(errno));
        return RETURN_FAILURE;
    }
    return RETURN_SUCCESS;
}

static errno_t tlmsave_run(void)
{
    if (p_stream[0] == '\0')
    {
        fprintf(stderr, "ERROR: .stream must be non-empty\n");
        return RETURN_FAILURE;
    }
    if (mkdir_if_needed(p_dir) != RETURN_SUCCESS)
    {
        return RETURN_FAILURE;
    }

    IMGID img = imgid_make_from_name(p_in);
    resolveIMGID(&img, ERRMODE_NULL, dcimg, dcnimg);
    if (img.ID == -1)
    {
        read_sharedmem_image(img.name, dcimg, dcnimg);
        resolveIMGID(&img, ERRMODE_NULL, dcimg, dcnimg);
    }
    if (img.ID == -1)
    {
        fprintf(stderr, "ERROR: cannot resolve input image '%s'\n", p_in);
        imgid_free(&img);
        return RETURN_FAILURE;
    }

    IMAGE   *im    = img.im;
    uint8_t  naxis = im->md[0].naxis;
    long     nx    = 1;
    long     ny    = 1;
    size_t   nz    = 1;

    if (naxis == 2)
    {
        nx = (long) im->md[0].size[0];
        ny = 1;
        nz = im->md[0].size[1];
    }
    else if (naxis == 3)
    {
        nx = (long) im->md[0].size[0];
        ny = (long) im->md[0].size[1];
        nz = im->md[0].size[2];
    }
    else
    {
        fprintf(stderr, "ERROR: tlmsave expects 2D (P, N) or 3D (nx, ny, N) image\n");
        imgid_free(&img);
        return RETURN_FAILURE;
    }

    MILK_TLM_ROW *rows  = NULL;
    size_t        nrows = 0;
    int64_t       ep    = 0;
    if (mlt_load_timing_arg(p_in_t, NULL, &rows, &nrows, &ep) != RETURN_SUCCESS)
    {
        imgid_free(&img);
        return RETURN_FAILURE;
    }
    if (nrows != nz)
    {
        fprintf(stderr, "ERROR: frame count mismatch between '%s' (%zu) and '%s' (%zu)\n", p_in,
                nz, p_in_t, nrows);
        free(rows);
        imgid_free(&img);
        return RETURN_FAILURE;
    }

    for (size_t i = 0; i < nrows; i++)
    {
        rows[i].frame_idx = (int64_t) i;
    }

    int64_t first_ns = mlt_row_teff_ns(&rows[0], (rows[0].t_acq_ns > 0));
    int64_t last_ns  = mlt_row_teff_ns(&rows[nrows - 1], (rows[nrows - 1].t_acq_ns > 0));

    char hms[64];
    char ut_str[64];
    char ut_end[64];
    char hms_dummy[64];
    format_ut_iso(first_ns, hms, sizeof(hms), ut_str, sizeof(ut_str));
    format_ut_iso(last_ns, hms_dummy, sizeof(hms_dummy), ut_end, sizeof(ut_end));

    char fits_path[MLT_PATH_MAX];
    char fits_bang[MLT_PATH_MAX + 2];
    char txt_path[MLT_PATH_MAX];
    int  nf = snprintf(fits_path, sizeof(fits_path), "%s/%s_%s.fits", p_dir, p_stream, hms);
    int  nt = snprintf(txt_path, sizeof(txt_path), "%s/%s_%s.txt", p_dir, p_stream, hms);
    if (nf < 0 || (size_t) nf >= sizeof(fits_path) || nt < 0 || (size_t) nt >= sizeof(txt_path))
    {
        free(rows);
        imgid_free(&img);
        return RETURN_FAILURE;
    }
    snprintf(fits_bang, sizeof(fits_bang), "!%s", fits_path);

    size_t ntot = (size_t) nx * (size_t) ny * nz;
    float *fbuf = NULL;
    float *fptr_data = NULL;
    if (im->md[0].datatype == _DATATYPE_FLOAT)
    {
        fptr_data = im->array.F;
    }
    else
    {
        fbuf = (float *) malloc(ntot * sizeof(float));
        if (fbuf == NULL)
        {
            free(rows);
            imgid_free(&img);
            return RETURN_FAILURE;
        }
        for (size_t i = 0; i < ntot; i++)
        {
            switch (im->md[0].datatype)
            {
                case _DATATYPE_DOUBLE:
                    fbuf[i] = (float) im->array.D[i];
                    break;
                case _DATATYPE_UINT16:
                    fbuf[i] = (float) im->array.UI16[i];
                    break;
                case _DATATYPE_INT32:
                    fbuf[i] = (float) im->array.SI32[i];
                    break;
                default:
                    fbuf[i] = 0.0f;
                    break;
            }
        }
        fptr_data = fbuf;
    }

    fitsfile *fptr    = NULL;
    int       status  = 0;
    long      naxes[3] = { nx, ny, (long) nz };
    fits_create_file(&fptr, fits_bang, &status);
    fits_create_img(fptr, FLOAT_IMG, 3, naxes, &status);

    long ep_val = (long) ep;
    if (ep_val <= 0)
    {
        ep_val = (long) ((first_ns / 1000000000LL / 86400LL) * 86400LL);
    }
    double mjd_str = 40587.0 + (double) first_ns / (86400.0 * 1e9);
    double mjd_end = 40587.0 + (double) last_ns / (86400.0 * 1e9);

    fits_update_key(fptr, TSTRING, "UT-STR", ut_str, "UT start timestamp", &status);
    fits_update_key(fptr, TSTRING, "UT-END", ut_end, "UT end timestamp", &status);
    fits_update_key(fptr, TDOUBLE, "MJD-STR", &mjd_str, "MJD start", &status);
    fits_update_key(fptr, TDOUBLE, "MJD-END", &mjd_end, "MJD end", &status);
    fits_update_key(fptr, TLONG, "MLTEPOCH", &ep_val, "Reference day 00:00:00 UT [s]", &status);

    long fpixel[3] = { 1, 1, 1 };
    fits_write_pix(fptr, TFLOAT, fpixel, (LONGLONG) ntot, fptr_data, &status);

    int cstat = 0;
    fits_close_file(fptr, &cstat);
    free(fbuf);

    if (status != 0 || cstat != 0)
    {
        char errmsg[FLEN_ERRMSG];
        fits_get_errstatus(status != 0 ? status : cstat, errmsg);
        fprintf(stderr, "ERROR: CFITSIO write failed for '%s': %s\n", fits_path, errmsg);
        free(rows);
        imgid_free(&img);
        return RETURN_FAILURE;
    }

    if (milk_tlm_write_file(txt_path, rows, nrows) != RETURN_SUCCESS)
    {
        free(rows);
        imgid_free(&img);
        return RETURN_FAILURE;
    }

    if (!dcquiet)
    {
        printf("mlt.tlmsave: wrote %zu frames -> %s + %s\n", nrows, fits_path, txt_path);
    }

    free(rows);
    imgid_free(&img);
    return RETURN_SUCCESS;
}

static MILK_HOT errno_t tlmsave_compute(const MILK_RUNNABLE *self)
{
    errno_t ret = RETURN_SUCCESS;
    DEBUG_TRACE_FSTART();
    MILK_PROCINFO_INIT(self)
    MILK_PROCINFO_LOOPSTART(self)
    ret = tlmsave_run();
    MILK_PROCINFO_END(self)
    DEBUG_TRACE_FEXIT();
    return ret;
}

const MILK_RUNNABLE runnable_mlt_tlmsave = {
    .kind = MILK_RUNNABLE_FPS,
    .app  = { .fps_name         = "mlttlmsave",
              .cmdkey           = "tlmsave",
              .description      = "save telemetry image + timing metadata as FITS + .txt",
              .description_long = "Write an in-memory 2D (P, N) or 3D (nx, ny, N) telemetry image "
                                  "together with its timing metadata to a paired .fits cube and "
                                  "7-column logshmim ASCII .txt file." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(mlt_tlmsave),
    .compute = tlmsave_compute,
};

EXPORT_FPSEXEC(milk, "mlt-tlmsave", runnable_mlt_tlmsave);
