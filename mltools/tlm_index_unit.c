// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    tlm_index_unit.c
 * @brief   Index telemetry `.txt` + `.fits[.fz]` cubes and report timing health
 */

#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "milk_runnable.h"
#include "COREMOD_memory/tlm_timing.h"
#include "tlm_fileset.h"

static char p_dir[FUNCTION_PARAMETER_STRMAXLEN]       = ".";
static char p_stream[FUNCTION_PARAMETER_STRMAXLEN]    = "";
static char p_outreport[FUNCTION_PARAMETER_STRMAXLEN] = "";

static int64_t o_ncubes    = 0;
static int64_t o_nframes   = 0;
static int64_t o_norphans  = 0;
static double  o_fps       = 0.0;
static double  o_jitter_us = 0.0;
static int64_t o_ndropped  = 0;
static int64_t o_has_acq   = 0;
static double  o_tstart    = 0.0;
static double  o_tend      = 0.0;
static int64_t o_epoch_sec = 0;

#define FPS_PARAMS_TLMINDEX(X)                                                                \
    X(".dir", p_dir, FPTYPE_DIRNAME, 1, FPFLAG_DEFAULT_INPUT, "Telemetry directory")          \
    X(".stream", p_stream, FPTYPE_STRING_NOT_STREAM, 1, FPFLAG_DEFAULT_INPUT,                 \
      "Stream prefix (auto if empty)")                                                        \
    X(".outreport", p_outreport, FPTYPE_FILENAME, 0, FPFLAG_DEFAULT_INPUT,                    \
      "Optional key=value report path")                                                       \
    X(".ncubes", &o_ncubes, FPTYPE_INT64, 0, FPFLAG_DEFAULT_OUTPUT, "Paired FITS+txt cubes") \
    X(".nframes", &o_nframes, FPTYPE_INT64, 0, FPFLAG_DEFAULT_OUTPUT, "Total frames")         \
    X(".norphans", &o_norphans, FPTYPE_INT64, 0, FPFLAG_DEFAULT_OUTPUT, "Orphan files")       \
    X(".fps", &o_fps, FPTYPE_FLOAT64, 0, FPFLAG_DEFAULT_OUTPUT, "Median frame rate [Hz]")     \
    X(".jitter_us", &o_jitter_us, FPTYPE_FLOAT64, 0, FPFLAG_DEFAULT_OUTPUT,                   \
      "MAD frame dt jitter [us]")                                                             \
    X(".ndropped", &o_ndropped, FPTYPE_INT64, 0, FPFLAG_DEFAULT_OUTPUT, "Dropped frames")     \
    X(".has_acq", &o_has_acq, FPTYPE_INT64, 0, FPFLAG_DEFAULT_OUTPUT, "t_acq available 0/1")  \
    X(".tstart_rel", &o_tstart, FPTYPE_FLOAT64, 0, FPFLAG_DEFAULT_OUTPUT,                     \
      "Start time [s] from UT day 0")                                                         \
    X(".tend_rel", &o_tend, FPTYPE_FLOAT64, 0, FPFLAG_DEFAULT_OUTPUT,                         \
      "End time [s] from UT day 0")                                                           \
    X(".epoch_sec", &o_epoch_sec, FPTYPE_INT64, 0, FPFLAG_DEFAULT_OUTPUT, "Day 00:00:00 UT")

MILK_PARAMS_DEFINE(mlt_tlmindex, FPS_PARAMS_TLMINDEX);

static int cmp_int64(const void *a, const void *b)
{
    int64_t va = *(const int64_t *) a;
    int64_t vb = *(const int64_t *) b;
    return (va > vb) - (va < vb);
}

static errno_t tlmindex_run(const MLT_FILESET *fs)
{
    size_t          nc     = fs->ncubes;
    MILK_TLM_TABLE *tables = (MILK_TLM_TABLE *) calloc(nc, sizeof(MILK_TLM_TABLE));
    int            *ok_arr = (int *) calloc(nc, sizeof(int));
    if (tables == NULL || ok_arr == NULL)
    {
        free(tables);
        free(ok_arr);
        return RETURN_FAILURE;
    }

    size_t c;
#pragma omp parallel for schedule(dynamic, 4) if (nc > 8)
    for (c = 0; c < nc; c++)
    {
        if (milk_tlm_read_file(fs->entries[c].txt_path, &tables[c]) == RETURN_SUCCESS)
        {
            ok_arr[c] = 1;
        }
    }

    int64_t  total_frames = 0;
    int64_t  dropped      = 0;
    int64_t  prev_cnt0    = 0;
    int      have_prev    = 0;
    uint64_t stride       = 1;

    for (c = 0; c < nc; c++)
    {
        if (!ok_arr[c])
        {
            fprintf(stderr, "ERROR: failed to read timing file '%s'\n", fs->entries[c].txt_path);
            for (size_t k = 0; k < nc; k++)
            {
                milk_tlm_free(&tables[k]);
            }
            free(tables);
            free(ok_arr);
            return RETURN_FAILURE;
        }
        total_frames += (int64_t) tables[c].nrows;
    }

    if (total_frames > 250000)
    {
        stride = (uint64_t) (total_frames / 250000);
    }

    size_t   dt_cap = (size_t) ((total_frames / (int64_t) stride) + (int64_t) nc + 16);
    int64_t *dts    = (int64_t *) malloc(dt_cap * sizeof(int64_t));
    size_t   ndt    = 0;
    if (dts == NULL)
    {
        for (c = 0; c < nc; c++)
        {
            milk_tlm_free(&tables[c]);
        }
        free(tables);
        free(ok_arr);
        return RETURN_FAILURE;
    }

    int64_t  prev_tns = 0;
    uint64_t sample_k = 0;
    for (c = 0; c < nc; c++)
    {
        const MILK_TLM_TABLE *tbl = &tables[c];
        for (size_t r = 0; r < tbl->nrows; r++)
        {
            int64_t tns  = mlt_row_teff_ns(&tbl->rows[r], fs->use_acq);
            int64_t cnt0 = (int64_t) tbl->rows[r].cnt0;
            if (have_prev)
            {
                if (cnt0 > prev_cnt0 + 1)
                {
                    dropped += (cnt0 - prev_cnt0 - 1);
                }
                int64_t dt = tns - prev_tns;
                if (dt > 0 && (sample_k % stride) == 0 && ndt < dt_cap)
                {
                    dts[ndt++] = dt;
                }
                sample_k++;
            }
            prev_tns  = tns;
            prev_cnt0 = cnt0;
            have_prev = 1;
        }
    }

    double fps_est   = 0.0;
    double jitter_us = 0.0;
    if (ndt > 0)
    {
        qsort(dts, ndt, sizeof(int64_t), cmp_int64);
        int64_t med_dt = dts[ndt / 2];
        if (med_dt > 0)
        {
            fps_est = 1e9 / (double) med_dt;
        }
        int64_t *absdev = (int64_t *) malloc(ndt * sizeof(int64_t));
        if (absdev != NULL)
        {
            for (size_t k = 0; k < ndt; k++)
            {
                int64_t d = dts[k] - med_dt;
                absdev[k] = (d >= 0) ? d : -d;
            }
            qsort(absdev, ndt, sizeof(int64_t), cmp_int64);
            jitter_us = 1.4826 * (double) absdev[ndt / 2] * 1e-3;
            free(absdev);
        }
    }

    free(dts);
    for (c = 0; c < nc; c++)
    {
        milk_tlm_free(&tables[c]);
    }
    free(tables);
    free(ok_arr);

    int64_t epoch_ns = fs->t_epoch_sec * 1000000000LL;
    o_ncubes         = (int64_t) nc;
    o_nframes        = total_frames;
    o_norphans       = (int64_t) (fs->norphan_txt + fs->norphan_fits);
    o_fps            = fps_est;
    o_jitter_us      = jitter_us;
    o_ndropped       = dropped;
    o_has_acq        = fs->has_acq;
    o_tstart         = (double) (fs->t_first_ns - epoch_ns) * 1e-9;
    o_tend           = (double) (fs->t_last_ns - epoch_ns) * 1e-9;
    o_epoch_sec      = fs->t_epoch_sec;

    if (p_outreport[0] != '\0' && strcmp(p_outreport, "none") != 0)
    {
        FILE *fp = fopen(p_outreport, "w");
        if (fp == NULL)
        {
            fprintf(stderr, "ERROR: cannot write report '%s'\n", p_outreport);
            return RETURN_FAILURE;
        }
        fprintf(fp, "stream=%s\n", fs->stream);
        fprintf(fp, "ncubes=%lld\n", (long long) o_ncubes);
        fprintf(fp, "nframes=%lld\n", (long long) o_nframes);
        fprintf(fp, "norphan_txt=%zu\n", fs->norphan_txt);
        fprintf(fp, "norphan_fits=%zu\n", fs->norphan_fits);
        fprintf(fp, "norphans=%lld\n", (long long) o_norphans);
        fprintf(fp, "fps=%.6f\n", o_fps);
        fprintf(fp, "jitter_us=%.3f\n", o_jitter_us);
        fprintf(fp, "ndropped=%lld\n", (long long) o_ndropped);
        fprintf(fp, "has_acq=%lld\n", (long long) o_has_acq);
        fprintf(fp, "epoch_sec=%lld\n", (long long) o_epoch_sec);
        fprintf(fp, "tstart_rel=%.9f\n", o_tstart);
        fprintf(fp, "tend_rel=%.9f\n", o_tend);
        fclose(fp);
    }

    if (!dcquiet)
    {
        printf("mlt.tlmindex: stream=%s cubes=%lld frames=%lld orphans=%lld "
               "fps=%.3f jitter=%.2fus dropped=%lld has_acq=%lld span=%.3fs\n",
               fs->stream, (long long) o_ncubes, (long long) o_nframes, (long long) o_norphans,
               o_fps, o_jitter_us, (long long) o_ndropped, (long long) o_has_acq,
               o_tend - o_tstart);
    }

    return RETURN_SUCCESS;
}

static MILK_HOT errno_t tlmindex_compute(const MILK_RUNNABLE *self)
{
    errno_t ret = RETURN_SUCCESS;
    DEBUG_TRACE_FSTART();
    MILK_PROCINFO_INIT(self)
    MILK_PROCINFO_LOOPSTART(self)

    MLT_FILESET fs;
    if (mlt_fileset_scan(p_dir, p_stream, "auto", &fs) != RETURN_SUCCESS)
    {
        ret = RETURN_FAILURE;
    }
    else
    {
        ret = tlmindex_run(&fs);
        mlt_fileset_free(&fs);
    }

    if (ret == RETURN_SUCCESS && dcfpsptr != NULL)
    {
        fpsresync_modvar_to_fps(dcfpsptr, dcfpsbindings, dcfpsnbindings);
    }

    MILK_PROCINFO_END(self)
    DEBUG_TRACE_FEXIT();
    return ret;
}

const MILK_RUNNABLE runnable_mlt_tlmindex = {
    .kind = MILK_RUNNABLE_FPS,
    .app  = { .fps_name         = "mlttlmindex",
              .cmdkey           = "tlmindex",
              .description      = "index telemetry cubes and check timing health",
              .description_long = "Scan paired .txt timing files and .fits/.fits.fz data cubes "
                                  "without decompressing FITS pixels, reporting frame rate, "
                                  "jitter, dropped frames, orphans, and time span." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(mlt_tlmindex),
    .compute = tlmindex_compute,
};

EXPORT_FPSEXEC(milk, "mlt-tlmindex", runnable_mlt_tlmindex);
