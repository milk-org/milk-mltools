// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    tlm_fileset.h
 * @brief   Telemetry directory discovery, cube pairing, time bounds, and timing resolvers
 */

#ifndef MILK_MLTOOLS_TLM_FILESET_H
#define MILK_MLTOOLS_TLM_FILESET_H

#include <stddef.h>
#include <stdint.h>

#include "COREMOD_memory/tlm_timing.h"

#ifdef __cplusplus
extern "C"
{
#endif

#define MLT_PATH_MAX 4096

    /**
     * struct MLT_CUBE_ENTRY - One paired telemetry cube and companion timing file
     */
    typedef struct
    {
        char         txt_path[MLT_PATH_MAX];
        char         fits_path[MLT_PATH_MAX];
        char         hdr_path[MLT_PATH_MAX];
        int          is_fz;
        int          has_hdr;
        MILK_TLM_ROW first_row;
        MILK_TLM_ROW last_row;
        size_t       approx_nrows;
        int64_t      t_first_ns;
        int64_t      t_last_ns;
    } MLT_CUBE_ENTRY;

    /**
     * struct MLT_FILESET - Sorted collection of telemetry cubes for one stream
     */
    typedef struct
    {
        MLT_CUBE_ENTRY *entries;
        size_t          ncubes;
        size_t          norphan_txt;
        size_t          norphan_fits;
        char            stream[128];
        int64_t         t_epoch_sec;
        int64_t         t_first_ns;
        int64_t         t_last_ns;
        int             has_acq;
        int             use_acq;
    } MLT_FILESET;

    /**
     * mlt_row_teff_ns - Select effective nanosecond timestamp for one timing row
     * @row:     Timing record
     * @use_acq: Prefer acquisition timestamp when non-zero
     *
     * Return: Acquisition timestamp when @use_acq is set and `t_acq_ns > 0`, else `t_log_ns`
     */
    static inline int64_t mlt_row_teff_ns(const MILK_TLM_ROW *row, int use_acq)
    {
        if (use_acq && row->t_acq_ns > 0)
        {
            return row->t_acq_ns;
        }
        return row->t_log_ns;
    }

    /**
     * mlt_fileset_scan - Scan a directory or FITS file and pair `.txt` and `.fits[.fz]` cubes
     * @path:          Telemetry directory path or single `.fits`/`.fits.fz` file
     * @stream_filter: Optional stream prefix (`""` or `"auto"` for automatic detection)
     * @tsrc_mode:     `"auto"`, `"acq"`, or `"log"`
     * @fs:            Output fileset (zero-initialized internally; free via mlt_fileset_free())
     *
     * Return: `RETURN_SUCCESS` on success, `RETURN_FAILURE` on error
     */
    errno_t mlt_fileset_scan(const char  *path,
                             const char  *stream_filter,
                             const char  *tsrc_mode,
                             MLT_FILESET *fs);

    /**
     * mlt_fileset_free - Release memory owned by an `MLT_FILESET`
     * @fs: Fileset to free (safe if NULL)
     */
    void mlt_fileset_free(MLT_FILESET *fs);

    /**
     * mlt_parse_time_bound - Parse a telemetry time boundary specification into Unix nanoseconds
     * @spec:        `""`/`"0"`/`"all"`, `"+<val>[ms|s|m|h]"`, `"HH:MM:SS[.frac]"`, rel s, or Unix s
     * @t_epoch_sec: Reference UT day boundary (Unix seconds at 00:00:00 UT)
     * @t_ref_ns:    Reference start timestamp in nanoseconds (used for `+<val>`)
     * @default_ns:  Value returned for empty/`"0"`/`"all"`/`"-1"`
     * @out_ns:      Output Unix timestamp in nanoseconds
     *
     * Return: `RETURN_SUCCESS` on success, `RETURN_FAILURE` on syntax error
     */
    errno_t mlt_parse_time_bound(const char *spec,
                                 int64_t     t_epoch_sec,
                                 int64_t     t_ref_ns,
                                 int64_t     default_ns,
                                 int64_t    *out_ns);

    /**
     * mlt_load_timing_arg - Resolve timing from an in-memory `6xN` image or `.txt` file path
     * @arg:           Stream/image name or path to `.txt` file
     * @t_eff_out:     Optional output array of `N` timestamps in seconds since `*epoch_sec_out`
     * @rows_out:      Optional output array of `N` full `MILK_TLM_ROW` records
     * @n_out:         Output frame count `N`
     * @epoch_sec_out: Output reference Unix seconds (`MLTEPOCH` or dataset day 00:00:00 UT)
     *
     * Return: `RETURN_SUCCESS` on success, `RETURN_FAILURE` on error
     */
    errno_t mlt_load_timing_arg(const char    *arg,
                                double       **t_eff_out,
                                MILK_TLM_ROW **rows_out,
                                size_t        *n_out,
                                int64_t       *epoch_sec_out);

#ifdef __cplusplus
}
#endif

#endif /* MILK_MLTOOLS_TLM_FILESET_H */
