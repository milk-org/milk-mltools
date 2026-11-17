// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    tlm_fileset.c
 * @brief   Telemetry directory scanning, cube pairing, time-bound parsing, and timing resolution
 */

#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "milk_runnable.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "tlm_fileset.h"

typedef struct
{
    char stem[512];
    char stream[128];
    int  has_txt;
    int  has_fits;
    int  has_fz;
    int  has_hdr;
    char txt_name[512];
    char fits_name[512];
    char fz_name[512];
    char hdr_name[512];
} mlt_stem_rec;

static int is_digit_char(char c)
{
    return (c >= '0' && c <= '9');
}

static int parse_hhmmss_suffix(const char *stem, char *stream_out, size_t stream_cap)
{
    size_t len = strlen(stem);
    if (len < 10)
    {
        return 0;
    }

    for (size_t i = len; i > 0; i--)
    {
        size_t pos = i - 1;
        if (stem[pos] != '_')
        {
            continue;
        }
        const char *ts = stem + pos + 1;
        if (strlen(ts) < 8)
        {
            continue;
        }
        if (!is_digit_char(ts[0]) || !is_digit_char(ts[1]) || ts[2] != ':' ||
            !is_digit_char(ts[3]) || !is_digit_char(ts[4]) || ts[5] != ':' ||
            !is_digit_char(ts[6]) || !is_digit_char(ts[7]))
        {
            continue;
        }
        if (ts[8] != '\0' && ts[8] != '.')
        {
            continue;
        }
        if (pos == 0 || pos >= stream_cap)
        {
            return 0;
        }
        memcpy(stream_out, stem, pos);
        stream_out[pos] = '\0';
        return 1;
    }
    return 0;
}

typedef enum
{
    MLT_EXT_NONE = 0,
    MLT_EXT_TXT  = 1,
    MLT_EXT_FITS = 2,
    MLT_EXT_FZ   = 3,
    MLT_EXT_HDR  = 4
} mlt_ext_kind;

static mlt_ext_kind classify_filename(const char *name, char *stem_out, size_t stem_cap)
{
    size_t len = strlen(name);
    struct
    {
        const char  *ext;
        size_t       elen;
        mlt_ext_kind kind;
    } const suffixes[] = {
        { ".fits.header", 12, MLT_EXT_HDR },
        { ".fits.fz", 8, MLT_EXT_FZ },
        { ".fits", 5, MLT_EXT_FITS },
        { ".txt", 4, MLT_EXT_TXT },
    };

    for (size_t s = 0; s < sizeof(suffixes) / sizeof(suffixes[0]); s++)
    {
        size_t elen = suffixes[s].elen;
        if (len > elen && strcmp(name + len - elen, suffixes[s].ext) == 0)
        {
            size_t slen = len - elen;
            if (slen >= stem_cap)
            {
                return MLT_EXT_NONE;
            }
            memcpy(stem_out, name, slen);
            stem_out[slen] = '\0';
            return suffixes[s].kind;
        }
    }
    return MLT_EXT_NONE;
}

typedef struct
{
    char         name[512];
    char         stem[512];
    char         stream[128];
    mlt_ext_kind kind;
} mlt_raw_file;

static int cmp_raw_file(const void *a, const void *b)
{
    const mlt_raw_file *fa = (const mlt_raw_file *) a;
    const mlt_raw_file *fb = (const mlt_raw_file *) b;
    int                 c  = strcmp(fa->stem, fb->stem);
    if (c != 0)
    {
        return c;
    }
    return (int) fa->kind - (int) fb->kind;
}

static int cmp_cube_entry(const void *a, const void *b)
{
    const MLT_CUBE_ENTRY *ca = (const MLT_CUBE_ENTRY *) a;
    const MLT_CUBE_ENTRY *cb = (const MLT_CUBE_ENTRY *) b;
    if (ca->t_first_ns < cb->t_first_ns)
    {
        return -1;
    }
    if (ca->t_first_ns > cb->t_first_ns)
    {
        return 1;
    }
    return strcmp(ca->txt_path, cb->txt_path);
}

static errno_t join_path(char *dst, size_t cap, const char *dir, const char *name)
{
    size_t dlen = strlen(dir);
    int    n;
    if (dlen > 0 && dir[dlen - 1] == '/')
    {
        n = snprintf(dst, cap, "%s%s", dir, name);
    }
    else
    {
        n = snprintf(dst, cap, "%s/%s", dir, name);
    }
    if (n < 0 || (size_t) n >= cap)
    {
        fprintf(stderr, "ERROR: path exceeds %zu bytes: %s/%s\n", cap, dir, name);
        return RETURN_FAILURE;
    }
    return RETURN_SUCCESS;
}

static int is_stream_match(const char *file_stream, const char *filter)
{
    if (filter == NULL || filter[0] == '\0' || strcmp(filter, "auto") == 0)
    {
        return 1;
    }
    return strcmp(file_stream, filter) == 0;
}

static errno_t finalize_fileset_spans(MLT_FILESET *fs, const char *tsrc_mode)
{
    if (fs->ncubes == 0)
    {
        fprintf(stderr, "ERROR: no paired telemetry cubes found\n");
        return RETURN_FAILURE;
    }

    int   *ok_arr  = (int *) calloc(fs->ncubes, sizeof(int));
    int   *acq_arr = (int *) calloc(fs->ncubes, sizeof(int));
    size_t i;

    if (ok_arr == NULL || acq_arr == NULL)
    {
        free(ok_arr);
        free(acq_arr);
        return RETURN_FAILURE;
    }

#pragma omp parallel for schedule(dynamic, 4) if (fs->ncubes > 8)
    for (i = 0; i < fs->ncubes; i++)
    {
        MLT_CUBE_ENTRY *e = &fs->entries[i];
        if (milk_tlm_peek_span(e->txt_path, &e->first_row, &e->last_row, &e->approx_nrows) ==
            RETURN_SUCCESS)
        {
            ok_arr[i]  = 1;
            acq_arr[i] = (e->first_row.t_acq_ns > 0) ? 1 : 0;
        }
    }

    size_t valid_n = 0;
    int    any_acq = 0;
    for (i = 0; i < fs->ncubes; i++)
    {
        if (ok_arr[i])
        {
            if (valid_n != i)
            {
                fs->entries[valid_n] = fs->entries[i];
            }
            if (acq_arr[i])
            {
                any_acq = 1;
            }
            valid_n++;
        }
        else
        {
            fs->norphan_txt++;
        }
    }
    free(ok_arr);
    free(acq_arr);
    fs->ncubes = valid_n;

    if (fs->ncubes == 0)
    {
        fprintf(stderr, "ERROR: all paired telemetry timing files failed to parse\n");
        return RETURN_FAILURE;
    }

    fs->has_acq = any_acq;
    if (tsrc_mode != NULL && strcmp(tsrc_mode, "log") == 0)
    {
        fs->use_acq = 0;
    }
    else if (tsrc_mode != NULL && strcmp(tsrc_mode, "acq") == 0)
    {
        fs->use_acq = 1;
    }
    else
    {
        fs->use_acq = any_acq;
    }

    for (i = 0; i < fs->ncubes; i++)
    {
        MLT_CUBE_ENTRY *e = &fs->entries[i];
        e->t_first_ns     = mlt_row_teff_ns(&e->first_row, fs->use_acq);
        e->t_last_ns      = mlt_row_teff_ns(&e->last_row, fs->use_acq);
    }

    qsort(fs->entries, fs->ncubes, sizeof(MLT_CUBE_ENTRY), cmp_cube_entry);

    fs->t_first_ns = fs->entries[0].t_first_ns;
    fs->t_last_ns  = fs->entries[fs->ncubes - 1].t_last_ns;
    for (i = 0; i < fs->ncubes; i++)
    {
        if (fs->entries[i].t_first_ns < fs->t_first_ns)
        {
            fs->t_first_ns = fs->entries[i].t_first_ns;
        }
        if (fs->entries[i].t_last_ns > fs->t_last_ns)
        {
            fs->t_last_ns = fs->entries[i].t_last_ns;
        }
    }

    fs->t_epoch_sec = (fs->t_first_ns / 1000000000LL / 86400LL) * 86400LL;
    return RETURN_SUCCESS;
}

static errno_t scan_single_file(const char *path, const char *tsrc_mode, MLT_FILESET *fs)
{
    char stem[MLT_PATH_MAX];
    char txt[MLT_PATH_MAX];
    char fits[MLT_PATH_MAX];
    int  is_fz = 0;

    size_t len = strlen(path);
    if (len >= sizeof(stem))
    {
        fprintf(stderr, "ERROR: path too long: %s\n", path);
        return RETURN_FAILURE;
    }

    if (len > 8 && strcmp(path + len - 8, ".fits.fz") == 0)
    {
        memcpy(stem, path, len - 8);
        stem[len - 8] = '\0';
        snprintf(fits, sizeof(fits), "%s", path);
        snprintf(txt, sizeof(txt), "%s.txt", stem);
        is_fz = 1;
    }
    else if (len > 5 && strcmp(path + len - 5, ".fits") == 0)
    {
        memcpy(stem, path, len - 5);
        stem[len - 5] = '\0';
        snprintf(fits, sizeof(fits), "%s", path);
        snprintf(txt, sizeof(txt), "%s.txt", stem);
    }
    else
    {
        fprintf(stderr, "ERROR: unsupported telemetry file extension: %s\n", path);
        return RETURN_FAILURE;
    }

    if (access(txt, R_OK) != 0)
    {
        fprintf(stderr, "ERROR: missing companion timing file: %s\n", txt);
        return RETURN_FAILURE;
    }

    fs->entries = (MLT_CUBE_ENTRY *) calloc(1, sizeof(MLT_CUBE_ENTRY));
    if (fs->entries == NULL)
    {
        return RETURN_FAILURE;
    }
    fs->ncubes = 1;
    snprintf(fs->entries[0].txt_path, sizeof(fs->entries[0].txt_path), "%s", txt);
    snprintf(fs->entries[0].fits_path, sizeof(fs->entries[0].fits_path), "%s", fits);
    fs->entries[0].is_fz = is_fz;

    const char *base = strrchr(stem, '/');
    base             = (base != NULL) ? base + 1 : stem;
    if (!parse_hhmmss_suffix(base, fs->stream, sizeof(fs->stream)))
    {
        snprintf(fs->stream, sizeof(fs->stream), "%s", "telemetry");
    }

    return finalize_fileset_spans(fs, tsrc_mode);
}

errno_t mlt_fileset_scan(const char  *path,
                         const char  *stream_filter,
                         const char  *tsrc_mode,
                         MLT_FILESET *fs)
{
    if (fs == NULL || path == NULL || path[0] == '\0')
    {
        return RETURN_FAILURE;
    }
    memset(fs, 0, sizeof(*fs));

    struct stat st;
    if (stat(path, &st) != 0)
    {
        fprintf(stderr, "ERROR: cannot stat telemetry path '%s': %s\n", path, strerror(errno));
        return RETURN_FAILURE;
    }
    if (S_ISREG(st.st_mode))
    {
        return scan_single_file(path, tsrc_mode, fs);
    }
    if (!S_ISDIR(st.st_mode))
    {
        fprintf(stderr, "ERROR: '%s' is neither a directory nor a regular file\n", path);
        return RETURN_FAILURE;
    }

    DIR *dp = opendir(path);
    if (dp == NULL)
    {
        fprintf(stderr, "ERROR: cannot open directory '%s': %s\n", path, strerror(errno));
        return RETURN_FAILURE;
    }

    size_t        cap   = 256;
    size_t        nraw  = 0;
    mlt_raw_file *raws  = (mlt_raw_file *) malloc(cap * sizeof(mlt_raw_file));
    char          streams[16][128];
    size_t        nstreams = 0;

    if (raws == NULL)
    {
        closedir(dp);
        return RETURN_FAILURE;
    }

    struct dirent *de;
    while ((de = readdir(dp)) != NULL)
    {
        if (de->d_name[0] == '.')
        {
            continue;
        }
        char         stem[512];
        mlt_ext_kind kind = classify_filename(de->d_name, stem, sizeof(stem));
        if (kind == MLT_EXT_NONE)
        {
            continue;
        }

        char fstream[128];
        if (!parse_hhmmss_suffix(stem, fstream, sizeof(fstream)))
        {
            continue;
        }
        if (!is_stream_match(fstream, stream_filter))
        {
            continue;
        }

        int seen = 0;
        for (size_t s = 0; s < nstreams; s++)
        {
            if (strcmp(streams[s], fstream) == 0)
            {
                seen = 1;
                break;
            }
        }
        if (!seen && nstreams < 16)
        {
            snprintf(streams[nstreams], sizeof(streams[nstreams]), "%s", fstream);
            nstreams++;
        }

        if (nraw == cap)
        {
            cap *= 2;
            mlt_raw_file *tmp = (mlt_raw_file *) realloc(raws, cap * sizeof(mlt_raw_file));
            if (tmp == NULL)
            {
                free(raws);
                closedir(dp);
                return RETURN_FAILURE;
            }
            raws = tmp;
        }
        snprintf(raws[nraw].name, sizeof(raws[nraw].name), "%s", de->d_name);
        snprintf(raws[nraw].stem, sizeof(raws[nraw].stem), "%s", stem);
        snprintf(raws[nraw].stream, sizeof(raws[nraw].stream), "%s", fstream);
        raws[nraw].kind = kind;
        nraw++;
    }
    closedir(dp);

    if (nraw == 0)
    {
        free(raws);
        fprintf(stderr, "ERROR: no telemetry files matching stream '%s' in '%s'\n",
                (stream_filter && stream_filter[0]) ? stream_filter : "*", path);
        return RETURN_FAILURE;
    }

    if ((stream_filter == NULL || stream_filter[0] == '\0' || strcmp(stream_filter, "auto") == 0) &&
        nstreams > 1)
    {
        fprintf(stderr, "ERROR: multiple telemetry streams found in '%s' (pass .stream=<name>):",
                path);
        for (size_t s = 0; s < nstreams; s++)
        {
            fprintf(stderr, "%s %s", (s == 0) ? "" : ",", streams[s]);
        }
        fprintf(stderr, "\n");
        free(raws);
        return RETURN_FAILURE;
    }

    snprintf(fs->stream, sizeof(fs->stream), "%s",
             (stream_filter && stream_filter[0] && strcmp(stream_filter, "auto") != 0)
                 ? stream_filter
                 : streams[0]);

    qsort(raws, nraw, sizeof(mlt_raw_file), cmp_raw_file);

    fs->entries = (MLT_CUBE_ENTRY *) calloc(nraw, sizeof(MLT_CUBE_ENTRY));
    if (fs->entries == NULL)
    {
        free(raws);
        return RETURN_FAILURE;
    }

    size_t i = 0;
    while (i < nraw)
    {
        mlt_stem_rec rec;
        memset(&rec, 0, sizeof(rec));
        snprintf(rec.stem, sizeof(rec.stem), "%s", raws[i].stem);

        size_t j = i;
        while (j < nraw && strcmp(raws[j].stem, rec.stem) == 0)
        {
            if (raws[j].kind == MLT_EXT_TXT)
            {
                rec.has_txt = 1;
                snprintf(rec.txt_name, sizeof(rec.txt_name), "%s", raws[j].name);
            }
            else if (raws[j].kind == MLT_EXT_FITS)
            {
                rec.has_fits = 1;
                snprintf(rec.fits_name, sizeof(rec.fits_name), "%s", raws[j].name);
            }
            else if (raws[j].kind == MLT_EXT_FZ)
            {
                rec.has_fz = 1;
                snprintf(rec.fz_name, sizeof(rec.fz_name), "%s", raws[j].name);
            }
            else if (raws[j].kind == MLT_EXT_HDR)
            {
                rec.has_hdr = 1;
                snprintf(rec.hdr_name, sizeof(rec.hdr_name), "%s", raws[j].name);
            }
            j++;
        }

        if (rec.has_txt && (rec.has_fits || rec.has_fz))
        {
            MLT_CUBE_ENTRY *e = &fs->entries[fs->ncubes];
            if (join_path(e->txt_path, sizeof(e->txt_path), path, rec.txt_name) != RETURN_SUCCESS)
            {
                free(raws);
                mlt_fileset_free(fs);
                return RETURN_FAILURE;
            }
            const char *fname = rec.has_fits ? rec.fits_name : rec.fz_name;
            e->is_fz          = rec.has_fits ? 0 : 1;
            if (join_path(e->fits_path, sizeof(e->fits_path), path, fname) != RETURN_SUCCESS)
            {
                free(raws);
                mlt_fileset_free(fs);
                return RETURN_FAILURE;
            }
            if (rec.has_hdr)
            {
                e->has_hdr = 1;
                if (join_path(e->hdr_path, sizeof(e->hdr_path), path, rec.hdr_name) !=
                    RETURN_SUCCESS)
                {
                    free(raws);
                    mlt_fileset_free(fs);
                    return RETURN_FAILURE;
                }
            }
            fs->ncubes++;
        }
        else if (rec.has_txt)
        {
            fs->norphan_txt++;
        }
        else if (rec.has_fits || rec.has_fz)
        {
            fs->norphan_fits++;
        }

        i = j;
    }

    free(raws);
    if (fs->ncubes == 0)
    {
        fprintf(stderr, "ERROR: no paired (.txt + .fits[.fz]) cubes for stream '%s' in '%s'\n",
                fs->stream, path);
        mlt_fileset_free(fs);
        return RETURN_FAILURE;
    }

    return finalize_fileset_spans(fs, tsrc_mode);
}

void mlt_fileset_free(MLT_FILESET *fs)
{
    if (fs == NULL)
    {
        return;
    }
    free(fs->entries);
    memset(fs, 0, sizeof(*fs));
}

static errno_t parse_decimal_ns(const char *s, int64_t *sec_out, int64_t *nsec_out)
{
    if (s == NULL || *s == '\0')
    {
        return RETURN_FAILURE;
    }
    int neg = 0;
    if (*s == '-')
    {
        neg = 1;
        s++;
    }
    if (!is_digit_char(*s))
    {
        return RETURN_FAILURE;
    }
    int64_t sec = 0;
    while (is_digit_char(*s))
    {
        sec = sec * 10 + (*s - '0');
        s++;
    }
    int64_t nsec = 0;
    if (*s == '.')
    {
        s++;
        int64_t scale = 100000000LL;
        while (is_digit_char(*s))
        {
            if (scale > 0)
            {
                nsec += (*s - '0') * scale;
                scale /= 10;
            }
            s++;
        }
    }
    if (*s != '\0')
    {
        return RETURN_FAILURE;
    }
    if (neg)
    {
        sec  = -sec;
        nsec = -nsec;
    }
    *sec_out  = sec;
    *nsec_out = nsec;
    return RETURN_SUCCESS;
}

errno_t mlt_parse_time_bound(const char *spec,
                             int64_t     t_epoch_sec,
                             int64_t     t_ref_ns,
                             int64_t     default_ns,
                             int64_t    *out_ns)
{
    if (out_ns == NULL)
    {
        return RETURN_FAILURE;
    }
    if (spec == NULL || spec[0] == '\0' || strcmp(spec, "0") == 0 || strcmp(spec, "all") == 0 ||
        strcmp(spec, "-1") == 0)
    {
        *out_ns = default_ns;
        return RETURN_SUCCESS;
    }

    if (spec[0] == '+')
    {
        char  *endp = NULL;
        double val  = strtod(spec + 1, &endp);
        double mult = 1e9;
        if (endp == spec + 1 || !isfinite(val))
        {
            fprintf(stderr, "ERROR: invalid relative offset '%s'\n", spec);
            return RETURN_FAILURE;
        }
        if (*endp == '\0' || strcmp(endp, "s") == 0)
        {
            mult = 1e9;
        }
        else if (strcmp(endp, "ms") == 0)
        {
            mult = 1e6;
        }
        else if (strcmp(endp, "m") == 0)
        {
            mult = 60.0 * 1e9;
        }
        else if (strcmp(endp, "h") == 0)
        {
            mult = 3600.0 * 1e9;
        }
        else
        {
            fprintf(stderr, "ERROR: invalid unit in offset '%s'\n", spec);
            return RETURN_FAILURE;
        }
        *out_ns = t_ref_ns + llround(val * mult);
        return RETURN_SUCCESS;
    }

    if (strchr(spec, ':') != NULL)
    {
        int     hh = 0;
        int     mm = 0;
        int     ss = 0;
        char    rest[32];
        int     nread = 0;
        rest[0]       = '\0';
        if (sscanf(spec, "%2d:%2d:%2d%31s%n", &hh, &mm, &ss, rest, &nread) < 3)
        {
            fprintf(stderr, "ERROR: invalid HH:MM:SS time '%s'\n", spec);
            return RETURN_FAILURE;
        }
        int64_t frac_ns = 0;
        if (rest[0] == '.')
        {
            int64_t dummy_sec = 0;
            char    tmp[40];
            snprintf(tmp, sizeof(tmp), "0%s", rest);
            if (parse_decimal_ns(tmp, &dummy_sec, &frac_ns) != RETURN_SUCCESS)
            {
                fprintf(stderr, "ERROR: invalid fractional seconds in '%s'\n", spec);
                return RETURN_FAILURE;
            }
        }
        else if (rest[0] != '\0')
        {
            fprintf(stderr, "ERROR: trailing characters in '%s'\n", spec);
            return RETURN_FAILURE;
        }
        int64_t sec_total = t_epoch_sec + (int64_t) hh * 3600LL + (int64_t) mm * 60LL + ss;
        *out_ns           = sec_total * 1000000000LL + frac_ns;
        return RETURN_SUCCESS;
    }

    int64_t sec  = 0;
    int64_t nsec = 0;
    if (parse_decimal_ns(spec, &sec, &nsec) != RETURN_SUCCESS)
    {
        fprintf(stderr, "ERROR: invalid numeric time bound '%s'\n", spec);
        return RETURN_FAILURE;
    }
    if (sec >= 1000000000LL)
    {
        *out_ns = sec * 1000000000LL + nsec;
    }
    else
    {
        *out_ns = (t_epoch_sec + sec) * 1000000000LL + nsec;
    }
    return RETURN_SUCCESS;
}

errno_t mlt_load_timing_arg(const char    *arg,
                            double       **t_eff_out,
                            MILK_TLM_ROW **rows_out,
                            size_t        *n_out,
                            int64_t       *epoch_sec_out)
{
    if (arg == NULL || arg[0] == '\0' || n_out == NULL || epoch_sec_out == NULL)
    {
        return RETURN_FAILURE;
    }

    IMGID img = imgid_make_from_name(arg);
    resolveIMGID(&img, ERRMODE_NULL, dcimg, dcnimg);
    if (img.ID == -1 && strstr(arg, ".txt") == NULL && strchr(arg, '/') == NULL)
    {
        read_sharedmem_image(img.name, dcimg, dcnimg);
        resolveIMGID(&img, ERRMODE_NULL, dcimg, dcnimg);
    }

    if (img.ID != -1)
    {
        IMAGE *im = img.im;
        if (im->md[0].naxis < 2 || im->md[0].size[0] < 6)
        {
            fprintf(stderr, "ERROR: timing image '%s' must be (6, N), got naxis=%d size[0]=%u\n",
                    arg, (int) im->md[0].naxis, im->md[0].size[0]);
            imgid_free(&img);
            return RETURN_FAILURE;
        }
        size_t   n      = im->md[0].size[1];
        uint32_t stride = im->md[0].size[0];
        uint8_t  dt     = im->md[0].datatype;
        if (dt != _DATATYPE_DOUBLE && dt != _DATATYPE_FLOAT)
        {
            fprintf(stderr, "ERROR: timing image '%s' must be FLOAT or DOUBLE\n", arg);
            imgid_free(&img);
            return RETURN_FAILURE;
        }

        long    ep_l = 0;
        int64_t ep   = 0;
        if (image_read_keyword_L(img.name, "MLTEPOCH", &ep_l) >= 0 && ep_l > 0)
        {
            ep = (int64_t) ep_l;
        }
        *epoch_sec_out = ep;
        *n_out         = n;

        if (t_eff_out != NULL)
        {
            double *arr = (double *) malloc(n * sizeof(double));
            if (arr == NULL)
            {
                imgid_free(&img);
                return RETURN_FAILURE;
            }
            for (size_t i = 0; i < n; i++)
            {
                size_t idx = i * (size_t) stride;
                arr[i]     = (dt == _DATATYPE_DOUBLE) ? im->array.D[idx]
                                                      : (double) im->array.F[idx];
            }
            *t_eff_out = arr;
        }

        if (rows_out != NULL)
        {
            MILK_TLM_ROW *rows = (MILK_TLM_ROW *) calloc(n, sizeof(MILK_TLM_ROW));
            if (rows == NULL)
            {
                if (t_eff_out != NULL)
                {
                    free(*t_eff_out);
                    *t_eff_out = NULL;
                }
                imgid_free(&img);
                return RETURN_FAILURE;
            }
            int64_t ep_ns = ep * 1000000000LL;
            for (size_t i = 0; i < n; i++)
            {
                size_t b = i * (size_t) stride;
                double t1 =
                    (dt == _DATATYPE_DOUBLE) ? im->array.D[b + 1] : (double) im->array.F[b + 1];
                double t2 =
                    (dt == _DATATYPE_DOUBLE) ? im->array.D[b + 2] : (double) im->array.F[b + 2];
                double c0 =
                    (dt == _DATATYPE_DOUBLE) ? im->array.D[b + 3] : (double) im->array.F[b + 3];
                double c1 =
                    (dt == _DATATYPE_DOUBLE) ? im->array.D[b + 4] : (double) im->array.F[b + 4];
                double mi =
                    (dt == _DATATYPE_DOUBLE) ? im->array.D[b + 5] : (double) im->array.F[b + 5];
                rows[i].frame_idx = (int64_t) i;
                rows[i].t_log_ns  = ep_ns + llround(t1 * 1e9);
                rows[i].t_acq_ns  = (t2 != 0.0) ? (ep_ns + llround(t2 * 1e9)) : 0;
                rows[i].cnt0      = (c0 >= 0.0) ? (uint64_t) llround(c0) : 0;
                rows[i].cnt1      = (c1 >= 0.0) ? (uint64_t) llround(c1) : 0;
                rows[i].main_idx  = (mi >= 0.0) ? (uint64_t) llround(mi) : 0;
            }
            *rows_out = rows;
        }
        imgid_free(&img);
        return RETURN_SUCCESS;
    }
    imgid_free(&img);

    MILK_TLM_TABLE tbl;
    if (milk_tlm_read_file(arg, &tbl) != RETURN_SUCCESS)
    {
        fprintf(stderr, "ERROR: '%s' is neither a loaded image nor a readable timing file\n", arg);
        return RETURN_FAILURE;
    }
    if (tbl.nrows == 0)
    {
        milk_tlm_free(&tbl);
        fprintf(stderr, "ERROR: timing file '%s' has 0 rows\n", arg);
        return RETURN_FAILURE;
    }

    int     use_acq = (tbl.rows[0].t_acq_ns > 0) ? 1 : 0;
    int64_t first_ns = mlt_row_teff_ns(&tbl.rows[0], use_acq);
    int64_t ep       = (first_ns / 1000000000LL / 86400LL) * 86400LL;
    int64_t ep_ns    = ep * 1000000000LL;

    *epoch_sec_out = ep;
    *n_out         = tbl.nrows;

    if (t_eff_out != NULL)
    {
        double *arr = (double *) malloc(tbl.nrows * sizeof(double));
        if (arr == NULL)
        {
            milk_tlm_free(&tbl);
            return RETURN_FAILURE;
        }
        for (size_t i = 0; i < tbl.nrows; i++)
        {
            arr[i] = (double) (mlt_row_teff_ns(&tbl.rows[i], use_acq) - ep_ns) * 1e-9;
        }
        *t_eff_out = arr;
    }

    if (rows_out != NULL)
    {
        *rows_out = tbl.rows;
        tbl.rows  = NULL;
        tbl.nrows = 0;
        tbl.cap   = 0;
    }
    else
    {
        milk_tlm_free(&tbl);
    }

    return RETURN_SUCCESS;
}
