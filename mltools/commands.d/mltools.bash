# SPDX-FileCopyrightText: 2026 Olivier Guyon et al
#
# SPDX-License-Identifier: LGPL-3.0-or-later
#
# mltools.bash - High-level machine learning & time-series commands (mlt.*)
# Loaded automatically from $MILK_INSTALLDIR/share/milk/commands.d/mltools.bash

declare -gi __MLT_SEQ=0

__mlt_short_path() {
    local -n __sp_out="$1"
    local __sp_raw="$2"
    local -n __sp_links="$3"
    if [[ -z "${__sp_raw}" || "${__sp_raw}" == "none" ]]; then
        __sp_out="${__sp_raw}"
        return 0
    fi
    if (( ${#__sp_raw} <= 54 )); then
        __sp_out="${__sp_raw}"
        return 0
    fi
    local __abs
    __abs="$(realpath -m "${__sp_raw}")" || return 1
    local __lnk="/tmp/mlt_p_${BASHPID}_$((++__MLT_SEQ))"
    ln -snf "${__abs}" "${__lnk}" || return 1
    __sp_links+=("${__lnk}")
    __sp_out="${__lnk}"
}

__mlt_short_outpath() {
    local -n __so_out="$1"
    local __so_raw="$2"
    local -n __so_links="$3"
    local __dir __base __lnk
    __dir="$(dirname "${__so_raw}")"
    __base="$(basename "${__so_raw}")"
    mkdir -p "${__dir}" || return 1
    if (( ${#__so_raw} <= 58 )); then
        __so_out="${__so_raw}"
        return 0
    fi
    local __abs
    __abs="$(realpath -m "${__dir}")" || return 1
    __lnk="/tmp/ml_${BASHPID}_$((++__MLT_SEQ))"
    ln -snf "${__abs}" "${__lnk}" || return 1
    __so_links+=("${__lnk}")
    if (( ${#__lnk} + 1 + ${#__base} > 62 )); then
        echo "mlt: output basename too long for FPS limit: ${__base}" >&2
        return 1
    fi
    __so_out="${__lnk}/${__base}"
}

__mlt_cleanup() {
    local -n __cl_links="$1"
    local __cl_l
    for __cl_l in "${__cl_links[@]}"; do
        rm -f "${__cl_l}"
    done
}

__mlt_rm_images() {
    local __im
    for __im in "$@"; do
        mem.rm "${__im}" 0 >/dev/null 2>&1 || true
    done
}

__mlt_read_kv() {
    local __kv_file="$1"
    local -n __kv_map="$2"
    local __line __k __v
    while IFS='=' read -r __k __v || [[ -n "${__k}" ]]; do
        [[ -z "${__k}" || "${__k}" == \#* ]] && continue
        __kv_map["${__k}"]="${__v}"
    done < "${__kv_file}"
}

__mlt_run_tlmindex() {
    local __idx_dir="$1"
    local __idx_stream="$2"
    local -n __idx_kv="$3"
    local -a __idx_links=()
    local __sdir __rep
    __rep="/tmp/mlt_idx_${BASHPID}_$((++__MLT_SEQ)).txt"
    __mlt_short_path __sdir "${__idx_dir}" __idx_links || return 1

    local -a __cmd=(mlt.tlmindex ".dir=${__sdir}" ".outreport=${__rep}")
    [[ -n "${__idx_stream}" ]] && __cmd+=(".stream=${__idx_stream}")
    "${__cmd[@]}" >/dev/null
    local __rc=$?
    if (( __rc == 0 )) && [[ -f "${__rep}" ]]; then
        __mlt_read_kv "${__rep}" __idx_kv
    fi
    rm -f "${__rep}"
    __mlt_cleanup __idx_links
    return ${__rc}
}

# Validate telemetry cubes/timing health and optionally compute two-stream time overlap
mlt.tlmcheck() {
    @args 'dir:str:telemetry directory' \
          'stream:str=:stream prefix (auto if empty)' \
          'dir2:str=:optional second telemetry directory' \
          'stream2:str=:optional second stream prefix' \
          'outreport:str=:optional output summary report file' -- "$@"

    local -A __k1=() __k2=()
    __mlt_run_tlmindex "${dir}" "${stream}" __k1 || return 1

    local __st1="${__k1[stream]:-${stream}}"
    local __warn1=""
    (( ${__k1[norphans]:-0} > 0 )) && __warn1+=" WARN_ORPHANS=${__k1[norphans]}"
    (( ${__k1[ndropped]:-0} > 0 )) && __warn1+=" WARN_DROPPED=${__k1[ndropped]}"
    (( ${__k1[has_acq]:-0} == 0 )) && __warn1+=" NOTE_NO_ACQ_TS"

    printf 'Stream A: %s (%s)\n' "${__st1}" "${dir}"
    printf '  cubes=%s  frames=%s  fps=%s Hz  jitter=%s us\n' \
        "${__k1[ncubes]}" "${__k1[nframes]}" "${__k1[fps]}" "${__k1[jitter_us]}"
    printf '  orphans=%s (txt=%s fits=%s)  dropped=%s  has_acq=%s%s\n' \
        "${__k1[norphans]}" "${__k1[norphan_txt]}" "${__k1[norphan_fits]}" \
        "${__k1[ndropped]}" "${__k1[has_acq]}" "${__warn1}"
    printf '  epoch_sec=%s  span_rel=[%s, %s] s\n' \
        "${__k1[epoch_sec]}" "${__k1[tstart_rel]}" "${__k1[tend_rel]}"

    local __ov_sec=0 __ov_ok=0 __ov0_a=0 __ov1_a=0
    if [[ -n "${dir2}" || -n "${stream2}" ]]; then
        local __d2="${dir2:-${dir}}"
        __mlt_run_tlmindex "${__d2}" "${stream2}" __k2 || return 1
        local __st2="${__k2[stream]:-${stream2}}"
        local __warn2=""
        (( ${__k2[norphans]:-0} > 0 )) && __warn2+=" WARN_ORPHANS=${__k2[norphans]}"
        (( ${__k2[ndropped]:-0} > 0 )) && __warn2+=" WARN_DROPPED=${__k2[ndropped]}"
        (( ${__k2[has_acq]:-0} == 0 )) && __warn2+=" NOTE_NO_ACQ_TS"

        printf 'Stream B: %s (%s)\n' "${__st2}" "${__d2}"
        printf '  cubes=%s  frames=%s  fps=%s Hz  jitter=%s us\n' \
            "${__k2[ncubes]}" "${__k2[nframes]}" "${__k2[fps]}" "${__k2[jitter_us]}"
        printf '  orphans=%s (txt=%s fits=%s)  dropped=%s  has_acq=%s%s\n' \
            "${__k2[norphans]}" "${__k2[norphan_txt]}" "${__k2[norphan_fits]}" \
            "${__k2[ndropped]}" "${__k2[has_acq]}" "${__warn2}"
        printf '  epoch_sec=%s  span_rel=[%s, %s] s\n' \
            "${__k2[epoch_sec]}" "${__k2[tstart_rel]}" "${__k2[tend_rel]}"

        local __ov_frames_a=0 __ov_frames_b=0
        @calc -v __ov_sec,__ov0_a,__ov1_a,__ov_frames_a,__ov_frames_b \
            "_epoff = f64(${__k2[epoch_sec]}) - f64(${__k1[epoch_sec]}); \
             _a0 = f64(${__k1[tstart_rel]}); _a1 = f64(${__k1[tend_rel]}); \
             _b0 = f64(${__k2[tstart_rel]}) + _epoff; _b1 = f64(${__k2[tend_rel]}) + _epoff; \
             _o0 = max(_a0, _b0); _o1 = min(_a1, _b1); \
             _ov = max(0.0, _o1 - _o0); \
             _ov; _o0; _o1; _ov * ${__k1[fps]}; _ov * ${__k2[fps]}"
        printf 'Overlap: duration=%.6f s  A_rel=[%.6f, %.6f]  est_frames=(A:%.0f, B:%.0f)\n' \
            "${__ov_sec}" "${__ov0_a}" "${__ov1_a}" "${__ov_frames_a}" "${__ov_frames_b}"
        (( $(@calc -v __ov_ok "_ov > 0.0"; echo "${__ov_ok}") )) || true
    fi

    if [[ -n "${outreport}" ]]; then
        mkdir -p "$(dirname "${outreport}")" || return 1
        {
            echo "stream_a=${__st1}"
            echo "ncubes_a=${__k1[ncubes]}"
            echo "nframes_a=${__k1[nframes]}"
            echo "norphan_txt_a=${__k1[norphan_txt]}"
            echo "norphan_fits_a=${__k1[norphan_fits]}"
            echo "fps_a=${__k1[fps]}"
            echo "jitter_us_a=${__k1[jitter_us]}"
            echo "ndropped_a=${__k1[ndropped]}"
            echo "has_acq_a=${__k1[has_acq]}"
            echo "epoch_sec_a=${__k1[epoch_sec]}"
            echo "tstart_rel_a=${__k1[tstart_rel]}"
            echo "tend_rel_a=${__k1[tend_rel]}"
            if [[ -n "${dir2}" || -n "${stream2}" ]]; then
                echo "stream_b=${__k2[stream]:-${stream2}}"
                echo "ncubes_b=${__k2[ncubes]}"
                echo "nframes_b=${__k2[nframes]}"
                echo "norphan_txt_b=${__k2[norphan_txt]}"
                echo "norphan_fits_b=${__k2[norphan_fits]}"
                echo "fps_b=${__k2[fps]}"
                echo "ndropped_b=${__k2[ndropped]}"
                echo "has_acq_b=${__k2[has_acq]}"
                echo "epoch_sec_b=${__k2[epoch_sec]}"
                echo "tstart_rel_b=${__k2[tstart_rel]}"
                echo "tend_rel_b=${__k2[tend_rel]}"
                echo "overlap_sec=${__ov_sec}"
                echo "overlap_start_a_rel=${__ov0_a}"
                echo "overlap_end_a_rel=${__ov1_a}"
            fi
        } > "${outreport}" || return 1
    fi
    return 0
}

# Generate synthetic multi-cube telemetry datasets with known lag, modes, and CCA ground truth
mlt.tlmsynth() {
    @args 'outdir:str:output directory' \
          'nframes:int=4000:frames for stream A' \
          'fpsa:float=1000.0:stream A frame rate [Hz]' \
          'fpsb:float=1000.0:stream B frame rate [Hz]' \
          'nx:int=16:spatial width' \
          'ny:int=16:spatial height' \
          'nmodes:int=4:spatial rank (>=2)' \
          'lag_sec:float=0.00235:lag of B behind A [s] (B(t) ~ A(t - lag_sec))' \
          'rho1:float=0.90:canonical correlation 1' \
          'rho2:float=0.65:canonical correlation 2' \
          'ar1:float=0.85:AR(1) temporal correlation at stream A rate' \
          'noise:float=0.15:sensor white noise std' \
          'cubesize:int=2000:frames per output cube' \
          'ndrop:int=0:simulated dropped frames in stream A' \
          'acq_zero:int=0:set t_acq=0 in stream B' -- "$@"

    local -a __links=()
    local __sdir __smA __smB
    mkdir -p "${outdir}" || return 1
    __mlt_short_path __sdir "${outdir}" __links || { __mlt_cleanup __links; return 1; }
    __mlt_short_outpath __smA "${outdir}/truth_modesA.fits" __links || { __mlt_cleanup __links; return 1; }
    __mlt_short_outpath __smB "${outdir}/truth_modesB.fits" __links || { __mlt_cleanup __links; return 1; }

    local __p=$(( nx * ny ))
    local __k=${nmodes}
    (( __k < 2 )) && __k=2
    (( __k > __p )) && __k=${__p}
    local __nl=$(( 2 * __k ))

    local __na_raw=$(( nframes + ndrop ))
    local __nb=0 __mfine=0 __dtfine=0 __sub=16 __tstart_master=0 __t0_day=43200
    @calc -v __nb,__mfine,__dtfine,__tstart_master \
        "_fmax = max(${fpsa}, ${fpsb}); \
         _f_fine = ${__sub} * _fmax; \
         _dt_f = 1.0 / _f_fine; \
         _dur = ${__na_raw} / ${fpsa} + 2.0 * abs(${lag_sec}) + 0.5; \
         _mf = ceil(_dur * _f_fine) + 128; \
         _mf = _mf + (_mf % 2); \
         _nb = floor(${__na_raw} * ${fpsb} / ${fpsa}); \
         _t0m = abs(${lag_sec}) + 0.1; \
         _nb; _mf; _dt_f; _t0m"

    __nb=${__nb%.*}
    __mfine=${__mfine%.*}

    @calc "
        _mlt_Ua, _mlt_sva, _mlt_Vta = svd(randn(zeros(${__p}, ${__k}), 101));
        _mlt_Ub, _mlt_svb, _mlt_Vtb = svd(randn(zeros(${__p}, ${__k}), 202));
        _mlt_a_sub = pow(clamp(${ar1}, 0.0, 0.995), ${fpsa} * ${__dtfine});
        _mlt_w = randn(zeros(${__mfine}, ${__nl}), 303);
        _mlt_F = fft(_mlt_w, 0);
        _mlt_om = 2.0 * pi * linspace(0, ${__mfine} - 1, ${__mfine}) / ${__mfine};
        _mlt_den = cplx(1.0 - _mlt_a_sub * cos(_mlt_om), _mlt_a_sub * sin(_mlt_om));
        _mlt_H = sqrt(max(1.0 - _mlt_a_sub * _mlt_a_sub, 1e-6)) / _mlt_den;
        _mlt_FH = _mlt_F * _mlt_H;
        _mlt_Zc = ifft(_mlt_FH, 0);
        _mlt_Zf = real(_mlt_Zc);
        _mlt_zmu = imean(_mlt_Zf, 0);
        _mlt_zsd = max(istd(_mlt_Zf, 0), 1e-6);
        _mlt_ZfT = (_mlt_Zf - _mlt_zmu.T) / _mlt_zsd.T;
        _mlt_idxA_full = linspace(0, ${__na_raw} - 1, ${__na_raw});
        _mlt_tqa_full = ${__tstart_master} + _mlt_idxA_full / ${fpsa};
        _mlt_Za_raw = zeros(${__k}, ${__na_raw});
        for _mlt_m = 0:(${__k} - 1) {
            _mlt_col = _mlt_ZfT[*, _mlt_m];
            _mlt_Za_raw[_mlt_m, *] = lut(_mlt_tqa_full, _mlt_col, 0.0, ${__dtfine}).T;
        }
        _mlt_idxB = linspace(0, ${__nb} - 1, ${__nb});
        _mlt_tqb = ${__tstart_master} + _mlt_idxB / ${fpsb} - (${lag_sec});
        _mlt_Zb_sh = zeros(${__k}, ${__nb});
        _mlt_Zb_ind = zeros(${__k}, ${__nb});
        for _mlt_m = 0:(${__k} - 1) {
            _mlt_c1 = _mlt_ZfT[*, _mlt_m];
            _mlt_c2 = _mlt_ZfT[*, ${__k} + _mlt_m];
            _mlt_Zb_sh[_mlt_m, *] = lut(_mlt_tqb, _mlt_c1, 0.0, ${__dtfine}).T;
            _mlt_Zb_ind[_mlt_m, *] = lut(_mlt_tqb, _mlt_c2, 0.0, ${__dtfine}).T;
        }
        _mlt_Zb = _mlt_Zb_ind * 1.0;
        _mlt_r1 = clamp(${rho1}, -0.999, 0.999);
        _mlt_r2 = clamp(${rho2}, -0.999, 0.999);
        _mlt_Zb[0, *] = _mlt_r1 * _mlt_Zb_sh[0, *] + sqrt(1.0 - _mlt_r1 * _mlt_r1) * _mlt_Zb_ind[0, *];
        _mlt_Zb[1, *] = _mlt_r2 * _mlt_Zb_sh[1, *] + sqrt(1.0 - _mlt_r2 * _mlt_r2) * _mlt_Zb_ind[1, *];
        _mlt_scale = [sqrt(1.0 / (1.0 + 0.5 * _m)) for _m = 0:(${__k} - 1)];
        _mlt_Xa_raw = (_mlt_Ua * _mlt_scale.T) @ _mlt_Za_raw + ${noise} * randn(zeros(${__p}, ${__na_raw}), 404);
        _mlt_Xb = (_mlt_Ub * _mlt_scale.T) @ _mlt_Zb + ${noise} * randn(zeros(${__p}, ${__nb}), 505);
        _mlt_t0d = f64(${__t0_day});
        _mlt_ta_all = f64(zeros(6, ${__na_raw}));
        _mlt_ta_log = _mlt_t0d + f64(_mlt_idxA_full) / f64(${fpsa});
        _mlt_ta_acq = _mlt_ta_log - f64(0.00025);
        _mlt_ta_all[0, *] = _mlt_ta_acq.T;
        _mlt_ta_all[1, *] = _mlt_ta_log.T;
        _mlt_ta_all[2, *] = _mlt_ta_acq.T;
        _mlt_ta_all[3, *] = f64(_mlt_idxA_full).T;
        _mlt_ta_all[5, *] = f64(_mlt_idxA_full).T;
        _mlt_tb = f64(zeros(6, ${__nb}));
        _mlt_tb_log = _mlt_t0d + f64(_mlt_idxB) / f64(${fpsb});
        _mlt_tb_acq = (${acq_zero} != 0) ? f64(zeros(${__nb})) : (_mlt_tb_log - f64(0.00025));
        _mlt_tb[0, *] = ((${acq_zero} != 0) ? _mlt_tb_log : _mlt_tb_acq).T;
        _mlt_tb[1, *] = _mlt_tb_log.T;
        _mlt_tb[2, *] = _mlt_tb_acq.T;
        _mlt_tb[3, *] = f64(_mlt_idxB).T;
        _mlt_tb[5, *] = f64(_mlt_idxB).T;
        _mlt_mA3 = reshape(_mlt_Ua, ${nx}, ${ny}, ${__k});
        _mlt_mB3 = reshape(_mlt_Ub, ${nx}, ${ny}, ${__k});
    " || {
        __mlt_rm_images _mlt_Ua _mlt_sva _mlt_Vta _mlt_Ub _mlt_svb _mlt_Vtb _mlt_w _mlt_F \
            _mlt_om _mlt_den _mlt_H _mlt_FH _mlt_Zc _mlt_Zf _mlt_zmu _mlt_zsd _mlt_ZfT _mlt_idxA_full _mlt_tqa_full _mlt_Za_raw \
            _mlt_idxB _mlt_tqb _mlt_Zb_sh _mlt_Zb_ind _mlt_Zb _mlt_scale _mlt_Xa_raw _mlt_Xb \
            _mlt_ta_all _mlt_ta_log _mlt_ta_acq _mlt_tb _mlt_tb_log _mlt_tb_acq _mlt_mA3 _mlt_mB3 \
            _mlt_col
        __mlt_cleanup __links
        return 1
    }

    if (( ndrop > 0 && nframes >= 4 )); then
        local __d0=$(( nframes / 2 ))
        local __d1=$(( __d0 + ndrop ))
        @calc "
            _mlt_Xa = cat(1, _mlt_Xa_raw[*, 0:$(( __d0 - 1 ))], _mlt_Xa_raw[*, ${__d1}:$(( __na_raw - 1 ))]);
            _mlt_ta = cat(1, _mlt_ta_all[*, 0:$(( __d0 - 1 ))], _mlt_ta_all[*, ${__d1}:$(( __na_raw - 1 ))]);
        " || return 1
    else
        @calc "_mlt_Xa = _mlt_Xa_raw; _mlt_ta = _mlt_ta_all;" || return 1
    fi

    iofits.saveFITS _mlt_mA3 "${__smA}" >/dev/null || return 1
    iofits.saveFITS _mlt_mB3 "${__smB}" >/dev/null || return 1

    # Write Unix epoch 2026-01-01 00:00:00 UT = 1767225600 on timing matrices before saving.
    mem.imwritekwL _mlt_ta MLTEPOCH 1767225600 Synth_epoch_UT_s >/dev/null
    mem.imwritekwL _mlt_tb MLTEPOCH 1767225600 Synth_epoch_UT_s >/dev/null

    local __cs=${cubesize}
    (( __cs < 1 )) && __cs=${nframes}

    local __s0=0 __s1
    while (( __s0 < nframes )); do
        __s1=$(( __s0 + __cs - 1 ))
        (( __s1 >= nframes )) && __s1=$(( nframes - 1 ))
        @calc "
            _mlt_ca = reshape(_mlt_Xa[*, ${__s0}:${__s1}], ${nx}, ${ny}, $(( __s1 - __s0 + 1 )));
            _mlt_ct = f64(_mlt_ta[*, ${__s0}:${__s1}]);
        " || return 1
        mem.imwritekwL _mlt_ct MLTEPOCH 1767225600 Synth_epoch_UT_s >/dev/null
        mlt.tlmsave _mlt_ca _mlt_ct "${__sdir}" "streamA" >/dev/null || return 1
        __mlt_rm_images _mlt_ca _mlt_ct
        __s0=$(( __s1 + 1 ))
    done

    __s0=0
    while (( __s0 < __nb )); do
        __s1=$(( __s0 + __cs - 1 ))
        (( __s1 >= __nb )) && __s1=$(( __nb - 1 ))
        @calc "
            _mlt_cb = reshape(_mlt_Xb[*, ${__s0}:${__s1}], ${nx}, ${ny}, $(( __s1 - __s0 + 1 )));
            _mlt_ct = f64(_mlt_tb[*, ${__s0}:${__s1}]);
        " || return 1
        mem.imwritekwL _mlt_ct MLTEPOCH 1767225600 Synth_epoch_UT_s >/dev/null
        mlt.tlmsave _mlt_cb _mlt_ct "${__sdir}" "streamB" >/dev/null || return 1
        __mlt_rm_images _mlt_cb _mlt_ct
        __s0=$(( __s1 + 1 ))
    done

    cat > "${outdir}/truth_summary.txt" <<EOF
nframes_a=${nframes}
nframes_b=${__nb}
fpsa=${fpsa}
fpsb=${fpsb}
nx=${nx}
ny=${ny}
nmodes=${__k}
lag_sec=${lag_sec}
lag_ms=$(@calc -v __lm "${lag_sec} * 1000.0"; echo "${__lm}")
rho1=${rho1}
rho2=${rho2}
ar1=${ar1}
noise=${noise}
ndrop=${ndrop}
acq_zero=${acq_zero}
EOF

    __mlt_rm_images _mlt_Ua _mlt_sva _mlt_Vta _mlt_Ub _mlt_svb _mlt_Vtb _mlt_w _mlt_F \
        _mlt_om _mlt_den _mlt_H _mlt_FH _mlt_Zc _mlt_Zf _mlt_zmu _mlt_zsd _mlt_ZfT _mlt_idxA_full _mlt_tqa_full _mlt_Za_raw \
        _mlt_idxB _mlt_tqb _mlt_Zb_sh _mlt_Zb_ind _mlt_Zb _mlt_scale _mlt_Xa_raw _mlt_Xa _mlt_Xb \
        _mlt_ta_all _mlt_ta _mlt_ta_log _mlt_ta_acq _mlt_tb _mlt_tb_log _mlt_tb_acq \
        _mlt_mA3 _mlt_mB3 _mlt_col
    __mlt_cleanup __links
    printf 'mlt.tlmsynth: wrote %s (A:%d @ %sHz, B:%d @ %sHz, lag=%ss, rho=(%s,%s))\n' \
        "${outdir}" "${nframes}" "${fpsa}" "${__nb}" "${fpsb}" "${lag_sec}" "${rho1}" "${rho2}"
}

# Compute PCA modes, eigenvalues, and modal coefficient telemetry from FITS cubes
mlt.pca() {
    @args 'dir:str:telemetry directory or FITS file' \
          'stream:str=:stream prefix (auto if empty)' \
          'outdir:str:output directory' \
          'nmodes:int=50:number of PCA modes' \
          'method:str=auto:solver (auto|cov|gram|rsvd)' \
          'demean:int=1:subtract temporal mean (1=yes, 0=no)' \
          'tstart:str=:start time bound' \
          'tend:str=:end time bound' \
          'maxframes:int=0:maximum frames (0=all)' \
          'decim:int=1:temporal stride (>=1)' \
          'mask:str=:optional 2D mask FITS or image' \
          'roi:str=:optional ROI x0:x1,y0:y1' \
          'bin:int=1:spatial binning factor (>=1)' -- "$@"

    local -a __links=()
    local __sdir __sout __smask=""
    mkdir -p "${outdir}" || return 1
    __mlt_short_path __sdir "${dir}" __links || { __mlt_cleanup __links; return 1; }
    __mlt_short_path __sout "${outdir}" __links || { __mlt_cleanup __links; return 1; }

    local -a __largs=(".dir=${__sdir}" ".out=@L:_mlt_X" ".decim=${decim}" ".bin=${bin}" ".packed=1")
    [[ -n "${stream}" ]] && __largs+=(".stream=${stream}")
    [[ -n "${tstart}" ]] && __largs+=(".tstart=${tstart}")
    [[ -n "${tend}" ]] && __largs+=(".tend=${tend}")
    (( maxframes > 0 )) && __largs+=(".maxframes=${maxframes}")
    [[ -n "${roi}" ]] && __largs+=(".roi=${roi}")
    if [[ -n "${mask}" && "${mask}" != "none" ]]; then
        __mlt_short_path __smask "${mask}" __links || { __mlt_cleanup __links; return 1; }
        __largs+=(".mask=${__smask}")
    fi

    mlt.tlmload "${__largs[@]}" >/dev/null || { __mlt_cleanup __links; return 1; }

    local __p __n __nx __ny __masked __ep
    @calc -v __p,__n,__nx,__ny,__masked,__ep \
        "s._mlt_X.xsize; s._mlt_X.ysize; s._mlt_X.kw.NXOUT; s._mlt_X.kw.NYOUT; s._mlt_X.kw.MASKED; s._mlt_X.kw.MLTEPOCH"
    __p=${__p%.*}
    __n=${__n%.*}
    __nx=${__nx%.*}
    __ny=${__ny%.*}
    __masked=${__masked%.*}

    local __k=${nmodes}
    (( __k > __p )) && __k=${__p}
    local __nmax=$(( __n - (demean ? 1 : 0) ))
    (( __nmax < 1 )) && __nmax=1
    (( __k > __nmax )) && __k=${__nmax}

    local __sol="${method}"
    if [[ "${__sol}" == "auto" ]]; then
        if (( __n < __p && __n <= 4096 )); then
            __sol="gram"
        elif (( __p <= 8192 )); then
            __sol="cov"
        else
            __sol="rsvd"
        fi
    fi

    local __norm_n=$(( __n > 1 ? __n - 1 : 1 ))
    local __tot_var=0

    @calc -v __tot_var "
        _mlt_mu = imean(_mlt_X, 1);
        _mlt_Xc = (${demean} != 0) ? (_mlt_X - _mlt_mu) : (_mlt_X * 1.0);
        if \"${__sol}\" == \"cov\" {
            _mlt_C = (_mlt_Xc @ _mlt_Xc.T) / ${__norm_n};
            _mlt_wa, _mlt_Va = eigh(_mlt_C, ${__k});
            _mlt_w = max(_mlt_wa[-*], 0.0);
            _mlt_modes = _mlt_Va[*, -*];
            _mlt_tv = trace(_mlt_C);
        } elif \"${__sol}\" == \"gram\" {
            _mlt_G = (_mlt_Xc.T @ _mlt_Xc) / ${__norm_n};
            _mlt_wa, _mlt_Ua = eigh(_mlt_G, ${__k});
            _mlt_w = max(_mlt_wa[-*], 0.0);
            _mlt_U = _mlt_Ua[*, -*];
            _mlt_sv = sqrt(max(_mlt_w, 1e-30) * ${__norm_n});
            _mlt_modes = (_mlt_Xc @ _mlt_U) / _mlt_sv.T;
            _mlt_tv = trace(_mlt_G);
        } elif \"${__sol}\" == \"rsvd\" {
            _mlt_kr = min(${__k} + 16, min(${__p}, ${__n}));
            _mlt_Om = randn(zeros(${__n}, _mlt_kr), 42);
            _mlt_Y = _mlt_Xc @ _mlt_Om;
            _mlt_Q, _mlt_sv0, _mlt_Vt0 = svd(_mlt_Y);
            _mlt_Y = _mlt_Xc @ (_mlt_Xc.T @ _mlt_Q);
            _mlt_Q, _mlt_sv0, _mlt_Vt0 = svd(_mlt_Y);
            _mlt_B = _mlt_Q.T @ _mlt_Xc;
            _mlt_Ub, _mlt_sv, _mlt_Vtb = svd(_mlt_B);
            _mlt_QU = _mlt_Q @ _mlt_Ub;
            _mlt_modes = _mlt_QU[*, 0:$(( __k - 1 ))];
            _mlt_w = pow(_mlt_sv[0:$(( __k - 1 ))], 2) / ${__norm_n};
            _mlt_tv = itot(_mlt_Xc * _mlt_Xc) / ${__norm_n};
        } else {
            _mlt_tv = -1.0;
        }
        _mlt_ramp = 1.0 + ix(_mlt_modes, 0, 0) * 1e-4;
        _mlt_sgn = where(itot(_mlt_modes * _mlt_modes * _mlt_modes * _mlt_ramp, 0) >= 0.0, 1.0, -1.0);
        _mlt_modes = _mlt_modes * _mlt_sgn.T;
        _mlt_A = _mlt_modes.T @ _mlt_Xc;
        _mlt_tv
    " || {
        __mlt_rm_images _mlt_X _mlt_X_t _mlt_mu _mlt_Xc _mlt_C _mlt_wa _mlt_Va _mlt_w \
            _mlt_modes _mlt_G _mlt_Ua _mlt_U _mlt_sv _mlt_Om _mlt_Y _mlt_Q _mlt_sv0 _mlt_Vt0 \
            _mlt_B _mlt_Ub _mlt_Vtb _mlt_ramp _mlt_sgn _mlt_A
        __mlt_cleanup __links
        return 1
    }

    if (( __masked )); then
        iofits.loadfits "${__smask}" _mlt_m2d 2 >/dev/null || { __mlt_cleanup __links; return 1; }
        @calc "
            _mlt_modes3 = mlt_unmask(_mlt_modes, _mlt_m2d);
            _mlt_mean2d = mlt_unmask(_mlt_mu, _mlt_m2d);
        " || return 1
    else
        @calc "
            _mlt_modes3 = reshape(_mlt_modes, ${__nx}, ${__ny}, ${__k});
            _mlt_mean2d = reshape(_mlt_mu, ${__nx}, ${__ny});
        " || return 1
    fi

    local __f_modes __f_mean
    __mlt_short_outpath __f_modes "${outdir}/modes.fits" __links || { __mlt_cleanup __links; return 1; }
    __mlt_short_outpath __f_mean "${outdir}/mean.fits" __links || { __mlt_cleanup __links; return 1; }
    iofits.saveFITS _mlt_modes3 "${__f_modes}" >/dev/null || return 1
    iofits.saveFITS _mlt_mean2d "${__f_mean}" >/dev/null || return 1

    mlt.tlmsave _mlt_A _mlt_X_t "${__sout}" "coeffs" >/dev/null || return 1

    local -a __warr=()
    @calc -a __warr "_mlt_w"
    local __cum=0 __m=0
    {
        printf '# mode\teigval\tvar_frac\tcum_var_frac\n'
        for (( __m = 0; __m < __k; __m++ )); do
            local __ev="${__warr[${__m}]:-0}"
            local __vf __cf
            @calc -v __vf,__cf \
                "_tv = max(${__tot_var}, 1e-30); _v = ${__ev} / _tv; _c = ${__cum} + _v; _v; _c"
            __cum="${__cf}"
            printf '%d\t%.8g\t%.8g\t%.8g\n' "${__m}" "${__ev}" "${__vf}" "${__cf}"
        done
    } > "${outdir}/eigval.txt"

    cat > "${outdir}/pca_summary.txt" <<EOF
method=${__sol}
nframes=${__n}
npix=${__p}
nx=${__nx}
ny=${__ny}
nmodes=${__k}
demean=${demean}
total_variance=${__tot_var}
explained_frac=${__cum}
EOF

    __mlt_rm_images _mlt_X _mlt_X_t _mlt_mu _mlt_Xc _mlt_C _mlt_wa _mlt_Va _mlt_w \
        _mlt_modes _mlt_G _mlt_Ua _mlt_U _mlt_sv _mlt_Om _mlt_Y _mlt_Q _mlt_sv0 _mlt_Vt0 \
        _mlt_B _mlt_Ub _mlt_Vtb _mlt_QU _mlt_ramp _mlt_sgn _mlt_A _mlt_modes3 _mlt_mean2d _mlt_m2d
    __mlt_cleanup __links
    printf 'mlt.pca: method=%s P=%d N=%d modes=%d explained=%.4f -> %s\n' \
        "${__sol}" "${__p}" "${__n}" "${__k}" "${__cum}" "${outdir}"
}

# Project telemetry cubes onto saved PCA modes.fits + mean.fits
mlt.pcaproj() {
    @args 'dir:str:telemetry directory or FITS file' \
          'stream:str=:stream prefix (auto if empty)' \
          'modes_fits:str:modes.fits saved by mlt.pca' \
          'mean_fits:str:mean.fits saved by mlt.pca' \
          'outdir:str:output directory' \
          'tstart:str=:start time bound' \
          'tend:str=:end time bound' \
          'maxframes:int=0:maximum frames (0=all)' \
          'decim:int=1:temporal stride (>=1)' \
          'mask:str=:optional 2D mask FITS' \
          'roi:str=:optional ROI x0:x1,y0:y1' \
          'bin:int=1:spatial binning factor (>=1)' -- "$@"

    local -a __links=()
    local __sdir __sout __smodes __smean __smask=""
    mkdir -p "${outdir}" || return 1
    __mlt_short_path __sdir "${dir}" __links || { __mlt_cleanup __links; return 1; }
    __mlt_short_path __sout "${outdir}" __links || { __mlt_cleanup __links; return 1; }
    __mlt_short_path __smodes "${modes_fits}" __links || { __mlt_cleanup __links; return 1; }
    __mlt_short_path __smean "${mean_fits}" __links || { __mlt_cleanup __links; return 1; }

    local -a __largs=(".dir=${__sdir}" ".out=@L:_mlt_X" ".decim=${decim}" ".bin=${bin}" ".packed=1")
    [[ -n "${stream}" ]] && __largs+=(".stream=${stream}")
    [[ -n "${tstart}" ]] && __largs+=(".tstart=${tstart}")
    [[ -n "${tend}" ]] && __largs+=(".tend=${tend}")
    (( maxframes > 0 )) && __largs+=(".maxframes=${maxframes}")
    [[ -n "${roi}" ]] && __largs+=(".roi=${roi}")
    if [[ -n "${mask}" && "${mask}" != "none" ]]; then
        __mlt_short_path __smask "${mask}" __links || { __mlt_cleanup __links; return 1; }
        __largs+=(".mask=${__smask}")
    fi

    mlt.tlmload "${__largs[@]}" >/dev/null || { __mlt_cleanup __links; return 1; }
    iofits.loadfits "${__smodes}" _mlt_M3 2 >/dev/null || { __mlt_cleanup __links; return 1; }
    iofits.loadfits "${__smean}" _mlt_mu2 2 >/dev/null || { __mlt_cleanup __links; return 1; }

    local __p __n __masked
    @calc -v __p,__n,__masked "s._mlt_X.xsize; s._mlt_X.ysize; s._mlt_X.kw.MASKED"
    __p=${__p%.*}
    __n=${__n%.*}
    __masked=${__masked%.*}

    if (( __masked )); then
        iofits.loadfits "${__smask}" _mlt_m2d 2 >/dev/null || { __mlt_cleanup __links; return 1; }
        @calc "
            _mlt_M = mlt_mask(_mlt_M3, _mlt_m2d);
            _mlt_mu = mlt_mask(_mlt_mu2, _mlt_m2d);
        " || return 1
    else
        @calc "
            _mlt_M = reshape(_mlt_M3, ${__p}, -1);
            _mlt_mu = reshape(_mlt_mu2, ${__p}, 1);
        " || return 1
    fi

    local __res_rms=0 __exp_frac=0
    @calc -v __res_rms,__exp_frac "
        _mlt_Xc = _mlt_X - _mlt_mu;
        _mlt_A = _mlt_M.T @ _mlt_Xc;
        _mlt_et = itot(_mlt_Xc * _mlt_Xc);
        _mlt_ea = itot(_mlt_A * _mlt_A);
        _mlt_er = max(_mlt_et - _mlt_ea, 0.0);
        sqrt(_mlt_er / (${__p} * ${__n})); _mlt_ea / max(_mlt_et, 1e-30)
    " || return 1

    mlt.tlmsave _mlt_A _mlt_X_t "${__sout}" "coeffs" >/dev/null || return 1
    cat > "${outdir}/pcaproj_summary.txt" <<EOF
nframes=${__n}
npix=${__p}
residual_rms=${__res_rms}
explained_frac=${__exp_frac}
EOF

    __mlt_rm_images _mlt_X _mlt_X_t _mlt_M3 _mlt_mu2 _mlt_m2d _mlt_M _mlt_mu _mlt_Xc _mlt_A
    __mlt_cleanup __links
    printf 'mlt.pcaproj: N=%d res_rms=%.6g explained=%.4f -> %s\n' \
        "${__n}" "${__res_rms}" "${__exp_frac}" "${outdir}"
}

# Measure time latency between two telemetry streams (tau > 0 means B lags behind A)
mlt.latency() {
    @args 'dira:str:stream A directory' \
          'streama:str=:stream A prefix (auto if empty)' \
          'dirb:str:stream B directory' \
          'streamb:str=:stream B prefix (auto if empty)' \
          'outdir:str:output directory' \
          'maxlag_ms:float=50.0:coarse lag search half-window [ms]' \
          'fine_ms:float=2.0:fine continuous timestamp scan half-window [ms]' \
          'step_ms:float=0.05:fine scan step [ms]' \
          'nmodes:int=8:spatial projection modes per stream' \
          'whiten_order:int=8:causal AR whitening order (0=off)' \
          'nblocks:int=4:blocks for uncertainty and drift estimate' \
          'tstart:str=:start time bound' \
          'tend:str=:end time bound' \
          'maxframes:int=0:maximum frames from A (0=all)' \
          'tsrc:str=auto:timestamp source (auto|acq|log)' -- "$@"

    local -a __links=()
    local __sda __sdb
    mkdir -p "${outdir}" || return 1
    __mlt_short_path __sda "${dira}" __links || { __mlt_cleanup __links; return 1; }
    __mlt_short_path __sdb "${dirb}" __links || { __mlt_cleanup __links; return 1; }

    local -A __ka_idx=() __kb_idx=()
    local __use_ts="${tstart}" __use_te="${tend}"
    if [[ -z "${__use_ts}" && -z "${__use_te}" ]]; then
        __mlt_run_tlmindex "${dira}" "${streama}" __ka_idx || { __mlt_cleanup __links; return 1; }
        __mlt_run_tlmindex "${dirb}" "${streamb}" __kb_idx || { __mlt_cleanup __links; return 1; }
        local __ov_s=0 __ov0=0 __ov1=0
        @calc -v __ov_s,__ov0,__ov1 \
            "_epoff = f64(${__kb_idx[epoch_sec]}) - f64(${__ka_idx[epoch_sec]}); \
             _a0 = f64(${__ka_idx[tstart_rel]}); _a1 = f64(${__ka_idx[tend_rel]}); \
             _b0 = f64(${__kb_idx[tstart_rel]}) + _epoff; _b1 = f64(${__kb_idx[tend_rel]}) + _epoff; \
             _o0 = max(_a0, _b0); _o1 = min(_a1, _b1); \
             max(0.0, _o1 - _o0); _o0; _o1"
        if (( $(@calc -v __ok "${__ov_s} > 0.0"; echo "${__ok}") == 0 )); then
            echo "mlt.latency: streams have zero overlapping time span" >&2
            __mlt_cleanup __links
            return 1
        fi
        __use_ts="${__ov0}"
        __use_te="${__ov1}"
    fi

    local -a __la=(".dir=${__sda}" ".out=@L:_mlt_A" ".packed=1" ".tsrc=${tsrc}")
    [[ -n "${streama}" ]] && __la+=(".stream=${streama}")
    [[ -n "${__use_ts}" ]] && __la+=(".tstart=${__use_ts}")
    [[ -n "${__use_te}" ]] && __la+=(".tend=${__use_te}")
    (( maxframes > 0 )) && __la+=(".maxframes=${maxframes}")

    mlt.tlmload "${__la[@]}" >/dev/null || { __mlt_cleanup __links; return 1; }

    # Compute exact A time window (+ margin for lag search) for loading B.
    local __pa __na __ep_a __ta0 __ta1
    @calc -v __pa,__na,__ep_a,__ta0,__ta1 \
        "s._mlt_A.xsize; s._mlt_A.ysize; s._mlt_A.kw.MLTEPOCH; _mlt_A_t[0, 0]; _mlt_A_t[0, s._mlt_A.ysize - 1]"
    __pa=${__pa%.*}
    __na=${__na%.*}
    __ep_a=${__ep_a%.*}

    local __b_t0_sec __b_t1_sec
    @calc -v __b_t0_sec,__b_t1_sec \
        "_pad = (${maxlag_ms} + ${fine_ms} + 10.0) * 1e-3; \
         f64(${__ep_a}) + f64(${__ta0}) - _pad; \
         f64(${__ep_a}) + f64(${__ta1}) + _pad"

    local -a __lb=(".dir=${__sdb}" ".out=@L:_mlt_B" ".packed=1" ".tsrc=${tsrc}" \
                   ".tstart=${__b_t0_sec}" ".tend=${__b_t1_sec}")
    [[ -n "${streamb}" ]] && __lb+=(".stream=${streamb}")

    mlt.tlmload "${__lb[@]}" >/dev/null || {
        __mlt_rm_images _mlt_A _mlt_A_t
        __mlt_cleanup __links
        return 1
    }

    local __pb __nb
    @calc -v __pb,__nb "s._mlt_B.xsize; s._mlt_B.ysize"
    __pb=${__pb%.*}
    __nb=${__nb%.*}

    local __ka=${nmodes} __kb=${nmodes}
    (( __ka > __pa )) && __ka=${__pa}
    (( __ka > __na - 1 )) && __ka=$(( __na - 1 ))
    (( __ka < 1 )) && __ka=1
    (( __kb > __pb )) && __kb=${__pb}
    (( __kb > __nb - 1 )) && __kb=$(( __nb - 1 ))
    (( __kb < 1 )) && __kb=1

    # Reduce A and B spatially into compact modal matrices _mlt_Am (ka x Na) and _mlt_Bm (kb x Nb).
    @calc "
        _mlt_Xa = _mlt_A - imean(_mlt_A, 1);
        _mlt_Xb = _mlt_B - imean(_mlt_B, 1);
        if ${__pa} <= 2048 {
            _mlt_Ca = (_mlt_Xa @ _mlt_Xa.T) / max(${__na} - 1, 1);
            _mlt_wa, _mlt_Va = eigh(_mlt_Ca, ${__ka});
            _mlt_Ma = _mlt_Va[*, -*];
        } else {
            _mlt_kra = min(${__ka} + 8, min(${__pa}, ${__na}));
            _mlt_Qa, _mlt_sa0, _mlt_Vta0 = svd(_mlt_Xa @ randn(zeros(${__na}, _mlt_kra), 11));
            _mlt_Qa, _mlt_sa0, _mlt_Vta0 = svd(_mlt_Xa @ (_mlt_Xa.T @ _mlt_Qa));
            _mlt_Ua, _mlt_sa1, _mlt_Vta1 = svd(_mlt_Qa.T @ _mlt_Xa);
            _mlt_QaU = _mlt_Qa @ _mlt_Ua;
            _mlt_Ma = _mlt_QaU[*, 0:$(( __ka - 1 ))];
        }
        if ${__pb} <= 2048 {
            _mlt_Cb = (_mlt_Xb @ _mlt_Xb.T) / max(${__nb} - 1, 1);
            _mlt_wb, _mlt_Vb = eigh(_mlt_Cb, ${__kb});
            _mlt_Mb = _mlt_Vb[*, -*];
        } else {
            _mlt_krb = min(${__kb} + 8, min(${__pb}, ${__nb}));
            _mlt_Qb, _mlt_sb0, _mlt_Vtb0 = svd(_mlt_Xb @ randn(zeros(${__nb}, _mlt_krb), 22));
            _mlt_Qb, _mlt_sb0, _mlt_Vtb0 = svd(_mlt_Xb @ (_mlt_Xb.T @ _mlt_Qb));
            _mlt_Ub, _mlt_sb1, _mlt_Vtb1 = svd(_mlt_Qb.T @ _mlt_Xb);
            _mlt_QbU = _mlt_Qb @ _mlt_Ub;
            _mlt_Mb = _mlt_QbU[*, 0:$(( __kb - 1 ))];
        }
        _mlt_Am = _mlt_Ma.T @ _mlt_Xa;
        _mlt_Bm = _mlt_Mb.T @ _mlt_Xb;
    " || return 1

    local __fps_a=1000.0 __fps_b=1000.0
    @calc -v __fps_a,__fps_b "
        _dta = imedian(_mlt_A_t[0, 1:$(( __na - 1 ))] - _mlt_A_t[0, 0:$(( __na - 2 ))]);
        _dtb = imedian(_mlt_B_t[0, 1:$(( __nb - 1 ))] - _mlt_B_t[0, 0:$(( __nb - 2 ))]);
        1.0 / max(_dta, 1e-9); 1.0 / max(_dtb, 1e-9)
    "
    local __sync_mode="linear"
    if (( $(@calc -v __r "${__fps_b} > 1.3 * ${__fps_a}"; echo "${__r}") )); then
        __sync_mode="window"
    fi

    # Initial alignment at lag=0 for coarse lag discovery.
    mlt.tlmsync _mlt_A_t _mlt_Bm _mlt_B_t _mlt_Bs0 ".lag=0.0" ".mode=${__sync_mode}" >/dev/null || return 1
    local __i0 __i1 __nval
    @calc -v __i0,__i1,__nval "s._mlt_Bs0.kw.SYNCI0; s._mlt_Bs0.kw.SYNCI1; s._mlt_Bs0.kw.SYNCNV"
    __i0=${__i0%.*}
    __i1=${__i1%.*}
    __nval=${__nval%.*}
    if (( __nval < 16 )); then
        echo "mlt.latency: insufficient valid overlapping samples (${__nval}) at lag=0" >&2
        __mlt_cleanup __links
        return 1
    fi

    local __dt_a_ms __maxlag_samp
    @calc -v __dt_a_ms,__maxlag_samp \
        "_dt = 1000.0 / ${__fps_a}; _ml = max(2, min(${__nval} / 3, ceil(${maxlag_ms} / _dt))); _dt; _ml"
    __maxlag_samp=${__maxlag_samp%.*}

    local __tau_coarse_ms=0 __coarse_val=0 __coarse_fwhm=0
    @calc -v __tau_coarse_ms,__coarse_val,__coarse_fwhm "
        _mlt_Am0 = _mlt_Am[*, ${__i0}:${__i1}];
        _mlt_Bm0 = _mlt_Bs0[*, ${__i0}:${__i1}];
        if ${whiten_order} > 0 {
            _mlt_Aw0, _mlt_Bw0 = mlt_arwhiten(_mlt_Am0, _mlt_Bm0, order=${whiten_order}, ridge=1e-5);
        } else {
            _mlt_Aw0 = _mlt_Am0;
            _mlt_Bw0 = _mlt_Bm0;
        }
        _mlt_xc0 = mlt_xcov(_mlt_Aw0, _mlt_Bw0, maxlag=${__maxlag_samp}, norm=1);
        _pos0, _val0, _fwhm0 = mlt_peak1d(_mlt_xc0, -${__maxlag_samp} * ${__dt_a_ms}, ${__dt_a_ms});
        _pos0; _val0; _fwhm0
    " || return 1

    # Continuous fine timestamp scan around tau_coarse_ms.
    local __f0_sec __f1_sec __nfine
    @calc -v __f0_sec,__f1_sec,__nfine \
        "_tc = ${__tau_coarse_ms} * 1e-3; _fh = ${fine_ms} * 1e-3; _st = max(${step_ms} * 1e-3, 1e-6); \
         _nf = max(5, round((2.0 * _fh) / _st) + 1); \
         _tc - _fh; _tc + _fh; _nf"
    __nfine=${__nfine%.*}

    # Determine common overlap slice across endpoints [f0_sec, f1_sec] so all fine points match.
    mlt.tlmsync _mlt_A_t _mlt_Bm _mlt_B_t _mlt_Bedge ".lag=${__f0_sec}" ".mode=${__sync_mode}" >/dev/null || return 1
    local __e0a __e1a
    @calc -v __e0a,__e1a "s._mlt_Bedge.kw.SYNCI0; s._mlt_Bedge.kw.SYNCI1"
    mlt.tlmsync _mlt_A_t _mlt_Bm _mlt_B_t _mlt_Bedge ".lag=${__f1_sec}" ".mode=${__sync_mode}" >/dev/null || return 1
    local __e0b __e1b
    @calc -v __e0b,__e1b "s._mlt_Bedge.kw.SYNCI0; s._mlt_Bedge.kw.SYNCI1"

    local __s0 __s1 __ncommon
    @calc -v __s0,__s1,__ncommon \
        "_s0 = max(${__e0a}, ${__e0b}); _s1 = min(${__e1a}, ${__e1b}); _s0; _s1; max(0, _s1 - _s0 + 1)"
    __s0=${__s0%.*}
    __s1=${__s1%.*}
    __ncommon=${__ncommon%.*}
    if (( __ncommon < 16 )); then
        echo "mlt.latency: insufficient common samples (${__ncommon}) across fine lag window" >&2
        __mlt_cleanup __links
        return 1
    fi

    @calc "_mlt_fcurve = zeros(${__nfine}); _mlt_Asub = _mlt_Am[*, ${__s0}:${__s1}];" || return 1
    if (( whiten_order > 0 )); then
        @calc "_mlt_Awsub = mlt_arfilt1(_mlt_Asub, order=${whiten_order}, ridge=1e-5);" || return 1
    else
        @calc "_mlt_Awsub = _mlt_Asub;" || return 1
    fi

    local __fi __tau_j __f0_ms __actual_step_ms __actual_step_sec
    @calc -v __f0_ms,__actual_step_ms,__actual_step_sec \
        "_f0m = ${__f0_sec} * 1000.0; _sm = (2.0 * ${fine_ms}) / max(${__nfine} - 1, 1); _f0m; _sm; _sm * 1e-3"

    for (( __fi = 0; __fi < __nfine; __fi++ )); do
        @calc -v __tau_j "${__f0_sec} + ${__fi} * ${__actual_step_sec}"
        mlt.tlmsync _mlt_A_t _mlt_Bm _mlt_B_t _mlt_Bj ".lag=${__tau_j}" ".mode=${__sync_mode}" >/dev/null || return 1
        @calc "
            _mlt_Bsub = _mlt_Bj[*, ${__s0}:${__s1}];
            if ${whiten_order} > 0 {
                _mlt_Aw_j, _mlt_Bw_j = mlt_arwhiten(_mlt_Asub, _mlt_Bsub, order=${whiten_order}, ridge=1e-5);
            } else {
                _mlt_Aw_j = _mlt_Asub;
                _mlt_Bw_j = _mlt_Bsub;
            }
            _mlt_xcj = mlt_xcov(_mlt_Aw_j, _mlt_Bw_j, maxlag=1, norm=1);
            _mlt_fcurve[${__fi}] = _mlt_xcj[1];
        " || return 1
    done

    local __tau_ms=0 __peak_corr=0 __fwhm_ms=0
    @calc -v __tau_ms,__peak_corr,__fwhm_ms \
        "_fp, _fv, _fw = mlt_peak1d(_mlt_fcurve, ${__f0_ms}, ${__actual_step_ms}); _fp; _fv; _fw"

    local __tau_sec __tau_frames_a
    @calc -v __tau_sec,__tau_frames_a "${__tau_ms} * 1e-3; (${__tau_ms} * 1e-3) * ${__fps_a}"

    # Block repeatability / drift across nblocks contiguous segments centered near tau_sec.
    local __tau_err_ms=0 __drift_ppm=0
    local __nb_eff=${nblocks}
    (( __nb_eff > 1 && __ncommon / __nb_eff < 64 )) && __nb_eff=1
    if (( __nb_eff >= 2 )); then
        mlt.tlmsync _mlt_A_t _mlt_Bm _mlt_B_t _mlt_Bfin ".lag=${__tau_sec}" ".mode=${__sync_mode}" >/dev/null || return 1
        local __blen=$(( __ncommon / __nb_eff ))
        local __bmax=$(( 8 ))
        (( __bmax >= __blen / 4 )) && __bmax=2
        local -a __blags=() __bt=()
        local __bi
        for (( __bi = 0; __bi < __nb_eff; __bi++ )); do
            local __b0=$(( __s0 + __bi * __blen ))
            local __b1=$(( __b0 + __blen - 1 ))
            local __dlag_ms=0 __tmid=0
            @calc -v __dlag_ms,__tmid "
                _mlt_Ab = _mlt_Am[*, ${__b0}:${__b1}];
                _mlt_Bb = _mlt_Bfin[*, ${__b0}:${__b1}];
                if ${whiten_order} > 0 {
                    _mlt_Awb, _mlt_Bwb = mlt_arwhiten(_mlt_Ab, _mlt_Bb, order=min(${whiten_order}, ${__blen} / 8), ridge=1e-5);
                } else {
                    _mlt_Awb = _mlt_Ab; _mlt_Bwb = _mlt_Bb;
                }
                _mlt_xcb = mlt_xcov(_mlt_Awb, _mlt_Bwb, maxlag=${__bmax}, norm=1);
                _bp, _bv, _bfw = mlt_peak1d(_mlt_xcb, -${__bmax} * ${__dt_a_ms}, ${__dt_a_ms});
                _tm = 0.5 * (_mlt_A_t[0, ${__b0}] + _mlt_A_t[0, ${__b1}]);
                _bp; _tm
            "
            __blags+=("$(@calc -v __z "${__tau_ms} + ${__dlag_ms}"; echo "${__z}")")
            __bt+=("${__tmid}")
        done
        local __s_str="${__blags[*]}" __t_str="${__bt[*]}"
        @calc -v __tau_err_ms,__drift_ppm "
            _bl = [${__s_str// /, }];
            _bt = [${__t_str// /, }];
            _terr = istd(_bl) / sqrt(max(${__nb_eff} - 1, 1));
            _btm = imean(_bt); _blm = imean(_bl);
            _num = itot((_bt - _btm) * (_bl - _blm));
            _den = itot((_bt - _btm) * (_bt - _btm));
            _slope_ms_per_s = (_den > 1e-12) ? (_num / _den) : 0.0;
            _terr; _slope_ms_per_s * 1000.0
        "
    fi

    local __f_curve_fits
    __mlt_short_outpath __f_curve_fits "${outdir}/latency_curve.fits" __links || { __mlt_cleanup __links; return 1; }
    iofits.saveFITS _mlt_fcurve "${__f_curve_fits}" >/dev/null || return 1

    local -a __fc=()
    @calc -a __fc "_mlt_fcurve"
    {
        printf '# lag_ms\trms_corr\n'
        for (( __fi = 0; __fi < __nfine; __fi++ )); do
            local __lm_j
            @calc -v __lm_j "${__f0_ms} + ${__fi} * ${__actual_step_ms}"
            printf '%.6f\t%.8g\n' "${__lm_j}" "${__fc[${__fi}]:-0}"
        done
    } > "${outdir}/latency_curve.txt"

    cat > "${outdir}/latency_summary.txt" <<EOF
stream_a=${__ka_idx[stream]:-${streama:-auto}}
stream_b=${__kb_idx[stream]:-${streamb:-auto}}
tau_sec=${__tau_sec}
tau_ms=${__tau_ms}
tau_coarse_ms=${__tau_coarse_ms}
tau_frames_a=${__tau_frames_a}
tau_err_ms=${__tau_err_ms}
peak_corr=${__peak_corr}
fwhm_ms=${__fwhm_ms}
drift_ppm=${__drift_ppm}
fps_a=${__fps_a}
fps_b=${__fps_b}
n_overlap=${__ncommon}
sync_mode=${__sync_mode}
whiten_order=${whiten_order}
EOF

    __mlt_rm_images _mlt_A _mlt_A_t _mlt_B _mlt_B_t _mlt_Xa _mlt_Xb _mlt_Ca _mlt_wa _mlt_Va \
        _mlt_Ma _mlt_Qa _mlt_QaU _mlt_sa0 _mlt_Vta0 _mlt_Ua _mlt_sa1 _mlt_Vta1 _mlt_Cb _mlt_wb _mlt_Vb \
        _mlt_Mb _mlt_Qb _mlt_QbU _mlt_sb0 _mlt_Vtb0 _mlt_Ub _mlt_sb1 _mlt_Vtb1 _mlt_Am _mlt_Bm \
        _mlt_Bs0 _mlt_Am0 _mlt_Bm0 _mlt_Aw0 _mlt_Bw0 _mlt_xc0 _mlt_Bedge _mlt_fcurve \
        _mlt_Asub _mlt_Awsub _mlt_Bj _mlt_Bsub _mlt_Aw_j _mlt_Bw_j _mlt_xcj \
        _mlt_Bfin _mlt_Ab _mlt_Bb _mlt_Awb _mlt_Bwb _mlt_xcb
    __mlt_cleanup __links
    printf 'mlt.latency: tau=%.4f +/- %.4f ms (peak_corr=%.4f, fwhm=%.3f ms, N=%d) -> %s\n' \
        "${__tau_ms}" "${__tau_err_ms}" "${__peak_corr}" "${__fwhm_ms}" "${__ncommon}" "${outdir}"
}

# Canonical Correlation Analysis (CCA) between two telemetry streams with held-out temporal CV
mlt.cca() {
    @args 'dira:str:stream A directory' \
          'streama:str=:stream A prefix (auto if empty)' \
          'dirb:str:stream B directory' \
          'streamb:str=:stream B prefix (auto if empty)' \
          'outdir:str:output directory' \
          'nmodes:int=10:number of canonical pairs' \
          'pca_ka:int=50:PCA subspace rank for stream A' \
          'pca_kb:int=50:PCA subspace rank for stream B' \
          'rega:float=1e-3:relative ridge on A subspace' \
          'regb:float=1e-3:relative ridge on B subspace' \
          'lag:str=0.0:lag of B behind A [s], or auto' \
          'train_frac:float=0.7:contiguous temporal training fraction' \
          'tstart:str=:start time bound' \
          'tend:str=:end time bound' \
          'maxframes:int=0:maximum frames from A (0=all)' \
          'maska:str=:optional 2D mask for A' \
          'maskb:str=:optional 2D mask for B' \
          'bin:int=1:spatial binning factor (>=1)' \
          'tsrc:str=auto:timestamp source (auto|acq|log)' -- "$@"

    local -a __links=()
    local __sda __sdb __sout __sma="" __smb=""
    mkdir -p "${outdir}" || return 1
    __mlt_short_path __sda "${dira}" __links || { __mlt_cleanup __links; return 1; }
    __mlt_short_path __sdb "${dirb}" __links || { __mlt_cleanup __links; return 1; }
    __mlt_short_path __sout "${outdir}" __links || { __mlt_cleanup __links; return 1; }

    local __lag_sec="${lag}"
    if [[ "${lag}" == "auto" ]]; then
        local -a __lat_args=("${dira}" "${streama}" "${dirb}" "${streamb}" "${outdir}/latency" ".tsrc=${tsrc}")
        [[ -n "${tstart}" ]] && __lat_args+=(".tstart=${tstart}")
        [[ -n "${tend}" ]] && __lat_args+=(".tend=${tend}")
        (( maxframes > 0 )) && __lat_args+=(".maxframes=${maxframes}")
        mlt.latency "${__lat_args[@]}" || {
            __mlt_cleanup __links
            return 1
        }
        local -A __kl=()
        __mlt_read_kv "${outdir}/latency/latency_summary.txt" __kl
        __lag_sec="${__kl[tau_sec]:-0.0}"
    fi

    local -A __ka_idx=() __kb_idx=()
    local __use_ts="${tstart}" __use_te="${tend}"
    if [[ -z "${__use_ts}" && -z "${__use_te}" ]]; then
        __mlt_run_tlmindex "${dira}" "${streama}" __ka_idx || { __mlt_cleanup __links; return 1; }
        __mlt_run_tlmindex "${dirb}" "${streamb}" __kb_idx || { __mlt_cleanup __links; return 1; }
        local __ov_s=0 __ov0=0 __ov1=0
        @calc -v __ov_s,__ov0,__ov1 \
            "_epoff = f64(${__kb_idx[epoch_sec]}) - f64(${__ka_idx[epoch_sec]}); \
             _a0 = f64(${__ka_idx[tstart_rel]}); _a1 = f64(${__ka_idx[tend_rel]}); \
             _b0 = f64(${__kb_idx[tstart_rel]}) + _epoff - (${__lag_sec}); \
             _b1 = f64(${__kb_idx[tend_rel]}) + _epoff - (${__lag_sec}); \
             _o0 = max(_a0, _b0); _o1 = min(_a1, _b1); \
             max(0.0, _o1 - _o0); _o0; _o1"
        if (( $(@calc -v __ok "${__ov_s} > 0.0"; echo "${__ok}") == 0 )); then
            echo "mlt.cca: streams have zero overlapping span at lag=${__lag_sec}" >&2
            __mlt_cleanup __links
            return 1
        fi
        __use_ts="${__ov0}"
        __use_te="${__ov1}"
    fi

    local -a __la=(".dir=${__sda}" ".out=@L:_mlt_A" ".packed=1" ".bin=${bin}" ".tsrc=${tsrc}")
    [[ -n "${streama}" ]] && __la+=(".stream=${streama}")
    [[ -n "${__use_ts}" ]] && __la+=(".tstart=${__use_ts}")
    [[ -n "${__use_te}" ]] && __la+=(".tend=${__use_te}")
    (( maxframes > 0 )) && __la+=(".maxframes=${maxframes}")
    if [[ -n "${maska}" && "${maska}" != "none" ]]; then
        __mlt_short_path __sma "${maska}" __links || { __mlt_cleanup __links; return 1; }
        __la+=(".mask=${__sma}")
    fi
    mlt.tlmload "${__la[@]}" >/dev/null || { __mlt_cleanup __links; return 1; }

    local __pa __na __nxa __nya __maskeda __ep_a __ta0 __ta1
    @calc -v __pa,__na,__nxa,__nya,__maskeda,__ep_a,__ta0,__ta1 \
        "s._mlt_A.xsize; s._mlt_A.ysize; s._mlt_A.kw.NXOUT; s._mlt_A.kw.NYOUT; \
         s._mlt_A.kw.MASKED; s._mlt_A.kw.MLTEPOCH; _mlt_A_t[0, 0]; _mlt_A_t[0, s._mlt_A.ysize - 1]"
    __pa=${__pa%.*}
    __na=${__na%.*}
    __nxa=${__nxa%.*}
    __nya=${__nya%.*}
    __maskeda=${__maskeda%.*}
    __ep_a=${__ep_a%.*}

    local __b_t0 __b_t1
    @calc -v __b_t0,__b_t1 \
        "_pad = abs(${__lag_sec}) + 0.05; \
         f64(${__ep_a}) + f64(${__ta0}) + (${__lag_sec}) - 0.05; \
         f64(${__ep_a}) + f64(${__ta1}) + (${__lag_sec}) + 0.05"

    local -a __lb=(".dir=${__sdb}" ".out=@L:_mlt_B" ".packed=1" ".bin=${bin}" ".tsrc=${tsrc}" \
                   ".tstart=${__b_t0}" ".tend=${__b_t1}")
    [[ -n "${streamb}" ]] && __lb+=(".stream=${streamb}")
    if [[ -n "${maskb}" && "${maskb}" != "none" ]]; then
        __mlt_short_path __smb "${maskb}" __links || { __mlt_cleanup __links; return 1; }
        __lb+=(".mask=${__smb}")
    fi
    mlt.tlmload "${__lb[@]}" >/dev/null || { __mlt_cleanup __links; return 1; }

    local __pb __nb __nxb __nyb __maskedb
    @calc -v __pb,__nb,__nxb,__nyb,__maskedb \
        "s._mlt_B.xsize; s._mlt_B.ysize; s._mlt_B.kw.NXOUT; s._mlt_B.kw.NYOUT; s._mlt_B.kw.MASKED"
    __pb=${__pb%.*}
    __nb=${__nb%.*}
    __nxb=${__nxb%.*}
    __nyb=${__nyb%.*}
    __maskedb=${__maskedb%.*}

    local __fps_a=1000.0 __fps_b=1000.0
    @calc -v __fps_a,__fps_b "
        _dta = imedian(_mlt_A_t[0, 1:$(( __na - 1 ))] - _mlt_A_t[0, 0:$(( __na - 2 ))]);
        _dtb = imedian(_mlt_B_t[0, 1:$(( __nb - 1 ))] - _mlt_B_t[0, 0:$(( __nb - 2 ))]);
        1.0 / max(_dta, 1e-9); 1.0 / max(_dtb, 1e-9)
    "
    local __sync_mode="linear"
    if (( $(@calc -v __r "${__fps_b} > 1.3 * ${__fps_a}"; echo "${__r}") )); then
        __sync_mode="window"
    fi

    mlt.tlmsync _mlt_A_t _mlt_B _mlt_B_t _mlt_Bsync ".lag=${__lag_sec}" ".mode=${__sync_mode}" >/dev/null || return 1
    local __i0 __i1 __nval
    @calc -v __i0,__i1,__nval "s._mlt_Bsync.kw.SYNCI0; s._mlt_Bsync.kw.SYNCI1; s._mlt_Bsync.kw.SYNCNV"
    __i0=${__i0%.*}
    __i1=${__i1%.*}
    __nval=${__nval%.*}
    if (( __nval < 20 )); then
        echo "mlt.cca: insufficient aligned overlap frames (${__nval})" >&2
        __mlt_cleanup __links
        return 1
    fi

    # Train/test temporal split & ranks.
    local __n_tr=$(( __nval )) __n_te=0 __do_cv=0
    if (( $(@calc -v __c "${train_frac} > 0.05 && ${train_frac} < 0.999"; echo "${__c}") )); then
        __n_tr=$(@calc -v __x "floor(${__nval} * ${train_frac})"; echo "${__x%.*}")
        __n_te=$(( __nval - __n_tr ))
        if (( __n_tr >= 12 && __n_te >= 10 )); then
            __do_cv=1
        else
            __n_tr=${__nval}
            __n_te=0
        fi
    fi

    local __ka=${pca_ka} __kb=${pca_kb}
    (( __ka > __pa )) && __ka=${__pa}
    (( __ka > __n_tr - 1 )) && __ka=$(( __n_tr - 1 ))
    (( __ka < 1 )) && __ka=1
    (( __kb > __pb )) && __kb=${__pb}
    (( __kb > __n_tr - 1 )) && __kb=$(( __n_tr - 1 ))
    (( __kb < 1 )) && __kb=1
    local __kcca=${nmodes}
    (( __kcca > __ka )) && __kcca=${__ka}
    (( __kcca > __kb )) && __kcca=${__kb}

    @calc "
        _mlt_Xav = _mlt_A[*, ${__i0}:${__i1}];
        _mlt_Xbv = _mlt_Bsync[*, ${__i0}:${__i1}];
        _mlt_tv = f64(_mlt_A_t[*, ${__i0}:${__i1}]);
    " || return 1
    mem.imwritekwL _mlt_tv MLTEPOCH "${__ep_a}" Valid_window_epoch_UT_s >/dev/null

    local -a __rho_tr=() __rho_te=() __rho_full=()
    if (( __do_cv )); then
        local __ntm1=$(( __n_tr - 1 ))
        @calc "
            _mlt_Xa_tr = _mlt_Xav[*, 0:${__ntm1}];
            _mlt_Xb_tr = _mlt_Xbv[*, 0:${__ntm1}];
            _mlt_Xa_te = _mlt_Xav[*, ${__n_tr}:$(( __nval - 1 ))];
            _mlt_Xb_te = _mlt_Xbv[*, ${__n_tr}:$(( __nval - 1 ))];
            _mlt_mua_tr = imean(_mlt_Xa_tr, 1);
            _mlt_mub_tr = imean(_mlt_Xb_tr, 1);
            _mlt_Xac_tr = _mlt_Xa_tr - _mlt_mua_tr;
            _mlt_Xbc_tr = _mlt_Xb_tr - _mlt_mub_tr;
            _mlt_Xac_te = _mlt_Xa_te - _mlt_mua_tr;
            _mlt_Xbc_te = _mlt_Xb_te - _mlt_mub_tr;

            if ${__pa} <= 4096 {
                _mlt_Ca = (_mlt_Xac_tr @ _mlt_Xac_tr.T) / max(${__n_tr} - 1, 1);
                _mlt_wa_a, _mlt_Va_a = eigh(_mlt_Ca, ${__ka});
                _mlt_wa = max(_mlt_wa_a[-*], 0.0);
                _mlt_Va = _mlt_Va_a[*, -*];
            } else {
                _mlt_kra = min(${__ka} + 12, min(${__pa}, ${__n_tr}));
                _mlt_Qa, _mlt_s0, _mlt_V0 = svd(_mlt_Xac_tr @ randn(zeros(${__n_tr}, _mlt_kra), 61));
                _mlt_Qa, _mlt_s0, _mlt_V0 = svd(_mlt_Xac_tr @ (_mlt_Xac_tr.T @ _mlt_Qa));
                _mlt_Ua, _mlt_sva, _mlt_Vta = svd(_mlt_Qa.T @ _mlt_Xac_tr);
                _mlt_QaU = _mlt_Qa @ _mlt_Ua;
                _mlt_Va = _mlt_QaU[*, 0:$(( __ka - 1 ))];
                _mlt_wa = pow(_mlt_sva[0:$(( __ka - 1 ))], 2) / max(${__n_tr} - 1, 1);
            }

            if ${__pb} <= 4096 {
                _mlt_Cb = (_mlt_Xbc_tr @ _mlt_Xbc_tr.T) / max(${__n_tr} - 1, 1);
                _mlt_wb_a, _mlt_Vb_a = eigh(_mlt_Cb, ${__kb});
                _mlt_wb = max(_mlt_wb_a[-*], 0.0);
                _mlt_Vb = _mlt_Vb_a[*, -*];
            } else {
                _mlt_krb = min(${__kb} + 12, min(${__pb}, ${__n_tr}));
                _mlt_Qb, _mlt_s1, _mlt_V1 = svd(_mlt_Xbc_tr @ randn(zeros(${__n_tr}, _mlt_krb), 72));
                _mlt_Qb, _mlt_s1, _mlt_V1 = svd(_mlt_Xbc_tr @ (_mlt_Xbc_tr.T @ _mlt_Qb));
                _mlt_Ub, _mlt_svb, _mlt_Vtb = svd(_mlt_Qb.T @ _mlt_Xbc_tr);
                _mlt_QbU = _mlt_Qb @ _mlt_Ub;
                _mlt_Vb = _mlt_QbU[*, 0:$(( __kb - 1 ))];
                _mlt_wb = pow(_mlt_svb[0:$(( __kb - 1 ))], 2) / max(${__n_tr} - 1, 1);
            }

            _mlt_sa = sqrt(max(_mlt_wa, 1e-30) + ${rega} * max(_mlt_wa[0], 1e-12));
            _mlt_sb = sqrt(max(_mlt_wb, 1e-30) + ${regb} * max(_mlt_wb[0], 1e-12));
            _mlt_Za_tr = (_mlt_Va.T @ _mlt_Xac_tr) / _mlt_sa;
            _mlt_Zb_tr = (_mlt_Vb.T @ _mlt_Xbc_tr) / _mlt_sb;
            _mlt_Ktr = (_mlt_Za_tr @ _mlt_Zb_tr.T) / max(${__n_tr} - 1, 1);
            _mlt_Pu, _mlt_rtr_all, _mlt_Qt = svd(_mlt_Ktr);
            _mlt_rtr = _mlt_rtr_all[0:$(( __kcca - 1 ))];
            _mlt_Puk = _mlt_Pu[*, 0:$(( __kcca - 1 ))];
            _mlt_Pus = _mlt_Puk / _mlt_sa;
            _mlt_Wa_tr = _mlt_Va @ _mlt_Pus;
            _mlt_Qv = _mlt_Qt.T;
            _mlt_Qvk = _mlt_Qv[*, 0:$(( __kcca - 1 ))];
            _mlt_Qvs = _mlt_Qvk / _mlt_sb;
            _mlt_Wb_tr = _mlt_Vb @ _mlt_Qvs;
            _mlt_U_te = _mlt_Wa_tr.T @ _mlt_Xac_te;
            _mlt_V_te = _mlt_Wb_tr.T @ _mlt_Xbc_te;
            _mlt_U_ten = _mlt_U_te - imean(_mlt_U_te, 1);
            _mlt_V_ten = _mlt_V_te - imean(_mlt_V_te, 1);
            _mlt_num = itot(_mlt_U_ten * _mlt_V_ten, 1);
            _mlt_den = sqrt(max(itot(_mlt_U_ten * _mlt_U_ten, 1) * itot(_mlt_V_ten * _mlt_V_ten, 1), 1e-30));
            _mlt_rte = _mlt_num / _mlt_den;
        " || return 1
        @calc -a __rho_tr "_mlt_rtr"
        @calc -a __rho_te "_mlt_rte"
    fi

    # Full-window CCA fit for maximum-SNR spatial maps, loading patterns, and variates.
    @calc "
        _mlt_mua = imean(_mlt_Xav, 1);
        _mlt_mub = imean(_mlt_Xbv, 1);
        _mlt_Xac = _mlt_Xav - _mlt_mua;
        _mlt_Xbc = _mlt_Xbv - _mlt_mub;

        if ${__pa} <= 4096 {
            _mlt_Ca = (_mlt_Xac @ _mlt_Xac.T) / max(${__nval} - 1, 1);
            _mlt_wa_a, _mlt_Va_a = eigh(_mlt_Ca, ${__ka});
            _mlt_wa = max(_mlt_wa_a[-*], 0.0);
            _mlt_Va = _mlt_Va_a[*, -*];
        } else {
            _mlt_kra = min(${__ka} + 12, min(${__pa}, ${__nval}));
            _mlt_Qa, _mlt_s0, _mlt_V0 = svd(_mlt_Xac @ randn(zeros(${__nval}, _mlt_kra), 61));
            _mlt_Qa, _mlt_s0, _mlt_V0 = svd(_mlt_Xac @ (_mlt_Xac.T @ _mlt_Qa));
            _mlt_Ua, _mlt_sva, _mlt_Vta = svd(_mlt_Qa.T @ _mlt_Xac);
            _mlt_QaU = _mlt_Qa @ _mlt_Ua;
            _mlt_Va = _mlt_QaU[*, 0:$(( __ka - 1 ))];
            _mlt_wa = pow(_mlt_sva[0:$(( __ka - 1 ))], 2) / max(${__nval} - 1, 1);
        }

        if ${__pb} <= 4096 {
            _mlt_Cb = (_mlt_Xbc @ _mlt_Xbc.T) / max(${__nval} - 1, 1);
            _mlt_wb_a, _mlt_Vb_a = eigh(_mlt_Cb, ${__kb});
            _mlt_wb = max(_mlt_wb_a[-*], 0.0);
            _mlt_Vb = _mlt_Vb_a[*, -*];
        } else {
            _mlt_krb = min(${__kb} + 12, min(${__pb}, ${__nval}));
            _mlt_Qb, _mlt_s1, _mlt_V1 = svd(_mlt_Xbc @ randn(zeros(${__nval}, _mlt_krb), 72));
            _mlt_Qb, _mlt_s1, _mlt_V1 = svd(_mlt_Xbc @ (_mlt_Xbc.T @ _mlt_Qb));
            _mlt_Ub, _mlt_svb, _mlt_Vtb = svd(_mlt_Qb.T @ _mlt_Xbc);
            _mlt_QbU = _mlt_Qb @ _mlt_Ub;
            _mlt_Vb = _mlt_QbU[*, 0:$(( __kb - 1 ))];
            _mlt_wb = pow(_mlt_svb[0:$(( __kb - 1 ))], 2) / max(${__nval} - 1, 1);
        }

        _mlt_sa = sqrt(max(_mlt_wa, 1e-30) + ${rega} * max(_mlt_wa[0], 1e-12));
        _mlt_sb = sqrt(max(_mlt_wb, 1e-30) + ${regb} * max(_mlt_wb[0], 1e-12));
        _mlt_Za = (_mlt_Va.T @ _mlt_Xac) / _mlt_sa;
        _mlt_Zb = (_mlt_Vb.T @ _mlt_Xbc) / _mlt_sb;
        _mlt_K = (_mlt_Za @ _mlt_Zb.T) / max(${__nval} - 1, 1);
        _mlt_Pu, _mlt_rfull_all, _mlt_Qt = svd(_mlt_K);
        _mlt_rfull = _mlt_rfull_all[0:$(( __kcca - 1 ))];
        _mlt_Puk = _mlt_Pu[*, 0:$(( __kcca - 1 ))];
        _mlt_Pus = _mlt_Puk / _mlt_sa;
        _mlt_Wa = _mlt_Va @ _mlt_Pus;
        _mlt_Qv = _mlt_Qt.T;
        _mlt_Qvk = _mlt_Qv[*, 0:$(( __kcca - 1 ))];
        _mlt_Qvs = _mlt_Qvk / _mlt_sb;
        _mlt_Wb = _mlt_Vb @ _mlt_Qvs;
        _mlt_Uvar = _mlt_Wa.T @ _mlt_Xac;
        _mlt_Vvar = _mlt_Wb.T @ _mlt_Xbc;
        _mlt_Pata = (_mlt_Xac @ _mlt_Uvar.T) / max(${__nval} - 1, 1);
        _mlt_Patb = (_mlt_Xbc @ _mlt_Vvar.T) / max(${__nval} - 1, 1);
    " || return 1
    @calc -a __rho_full "_mlt_rfull"

    if (( __maskeda )); then
        iofits.loadfits "${__sma}" _mlt_m2a 2 >/dev/null || return 1
        @calc "_mlt_Wa3 = mlt_unmask(_mlt_Wa, _mlt_m2a); _mlt_Pa3 = mlt_unmask(_mlt_Pata, _mlt_m2a);" || return 1
    else
        @calc "_mlt_Wa3 = reshape(_mlt_Wa, ${__nxa}, ${__nya}, ${__kcca}); _mlt_Pa3 = reshape(_mlt_Pata, ${__nxa}, ${__nya}, ${__kcca});" || return 1
    fi

    if (( __maskedb )); then
        iofits.loadfits "${__smb}" _mlt_m2b 2 >/dev/null || return 1
        @calc "_mlt_Wb3 = mlt_unmask(_mlt_Wb, _mlt_m2b); _mlt_Pb3 = mlt_unmask(_mlt_Patb, _mlt_m2b);" || return 1
    else
        @calc "_mlt_Wb3 = reshape(_mlt_Wb, ${__nxb}, ${__nyb}, ${__kcca}); _mlt_Pb3 = reshape(_mlt_Patb, ${__nxb}, ${__nyb}, ${__kcca});" || return 1
    fi

    local __fa_m __fb_m __fa_p __fb_p
    __mlt_short_outpath __fa_m "${outdir}/cca_A_maps.fits" __links || return 1
    __mlt_short_outpath __fb_m "${outdir}/cca_B_maps.fits" __links || return 1
    __mlt_short_outpath __fa_p "${outdir}/cca_A_patterns.fits" __links || return 1
    __mlt_short_outpath __fb_p "${outdir}/cca_B_patterns.fits" __links || return 1
    iofits.saveFITS _mlt_Wa3 "${__fa_m}" >/dev/null || return 1
    iofits.saveFITS _mlt_Wb3 "${__fb_m}" >/dev/null || return 1
    iofits.saveFITS _mlt_Pa3 "${__fa_p}" >/dev/null || return 1
    iofits.saveFITS _mlt_Pb3 "${__fb_p}" >/dev/null || return 1

    mlt.tlmsave _mlt_Uvar _mlt_tv "${__sout}" "cca_U" >/dev/null || return 1
    mlt.tlmsave _mlt_Vvar _mlt_tv "${__sout}" "cca_V" >/dev/null || return 1

    local __mi
    {
        printf '# mode\trho_train\trho_test\trho_full\n'
        for (( __mi = 0; __mi < __kcca; __mi++ )); do
            local __rf="${__rho_full[${__mi}]:-0}"
            local __rt="${__rho_tr[${__mi}]:-${__rf}}"
            local __rv="${__rho_te[${__mi}]:-${__rf}}"
            printf '%d\t%.6f\t%.6f\t%.6f\n' "${__mi}" "${__rt}" "${__rv}" "${__rf}"
        done
    } > "${outdir}/cca_rho.txt"

    cat > "${outdir}/cca_summary.txt" <<EOF
stream_a=${__ka_idx[stream]:-${streama:-auto}}
stream_b=${__kb_idx[stream]:-${streamb:-auto}}
lag_sec=${__lag_sec}
nvalid=${__nval}
n_train=${__n_tr}
n_test=${__n_te}
pca_ka=${__ka}
pca_kb=${__kb}
nmodes=${__kcca}
rega=${rega}
regb=${regb}
rho_test_0=${__rho_te[0]:-${__rho_full[0]:-0}}
rho_full_0=${__rho_full[0]:-0}
EOF

    __mlt_rm_images _mlt_A _mlt_A_t _mlt_B _mlt_B_t _mlt_Bsync _mlt_Xav _mlt_Xbv _mlt_tv \
        _mlt_Xa_tr _mlt_Xb_tr _mlt_Xa_te _mlt_Xb_te _mlt_mua_tr _mlt_mub_tr _mlt_Xac_tr _mlt_Xbc_tr \
        _mlt_Xac_te _mlt_Xbc_te _mlt_Ca _mlt_wa_a _mlt_Va_a _mlt_wa _mlt_Va _mlt_kra _mlt_Qa \
        _mlt_s0 _mlt_V0 _mlt_Ua _mlt_sva _mlt_Vta _mlt_Cb _mlt_wb_a _mlt_Vb_a _mlt_wb _mlt_Vb \
        _mlt_krb _mlt_Qb _mlt_s1 _mlt_V1 _mlt_Ub _mlt_svb _mlt_Vtb _mlt_sa _mlt_sb _mlt_Za_tr \
        _mlt_Zb_tr _mlt_Ktr _mlt_Pu _mlt_rtr_all _mlt_rtr _mlt_Qt _mlt_Wa_tr _mlt_Wb_tr _mlt_U_te \
        _mlt_V_te _mlt_U_ten _mlt_V_ten _mlt_num _mlt_den _mlt_rte _mlt_mua _mlt_mub _mlt_Xac \
        _mlt_Xbc _mlt_Za _mlt_Zb _mlt_K _mlt_rfull_all _mlt_rfull _mlt_Wa _mlt_Wb _mlt_Uvar \
        _mlt_Vvar _mlt_Pata _mlt_Patb _mlt_m2a _mlt_m2b _mlt_Wa3 _mlt_Wb3 _mlt_Pa3 _mlt_Pb3 \
        _mlt_QaU _mlt_QbU _mlt_Puk _mlt_Pus _mlt_Qv _mlt_Qvk _mlt_Qvs
    __mlt_cleanup __links
    printf 'mlt.cca: N=%d (%d train, %d test) lag=%ss rho_test[0]=%s rho_full[0]=%s -> %s\n' \
        "${__nval}" "${__n_tr}" "${__n_te}" "${__lag_sec}" \
        "${__rho_te[0]:-NA}" "${__rho_full[0]:-0}" "${outdir}"
}

# Compute Welch PSD and optional two-stream magnitude-squared coherence + cross-phase
mlt.psd() {
    @args 'dir:str:telemetry directory' \
          'stream:str=:stream prefix (auto if empty)' \
          'outdir:str:output directory' \
          'seglen:int=1024:Welch FFT segment length' \
          'overlap:float=0.5:segment overlap fraction [0..0.9]' \
          'dir2:str=:optional second telemetry directory for coherence' \
          'stream2:str=:optional second stream prefix' \
          'lag:float=0.0:lag [s] when aligning second stream' \
          'tstart:str=:start time bound' \
          'tend:str=:end time bound' \
          'maxframes:int=0:maximum frames (0=all)' -- "$@"

    local -a __links=()
    local __sdir __sd2
    mkdir -p "${outdir}" || return 1
    __mlt_short_path __sdir "${dir}" __links || { __mlt_cleanup __links; return 1; }

    local -A __k1=()
    __mlt_run_tlmindex "${dir}" "${stream}" __k1 || { __mlt_cleanup __links; return 1; }

    local -a __la=(".dir=${__sdir}" ".out=@L:_mlt_A" ".packed=1")
    [[ -n "${stream}" ]] && __la+=(".stream=${stream}")
    [[ -n "${tstart}" ]] && __la+=(".tstart=${tstart}")
    [[ -n "${tend}" ]] && __la+=(".tend=${tend}")
    (( maxframes > 0 )) && __la+=(".maxframes=${maxframes}")
    mlt.tlmload "${__la[@]}" >/dev/null || { __mlt_cleanup __links; return 1; }

    local __pa __na __ep_a __ta0 __ta1
    @calc -v __pa,__na,__ep_a,__ta0,__ta1 \
        "s._mlt_A.xsize; s._mlt_A.ysize; s._mlt_A.kw.MLTEPOCH; _mlt_A_t[0, 0]; _mlt_A_t[0, s._mlt_A.ysize - 1]"
    __pa=${__pa%.*}
    __na=${__na%.*}
    __ep_a=${__ep_a%.*}

    local __two=0 __s0=0 __s1=$(( __na - 1 )) __nv=${__na}
    if [[ -n "${dir2}" || -n "${stream2}" ]]; then
        local __d2="${dir2:-${dir}}"
        __mlt_short_path __sd2 "${__d2}" __links || { __mlt_cleanup __links; return 1; }
        local __b0 __b1
        @calc -v __b0,__b1 "f64(${__ep_a}) + f64(${__ta0}) + (${lag}) - 0.05; f64(${__ep_a}) + f64(${__ta1}) + (${lag}) + 0.05"
        local -a __lb=(".dir=${__sd2}" ".out=@L:_mlt_B" ".packed=1" ".tstart=${__b0}" ".tend=${__b1}")
        [[ -n "${stream2}" ]] && __lb+=(".stream=${stream2}")
        mlt.tlmload "${__lb[@]}" >/dev/null || { __mlt_cleanup __links; return 1; }
        mlt.tlmsync _mlt_A_t _mlt_B _mlt_B_t _mlt_Bsync ".lag=${lag}" ".mode=linear" >/dev/null || return 1
        local __i0 __i1
        @calc -v __i0,__i1,__nv "s._mlt_Bsync.kw.SYNCI0; s._mlt_Bsync.kw.SYNCI1; s._mlt_Bsync.kw.SYNCNV"
        __s0=${__i0%.*}
        __s1=${__i1%.*}
        __nv=${__nv%.*}
        __two=1
    fi

    local __sl=${seglen}
    (( __sl > __nv )) && __sl=${__nv}
    (( __sl % 2 != 0 && __sl > 2 )) && __sl=$(( __sl - 1 ))
    local __hop
    __hop=$(@calc -v __h "max(1, round(${__sl} * (1.0 - clamp(${overlap}, 0.0, 0.9))))"; echo "${__h%.*}")
    local __nseg=$(( (__nv - __sl) / __hop + 1 ))
    (( __nseg < 1 )) && __nseg=1
    local __nfreq=$(( __sl / 2 + 1 ))
    local __fps="${__k1[fps]}"

    @calc "
        _mlt_win = (0.5 - 0.5 * cos(2.0 * pi * linspace(0, ${__sl} - 1, ${__sl}) / max(${__sl} - 1, 1))).T;
        _mlt_w2 = itot(_mlt_win * _mlt_win);
        _mlt_scale_one = 2.0 / (${__fps} * _mlt_w2 * ${__nseg});
        _mlt_pxx = zeros(${__pa}, ${__nfreq});
        for _mlt_sg = 0:$(( __nseg - 1 )) {
            _mlt_a = ${__s0} + _mlt_sg * ${__hop};
            _mlt_b = _mlt_a + ${__sl} - 1;
            _mlt_seg1 = _mlt_A[*, _mlt_a:_mlt_b];
            _mlt_Ff1 = fft((_mlt_seg1 - imean(_mlt_seg1, 1)) * _mlt_win, 1);
            _mlt_F1 = _mlt_Ff1[*, 0:$(( __nfreq - 1 ))];
            _mlt_pxx += (real(_mlt_F1) * real(_mlt_F1) + imag(_mlt_F1) * imag(_mlt_F1)) * _mlt_scale_one;
        }
        _mlt_pxx[*, 0] *= 0.5;
        if (${__sl} % 2) == 0 {
            _mlt_pxx[*, $(( __nfreq - 1 ))] *= 0.5;
        }
        _mlt_freq = linspace(0, ${__nfreq} - 1, ${__nfreq}) * (${__fps} / ${__sl});
        _mlt_pxx_mean = imean(_mlt_pxx, 0);
    " || return 1

    if (( __two )); then
        local __pb
        __pb=$(@calc -v __x "s._mlt_Bsync.xsize"; echo "${__x%.*}")
        if (( __pb != __pa )); then
            echo "mlt.psd: pixel count mismatch A=${__pa} vs B=${__pb} for per-channel coherence" >&2
            __mlt_cleanup __links
            return 1
        fi
        @calc "
            _mlt_pyy = zeros(${__pa}, ${__nfreq});
            _mlt_pxy_re = zeros(${__pa}, ${__nfreq});
            _mlt_pxy_im = zeros(${__pa}, ${__nfreq});
            for _mlt_sg = 0:$(( __nseg - 1 )) {
                _mlt_a = ${__s0} + _mlt_sg * ${__hop};
                _mlt_b = _mlt_a + ${__sl} - 1;
                _mlt_seg1 = _mlt_A[*, _mlt_a:_mlt_b];
                _mlt_seg2 = _mlt_Bsync[*, _mlt_a:_mlt_b];
                _mlt_Ff1 = fft((_mlt_seg1 - imean(_mlt_seg1, 1)) * _mlt_win, 1);
                _mlt_Ff2 = fft((_mlt_seg2 - imean(_mlt_seg2, 1)) * _mlt_win, 1);
                _mlt_F1 = _mlt_Ff1[*, 0:$(( __nfreq - 1 ))];
                _mlt_F2 = _mlt_Ff2[*, 0:$(( __nfreq - 1 ))];
                _mlt_pxy = _mlt_F1 * conj(_mlt_F2);
                _mlt_pyy += (real(_mlt_F2) * real(_mlt_F2) + imag(_mlt_F2) * imag(_mlt_F2)) * _mlt_scale_one;
                _mlt_pxy_re += real(_mlt_pxy) * _mlt_scale_one;
                _mlt_pxy_im += imag(_mlt_pxy) * _mlt_scale_one;
            }
            _mlt_pyy[*, 0] *= 0.5;
            _mlt_pxy_re[*, 0] *= 0.5;
            _mlt_pxy_im[*, 0] *= 0.5;
            if (${__sl} % 2) == 0 {
                _mlt_pyy[*, $(( __nfreq - 1 ))] *= 0.5;
                _mlt_pxy_re[*, $(( __nfreq - 1 ))] *= 0.5;
                _mlt_pxy_im[*, $(( __nfreq - 1 ))] *= 0.5;
            }
            _mlt_coh = (_mlt_pxy_re * _mlt_pxy_re + _mlt_pxy_im * _mlt_pxy_im) / max(_mlt_pxx * _mlt_pyy, 1e-30);
            _mlt_phase = atan2(_mlt_pxy_im, _mlt_pxy_re);
            _mlt_coh_mean = imean(_mlt_coh, 0);
            _mlt_phase_mean = imean(_mlt_phase, 0);
        " || return 1
    fi

    local __f_psd __f_frq __f_coh __f_phs
    __mlt_short_outpath __f_psd "${outdir}/psd.fits" __links || return 1
    __mlt_short_outpath __f_frq "${outdir}/freq.fits" __links || return 1
    iofits.saveFITS _mlt_pxx "${__f_psd}" >/dev/null || return 1
    iofits.saveFITS _mlt_freq "${__f_frq}" >/dev/null || return 1
    if (( __two )); then
        __mlt_short_outpath __f_coh "${outdir}/coherence.fits" __links || return 1
        __mlt_short_outpath __f_phs "${outdir}/phase.fits" __links || return 1
        iofits.saveFITS _mlt_coh "${__f_coh}" >/dev/null || return 1
        iofits.saveFITS _mlt_phase "${__f_phs}" >/dev/null || return 1
    fi

    local -a __fa=() __pa_arr=() __ca_arr=() __pha_arr=()
    @calc -a __fa "_mlt_freq"
    @calc -a __pa_arr "_mlt_pxx_mean"
    if (( __two )); then
        @calc -a __ca_arr "_mlt_coh_mean"
        @calc -a __pha_arr "_mlt_phase_mean"
    fi

    local __fi
    {
        if (( __two )); then
            printf '# freq_hz\tpsd_mean\tcoherence_mean\tphase_rad\n'
            for (( __fi = 0; __fi < __nfreq; __fi++ )); do
                printf '%.6f\t%.8g\t%.6f\t%.6f\n' \
                    "${__fa[${__fi}]}" "${__pa_arr[${__fi}]}" \
                    "${__ca_arr[${__fi}]}" "${__pha_arr[${__fi}]}"
            done
        else
            printf '# freq_hz\tpsd_mean\n'
            for (( __fi = 0; __fi < __nfreq; __fi++ )); do
                printf '%.6f\t%.8g\n' "${__fa[${__fi}]}" "${__pa_arr[${__fi}]}"
            done
        fi
    } > "${outdir}/psd_mean.txt"

    __mlt_rm_images _mlt_A _mlt_A_t _mlt_B _mlt_B_t _mlt_Bsync _mlt_win _mlt_pxx _mlt_seg1 \
        _mlt_seg2 _mlt_Ff1 _mlt_Ff2 _mlt_F1 _mlt_F2 _mlt_freq _mlt_pxx_mean _mlt_pyy _mlt_pxy \
        _mlt_pxy_re _mlt_pxy_im _mlt_coh _mlt_phase _mlt_coh_mean _mlt_phase_mean
    __mlt_cleanup __links
    printf 'mlt.psd: N=%d seglen=%d nseg=%d fps=%s Hz -> %s\n' \
        "${__nv}" "${__sl}" "${__nseg}" "${__fps}" "${outdir}"
}


# Closed-loop AO modal PSD, ETF, optical-gain, and optimal integrator gain analysis
mlt.aoloop() {
    @args 'dir:str:telemetry directory or WFS modal FITS' \
          'outdir:str:output directory' \
          'loop:int=2:CACAO loop number (1 or 2)' \
          'fs_hz:float=2000.0:loop sampling rate [Hz]' \
          'latency_fr:float=2.0:total loop delay [frames]' \
          'loopgain:float=0.35:fallback scalar loop gain' \
          'loopmult:float=0.98:fallback scalar leak mult' \
          'mgain:str=:optional per-mode gain FITS' \
          'mmult:str=:optional per-mode leak FITS' \
          'selfrm_cal:str=:optional calibration selfRM FITS' \
          'selfrm_sky:str=:optional on-sky selfRM FITS' \
          'est_ogain:int=0:estimate optical gain from DM steps if selfRM absent' \
          'nseg:int=512:Welch FFT window length' \
          'wfs_stream:str=:override WFS modeval stream name or FITS' \
          'dm_stream:str=:override DM modeval stream name or FITS' \
          'ol_stream:str=:override OL modeval stream name or FITS' \
          'recompute_ol:int=1:recompute pseudo-OL modes from WFS+DM' -- "$@"

    mkdir -p "${outdir}" || return 1
    local -a __links=()
    local __s_out __s_rep __m_wfs __m_dm __m_ol=""
    local __s_mg="" __s_mm="" __s_cal="" __s_sky=""

    __mlt_short_path __s_out "${outdir}" __links || { __mlt_cleanup __links; return 1; }
    __mlt_short_outpath __s_rep "${outdir}/aoopt_summary.txt" __links || {
        __mlt_cleanup __links; return 1
    }

    if [[ -d "${dir}" ]]; then
        local __sdir __ws="${wfs_stream:-aol${loop}_modevalWFS}"
        local __ds="${dm_stream:-aol${loop}_modevalDM}"
        __mlt_short_path __sdir "${dir}" __links || { __mlt_cleanup __links; return 1; }
        mlt.tlmload ".dir=${__sdir}" ".stream=${__ws}" ".out=@L:_mlt_awfs" ".packed=ON" >/dev/null || {
            __mlt_rm_images _mlt_awfs _mlt_awfs_t; __mlt_cleanup __links; return 1
        }
        mlt.tlmload ".dir=${__sdir}" ".stream=${__ds}" ".out=@L:_mlt_adm" ".packed=ON" >/dev/null || {
            __mlt_rm_images _mlt_awfs _mlt_awfs_t _mlt_adm _mlt_adm_t
            __mlt_cleanup __links; return 1
        }
        __m_wfs="_mlt_awfs"
        __m_dm="_mlt_adm"
        if (( ! recompute_ol )) && [[ -n "${ol_stream}" ]]; then
            if mlt.tlmload ".dir=${__sdir}" ".stream=${ol_stream}" \
                ".out=@L:_mlt_aol" ".packed=ON" >/dev/null 2>&1; then
                __m_ol="_mlt_aol"
            fi
        fi
    else
        __mlt_short_path __m_wfs "${wfs_stream:-${dir}}" __links || {
            __mlt_cleanup __links; return 1
        }
        __mlt_short_path __m_dm "${dm_stream}" __links || { __mlt_cleanup __links; return 1; }
        [[ -n "${ol_stream}" ]] && __mlt_short_path __m_ol "${ol_stream}" __links
    fi

    [[ -n "${mgain}" && -f "${mgain}" ]] && __mlt_short_path __s_mg "${mgain}" __links
    [[ -n "${mmult}" && -f "${mmult}" ]] && __mlt_short_path __s_mm "${mmult}" __links
    [[ -n "${selfrm_cal}" && -f "${selfrm_cal}" ]] && \
        __mlt_short_path __s_cal "${selfrm_cal}" __links
    [[ -n "${selfrm_sky}" && -f "${selfrm_sky}" ]] && \
        __mlt_short_path __s_sky "${selfrm_sky}" __links

    local __est_on="OFF" __rec_on="ON"
    (( est_ogain )) && __est_on="ON"
    (( ! recompute_ol )) && __rec_on="OFF"

    local -a __cmd=(
        mlt.aoopt
        ".mval_wfs=${__m_wfs}"
        ".mval_dm=${__m_dm}"
        ".outdir=${__s_out}"
        ".outreport=${__s_rep}"
        ".fs_hz=${fs_hz}"
        ".latency_fr=${latency_fr}"
        ".loopgain=${loopgain}"
        ".loopmult=${loopmult}"
        ".est_ogain=${__est_on}"
        ".recompute_ol=${__rec_on}"
        ".nseg=${nseg}"
    )
    [[ -n "${__m_ol}" ]] && __cmd+=(".mval_ol=${__m_ol}")
    [[ -n "${__s_mg}" ]] && __cmd+=(".mgain=${__s_mg}")
    [[ -n "${__s_mm}" ]] && __cmd+=(".mmult=${__s_mm}")
    [[ -n "${__s_cal}" ]] && __cmd+=(".selfrm_cal=${__s_cal}")
    [[ -n "${__s_sky}" ]] && __cmd+=(".selfrm_sky=${__s_sky}")

    "${__cmd[@]}"
    local __rc=$?
    __mlt_rm_images _mlt_awfs _mlt_awfs_t _mlt_adm _mlt_adm_t _mlt_aol _mlt_aol_t
    __mlt_cleanup __links
    return ${__rc}
}

# Polychromatic AO PSF reconstruction from modal residual telemetry & DM modes
mlt.psfr() {
    @args 'mval:str:telemetry directory or modal FITS' \
          'modes_dm:str:CMmodesDM.fits path' \
          'outdir:str:output directory' \
          'stream:str=aol2_modevalWFS:modal stream prefix when mval is a dir' \
          'method:str=hybrid:method (hybrid|short_exp|otf)' \
          'lambda_um:float=1.65:central wavelength [um]' \
          'bw_frac:float=0.18:fractional filter bandwidth' \
          'nlambda:int=3:polychromatic slices' \
          'pixscale_ld:float=2.50:detector pixels per lambda/D' \
          'nwf:int=256:pupil grid size [pix]' \
          'ncam:int=256:detector size [pix]' \
          'tel_diam_m:float=8.2:telescope diameter [m]' \
          'pup_diam:float=0.9375:clear pupil diameter fraction' \
          'cobs:float=0.30:central obscuration ratio' \
          'dm_diam:float=0.9986413:DM span fraction of nwf' \
          'if_coupling:float=0.15:DM inter-actuator coupling' \
          'opd_scale:float=2.0:OPD [um] per stroke unit' \
          'seeing_arcsec:float=0.60:seeing FWHM @ 0.50 um [arcsec]' \
          'pup:str=:optional 2D pupil amplitude FITS' \
          'ncpa:str=:optional 2D static NCPA OPD [um] FITS' \
          'ogain:str=:optional per-mode optical gain FITS' \
          'ref_psf:str=:optional measured reference PSF FITS' \
          'decim:int=1:temporal decimation stride (>=1)' \
          'maxframes:int=0:max frames after decimation (0=all)' -- "$@"

    mkdir -p "${outdir}" || return 1
    local -a __links=()
    local __s_out __s_rep __s_modes __m_in
    local __s_pup="" __s_ncpa="" __s_og="" __s_ref=""

    __mlt_short_path __s_out "${outdir}" __links || { __mlt_cleanup __links; return 1; }
    __mlt_short_outpath __s_rep "${outdir}/psfrecon_summary.txt" __links || {
        __mlt_cleanup __links; return 1
    }
    __mlt_short_path __s_modes "${modes_dm}" __links || { __mlt_cleanup __links; return 1; }

    if [[ -d "${mval}" ]]; then
        local __sdir
        __mlt_short_path __sdir "${mval}" __links || { __mlt_cleanup __links; return 1; }
        mlt.tlmload ".dir=${__sdir}" ".stream=${stream}" ".out=@L:_mlt_pmv" ".packed=ON" >/dev/null || {
            __mlt_rm_images _mlt_pmv _mlt_pmv_t; __mlt_cleanup __links; return 1
        }
        __m_in="_mlt_pmv"
    else
        __mlt_short_path __m_in "${mval}" __links || { __mlt_cleanup __links; return 1; }
    fi

    [[ -n "${pup}" && -f "${pup}" ]] && __mlt_short_path __s_pup "${pup}" __links
    [[ -n "${ncpa}" && -f "${ncpa}" ]] && __mlt_short_path __s_ncpa "${ncpa}" __links
    [[ -n "${ogain}" && -f "${ogain}" ]] && __mlt_short_path __s_og "${ogain}" __links
    [[ -n "${ref_psf}" && -f "${ref_psf}" ]] && __mlt_short_path __s_ref "${ref_psf}" __links

    local -a __cmd=(
        mlt.psfrecon
        ".mval=${__m_in}"
        ".modes_dm=${__s_modes}"
        ".outdir=${__s_out}"
        ".outreport=${__s_rep}"
        ".method=${method}"
        ".lambda_um=${lambda_um}"
        ".bw_frac=${bw_frac}"
        ".nlambda=${nlambda}"
        ".pixscale_ld=${pixscale_ld}"
        ".nwf=${nwf}"
        ".ncam=${ncam}"
        ".tel_diam_m=${tel_diam_m}"
        ".pup_diam=${pup_diam}"
        ".cobs=${cobs}"
        ".dm_diam=${dm_diam}"
        ".if_coupling=${if_coupling}"
        ".opd_scale=${opd_scale}"
        ".seeing_arcsec=${seeing_arcsec}"
        ".decim=${decim}"
        ".maxframes=${maxframes}"
    )
    [[ -n "${__s_pup}" ]] && __cmd+=(".pup=${__s_pup}")
    [[ -n "${__s_ncpa}" ]] && __cmd+=(".ncpa=${__s_ncpa}")
    [[ -n "${__s_og}" ]] && __cmd+=(".ogain=${__s_og}")
    [[ -n "${__s_ref}" ]] && __cmd+=(".ref_psf=${__s_ref}")

    "${__cmd[@]}"
    local __rc=$?
    __mlt_rm_images _mlt_pmv _mlt_pmv_t
    __mlt_cleanup __links
    return ${__rc}
}

# End-to-end telemetry health, dual-stage AO optimization, & Vis/NIR PSF reconstruction suite
mlt.aotlm() {
    @args 'tlmdir:str:telemetry directory containing cubes and telemetry_manifest.env' \
          'outdir:str=:output analysis directory (default <tlmdir>/analysis)' \
          'decim_psf:int=4:temporal decimation for short-exposure PSFR (>=1)' -- "$@"

    if [[ ! -d "${tlmdir}" ]]; then
        echo "mlt.aotlm: directory not found: ${tlmdir}" >&2
        return 1
    fi
    local __out="${outdir:-${tlmdir}/analysis}"
    mkdir -p "${__out}" || return 1

    local TLM_SEEING="0.60" TLM_TEL_DIAM="8.2" TLM_NWF="256" TLM_PUP_DIAM="0.9375"
    local TLM_COBS="0.30" TLM_FS1="1000.0" TLM_FS2="2000.0"
    local TLM_LAT1="2.5" TLM_LAT2="2.0" TLM_GAIN1="0.35" TLM_GAIN2="0.35"
    local TLM_MULT1="0.98" TLM_MULT2="0.98"
    local TLM_DM2_DIAM="0.9986413" TLM_DM2_COUPL="0.15"
    local TLM_STREAMS="csim_wfs_nir csim_wfs_vis aol1_dmC aol2_dmC dm01disp dm02disp"
    TLM_STREAMS+=" aol1_modevalWFS aol1_modevalDM aol1_modevalOL"
    TLM_STREAMS+=" aol2_modevalWFS aol2_modevalDM aol2_modevalOL"
    TLM_STREAMS+=" psf_vis_final psf_nir_final"
    local TLM_PUP_FITS="" TLM_STG1_CALIB="" TLM_STG2_CALIB="" TLM_PARENT_OUT=""

    if [[ -f "${tlmdir}/telemetry_manifest.env" ]]; then
        # shellcheck disable=SC1090
        source "${tlmdir}/telemetry_manifest.env"
    fi
    [[ -z "${TLM_STG1_CALIB}" ]] && TLM_STG1_CALIB="${tlmdir}/calib_stg1"
    [[ -z "${TLM_STG2_CALIB}" ]] && TLM_STG2_CALIB="${tlmdir}/calib_stg2"
    [[ -z "${TLM_PARENT_OUT}" ]] && TLM_PARENT_OUT="$(dirname "${tlmdir}")"
    [[ -z "${TLM_PUP_FITS}" && -f "${tlmdir}/csim_pup.fits" ]] && \
        TLM_PUP_FITS="${tlmdir}/csim_pup.fits"

    local __rep="${__out}/TELEMETRY_REPORT.md"
    {
        echo "# Closed-Loop Telemetry, AO Optimization & PSF Reconstruction Report"
        echo ""
        echo "- **Telemetry Directory**: \`${tlmdir}\`"
        echo "- **Seeing**: \`${TLM_SEEING}\"\` @ 0.50 um | **Telescope**: \`${TLM_TEL_DIAM} m\`"
        echo ""
        echo "## 1. Captured Stream Health (\`mlt.tlmindex\`)"
        echo ""
        echo "| Stream | Cubes | Frames | Rate [Hz] | Jitter [us] | Dropped |"
        echo "| :--- | ---: | ---: | ---: | ---: | ---: |"
    } > "${__rep}"

    local __s
    for __s in ${TLM_STREAMS}; do
        local -A __kv=()
        if __mlt_run_tlmindex "${tlmdir}" "${__s}" __kv 2>/dev/null; then
            printf '| `%s` | %s | %s | %s | %s | %s |\n' \
                "${__s}" "${__kv[ncubes]}" "${__kv[nframes]}" "${__kv[fps]}" \
                "${__kv[jitter_us]}" "${__kv[ndropped]}" >> "${__rep}"
        fi
    done

    {
        echo ""
        echo "## 2. Modal Loop Diagnostics & Gain Optimization (\`mlt.aoopt\`)"
        echo ""
        echo "| Stage | Modes | Frames | OL RMS | Curr RMS | Opt RMS | Pred Var Red. | Mean Opt. Gain | 0-dB BW [Hz] | Dom. Peak [Hz] |"
        echo "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    } >> "${__rep}"

    local __stg
    for __stg in 1 2; do
        local __fs="${TLM_FS1}" __lat="${TLM_LAT1}" __gn="${TLM_GAIN1}" __ml="${TLM_MULT1}"
        local __cdir="${TLM_STG1_CALIB}" __lbl="Stage 1 (AO3k)"
        if (( __stg == 2 )); then
            __fs="${TLM_FS2}"; __lat="${TLM_LAT2}"; __gn="${TLM_GAIN2}"; __ml="${TLM_MULT2}"
            __cdir="${TLM_STG2_CALIB}"; __lbl="Stage 2 (SCExAO)"
        fi
        local __so="${__out}/aoopt_stg${__stg}"
        local __mg="${TLM_PARENT_OUT}/aol${__stg}_mgain.fits"
        local __sc="${TLM_PARENT_OUT}/selfRM_calib_stg${__stg}.fits"
        local __ss="${TLM_PARENT_OUT}/selfRM_onsky_stg${__stg}.fits"
        [[ ! -f "${__sc}" && -f "${__cdir}/selfRM_calib.fits" ]] && \
            __sc="${__cdir}/selfRM_calib.fits"

        local __est_step=1
        [[ -f "${__sc}" && -f "${__ss}" ]] && __est_step=0
        if mlt.aoloop "${tlmdir}" "${__so}" "${__stg}" "${__fs}" "${__lat}" "${__gn}" "${__ml}" \
            "${__mg}" "" "${__sc}" "${__ss}" "${__est_step}" 512; then
            local -A __ak=()
            __mlt_read_kv "${__so}/aoopt_summary.txt" __ak
            printf '| %s | %s | %s | %s | %s | %s | %s%% | %s | %s | %s |\n' \
                "${__lbl}" "${__ak[nmodes]}" "${__ak[nframes]}" "${__ak[rms_ol]}" \
                "${__ak[rms_curr]}" "${__ak[rms_opt]}" "${__ak[var_red_pct]}" \
                "${__ak[mean_ogain]}" "${__ak[median_bw_hz]}" "${__ak[dom_vib_hz]}" >> "${__rep}"
        fi
    done

    local __cm2="${TLM_STG2_CALIB}/CMmodesDM.fits"
    if [[ -f "${__cm2}" ]]; then
        {
            echo ""
            echo "## 3. Telemetry PSF Reconstruction (\`mlt.psfrecon\`)"
            echo ""
            echo "| Band | Optical Gain | Method | SR Recon | SR Ctrl | SR Meas | |dSR| | 2D Corr | In-Band [nm] | Fitting [nm] |"
            echo "| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
        } >> "${__rep}"

        local __og_file="${__out}/aoopt_stg2/ogain_est.fits"
        local __bspec
        for __bspec in "vis:0.75:0.15:2.20:Visible (0.75 um)" \
                       "nir:1.65:0.18:2.50:Near-IR H (1.65 um)"; do
            local __btag __lam __bw __ps __blbl
            IFS=':' read -r __btag __lam __bw __ps __blbl <<< "${__bspec}"
            local __ref="${TLM_PARENT_OUT}/psf_${__btag}_final.fits"
            local __og_mode
            for __og_mode in "unit" "calibrated"; do
                local __og_arg="" __og_lbl="Unity (1.0)"
                if [[ "${__og_mode}" == "calibrated" ]]; then
                    [[ ! -f "${__og_file}" ]] && continue
                    __og_arg="${__og_file}"
                    __og_lbl="selfRM/aoopt"
                fi
                local __po="${__out}/psfr_${__btag}_${__og_mode}"
                if mlt.psfr "${tlmdir}" "${__cm2}" "${__po}" "aol2_modevalWFS" "hybrid" \
                    "${__lam}" "${__bw}" 3 "${__ps}" "${TLM_NWF}" 256 "${TLM_TEL_DIAM}" \
                    "${TLM_PUP_DIAM}" "${TLM_COBS}" "${TLM_DM2_DIAM}" "${TLM_DM2_COUPL}" 2.0 \
                    "${TLM_SEEING}" "${TLM_PUP_FITS}" "" "${__og_arg}" "${__ref}" \
                    "${decim_psf}" 0; then
                    local -A __pk=()
                    __mlt_read_kv "${__po}/psfrecon_summary.txt" __pk
                    printf '| %s | %s | `%s` | %s | %s | %s | %s | %s | %s | %s |\n' \
                        "${__blbl}" "${__og_lbl}" "${__pk[method]}" "${__pk[sr_recon]}" \
                        "${__pk[sr_ctrl]}" "${__pk[sr_meas]}" "${__pk[sr_abs_err]}" \
                        "${__pk[corr_2d]}" "${__pk[inband_rms_nm]}" \
                        "${__pk[fitting_rms_nm]}" >> "${__rep}"
                fi
            done
        done
    fi

    echo "mlt.aotlm: report written -> ${__rep}"
    return 0
}
