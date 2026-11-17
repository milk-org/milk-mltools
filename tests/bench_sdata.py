# SPDX-FileCopyrightText: 2026 Olivier Guyon et al
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Read-only benchmark & validation against /data/sdata/20260101 SCExAO telemetry."""

import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

MILK_INSTALLDIR = Path(
    os.environ.get("MILK_INSTALLDIR", "/home/guyon/src/milk/_build_dev/_install")
)
MILK_CLI = MILK_INSTALLDIR / "bin" / "milk-cli"
SDATA_ROOT = Path(os.environ.get("MLT_SDATA_DIR", "/data/sdata/20260101"))


def parse_kv(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def run_milk(cmd: str, env: dict, timeout: int = 300) -> tuple[float, subprocess.CompletedProcess]:
    t0 = time.perf_counter()
    res = subprocess.run(
        [str(MILK_CLI), "-q", "-e", "-o", "-c", cmd],
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    dt = time.perf_counter() - t0
    if res.returncode != 0:
        raise RuntimeError(
            f"milk-cli failed (rc={res.returncode})\nCMD: {cmd}\n"
            f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
        )
    return dt, res


def main() -> int:
    iiwi_dir = SDATA_ROOT / "iiwi"
    dm_dir = SDATA_ROOT / "dm00disp03"
    ocam_dir = SDATA_ROOT / "ocam2d"
    for d in (iiwi_dir, dm_dir, ocam_dir):
        if not d.is_dir():
            print(f"Skipping /data/sdata benchmark: missing directory {d}", file=sys.stderr)
            return 0

    work = Path(tempfile.mkdtemp(prefix="mlt_sdata_bench_"))
    try:
        shm_dir = work / "shm"
        cache_dir = work / "calc_cache"
        tmp_dir = work / "tmp"
        shm_dir.mkdir(parents=True, exist_ok=True)
        cache_dir.mkdir(parents=True, exist_ok=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        env["MILK_INSTALLDIR"] = str(MILK_INSTALLDIR)
        env["MILK_SHM_DIR"] = str(shm_dir)
        env["MILK_CALC_CACHE"] = str(cache_dir)
        env["TMPDIR"] = str(tmp_dir)
        env["MILK_QUIET"] = "1"
        env["PATH"] = f"{MILK_INSTALLDIR / 'bin'}:{env.get('PATH', '')}"

        # Warm calc cache briefly on extension functions before timed runs
        run_milk("@calc 'x = mlt_xcov(zeros(2, 16), zeros(2, 16), maxlag=2, norm=1)'", env)

        print("=== 1. mlt.tlmcheck on iiwi (large multi-cube + orphan check) ===")
        rep_iiwi = work / "iiwi_report.txt"
        dt, res = run_milk(f"mlt.tlmcheck '{iiwi_dir}' iiwi '' '' '{rep_iiwi}'", env)
        k_iiwi = parse_kv(rep_iiwi)
        print(f"Wall time: {dt:.3f} s")
        print(res.stdout.strip())
        assert int(k_iiwi["ncubes_a"]) == 1875, f"Unexpected iiwi ncubes={k_iiwi}"
        assert int(k_iiwi["norphan_txt_a"]) == 15, f"Unexpected iiwi orphan txt={k_iiwi}"

        print("\n=== 2. mlt.tlmcheck on dm00disp03 vs ocam2d (overlap & has_acq) ===")
        rep_ov = work / "ov_report.txt"
        dt, res = run_milk(
            f"mlt.tlmcheck '{dm_dir}' dm00disp03 '{ocam_dir}' ocam2d '{rep_ov}'", env
        )
        k_ov = parse_kv(rep_ov)
        print(f"Wall time: {dt:.3f} s")
        print(res.stdout.strip())
        assert int(k_ov["has_acq_a"]) == 0, "dm00disp03 should have has_acq=0"
        assert int(k_ov["has_acq_b"]) == 1, "ocam2d should have has_acq=1"
        ov_sec = float(k_ov["overlap_sec"])
        assert ov_sec > 35.0, f"Overlap too short: {ov_sec} s"

        ov0 = float(k_ov["overlap_start_a_rel"])
        tstart_win = f"{ov0 + 5.0:.6f}"
        tend_win = "+30s"

        print("\n=== 3. mlt.pca (k=50, method=auto/rsvd, 30000 frames) on ocam2d (.fits.fz) ===")
        pca_out = work / "ocam2d_pca"
        dt_pca, res = run_milk(
            f"mlt.pca '{ocam_dir}' ocam2d '{pca_out}' 50 auto 1 '{tstart_win}' '' 30000",
            env,
        )
        kpca = parse_kv(pca_out / "pca_summary.txt")
        nfr_pca = int(kpca["nframes"])
        npix_pca = int(kpca["npix"])
        fps_proc = nfr_pca / dt_pca
        gb_s = (nfr_pca * npix_pca * 4) / (dt_pca * 1e9)
        print(
            f"Wall time: {dt_pca:.3f} s ({fps_proc:.0f} frames/s, {gb_s:.2f} GB/s float32)"
            f"  explained_50={kpca['explained_frac']}"
        )

        print("\n=== 4. mlt.latency on 30s dm00disp03 vs ocam2d overlap window ===")
        lat_out = work / "dm_ocam_latency"
        dt_lat, res = run_milk(
            f"mlt.latency '{dm_dir}' dm00disp03 '{ocam_dir}' ocam2d '{lat_out}' "
            f"25.0 1.5 0.05 8 8 4 '{tstart_win}' '{tend_win}'",
            env,
        )
        klat = parse_kv(lat_out / "latency_summary.txt")
        print(f"Wall time: {dt_lat:.3f} s")
        print(
            f"Latency tau = {float(klat['tau_ms']):.4f} +/- {float(klat['tau_err_ms']):.4f} ms "
            f"(peak_corr={float(klat['peak_corr']):.4f}, fwhm={float(klat['fwhm_ms']):.3f} ms, "
            f"drift={float(klat['drift_ppm']):.2f} ppm, N={klat['n_overlap']})"
        )

        print("\n=== 5. mlt.cca (k=10, pca=40, 30s overlap, held-out CV) dm00disp03 vs ocam2d ===")
        cca_out = work / "dm_ocam_cca"
        lag_sec = klat["tau_sec"]
        dt_cca, res = run_milk(
            f"mlt.cca '{dm_dir}' dm00disp03 '{ocam_dir}' ocam2d '{cca_out}' "
            f"10 40 40 1e-3 1e-3 '{lag_sec}' 0.7 '{tstart_win}' '{tend_win}'",
            env,
        )
        print(f"Wall time: {dt_cca:.3f} s")
        print((cca_out / "cca_rho.txt").read_text().strip())
        return 0
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
