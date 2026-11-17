# SPDX-FileCopyrightText: 2026 Olivier Guyon et al
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""End-to-end accuracy and integration tests for milk-mltools."""

import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

MILK_INSTALLDIR = Path(
    os.environ.get("MILK_INSTALLDIR", "/home/guyon/src/milk/_build_dev/_install")
)
MILK_CLI = MILK_INSTALLDIR / "bin" / "milk-cli"


@pytest.fixture()
def milk_env(tmp_path):
    """Return an isolated environment dictionary for headless milk-cli tests."""
    shm_dir = tmp_path / "shm"
    cache_dir = tmp_path / "calc_cache"
    tmp_dir = tmp_path / "tmp"
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
    return env


def run_milk(cmd: str, env: dict, check: bool = True) -> subprocess.CompletedProcess:
    """Execute a milk-cli command string with strict error checking."""
    res = subprocess.run(
        [str(MILK_CLI), "-q", "-e", "-o", "-c", cmd],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if check and res.returncode != 0:
        raise AssertionError(
            f"milk-cli failed (rc={res.returncode})\nCMD: {cmd}\n"
            f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
        )
    return res


def parse_kv(path: Path) -> dict[str, str]:
    """Parse key=value text files produced by mltools."""
    out = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def load_stream_cubes(directory: Path, prefix: str) -> tuple[np.ndarray, np.ndarray]:
    """Load and concatenate all FITS cubes and effective timestamps for a stream."""
    txt_files = sorted(directory.glob(f"{prefix}_*.txt"))
    assert txt_files, f"No {prefix}_*.txt files in {directory}"
    frames = []
    times = []
    for tf in txt_files:
        ff = tf.with_suffix(".fits")
        if not ff.exists():
            continue
        arr = np.loadtxt(tf, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr[None, :]
        # Column 3 is t_log, column 4 is t_acq; prefer t_acq if non-zero.
        t_eff = np.where(arr[:, 4] > 0.0, arr[:, 4], arr[:, 3])
        with fits.open(ff) as hdul:
            data = hdul[0].data.astype(np.float64)  # (N, ny, nx) in astropy
        if data.ndim == 2:
            data = data[None, :, :]
        frames.append(data)
        times.append(t_eff)
    cube = np.concatenate(frames, axis=0)
    ts = np.concatenate(times, axis=0)
    order = np.argsort(ts)
    return cube[order], ts[order]


def write_synthetic_cube(
    directory: Path, prefix: str, cube: np.ndarray, t_sec: np.ndarray, fps: float
) -> None:
    """Write a (N, ny, nx) cube + 7-column logshmim ASCII .txt file."""
    directory.mkdir(parents=True, exist_ok=True)
    epoch_ns = 1767225600 * 1_000_000_000 + 43200 * 1_000_000_000  # 2026-01-01 12:00:00
    t0_ns = epoch_ns + int(round(t_sec[0] * 1e9))
    sec_tot = t0_ns // 1_000_000_000
    nsec = t0_ns % 1_000_000_000
    hh = (sec_tot % 86400) // 3600
    mm = (sec_tot % 3600) // 60
    ss = sec_tot % 60
    stem = f"{prefix}_{hh:02d}:{mm:02d}:{ss:02d}.{nsec:09d}"

    fits.writeto(directory / f"{stem}.fits", cube.astype(np.float32), overwrite=True)
    lines = []
    for i in range(len(t_sec)):
        ti_ns = epoch_ns + int(round(t_sec[i] * 1e9))
        s = ti_ns / 1e9
        t_log = s + 0.0001
        lines.append(f"{i} {i} 0 {t_log:.9f} {s:.9f} {i} 0\n")
    (directory / f"{stem}.txt").write_text("".join(lines))


def subspace_cosines(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Compute column-wise max correlation / principal cosines between mode matrices (P, k)."""
    Qa, _ = np.linalg.qr(A)
    Qb, _ = np.linalg.qr(B)
    return np.linalg.svd(Qa.T @ Qb, compute_uv=False)


def test_tlm_io_edge_cases(tmp_path, milk_env):
    """Verify tlmindex, tlmload, tlmsave, orphans, drops, acq=0 fallback, mask/ROI/bin, >63 path."""
    # Deep (>75 chars) path to exercise __mlt_short_path
    deep_dir = tmp_path / ("very_long_directory_name_" * 3) / "synth"
    assert len(str(deep_dir)) > 75

    run_milk(
        f"mlt.tlmsynth '{deep_dir}' 1200 1000.0 1000.0 12 10 3 0.0 0.8 0.5 0.7 0.1 500 3 1",
        milk_env,
    )

    # Inject 1 orphan .txt and 1 orphan .fits for streamA
    (deep_dir / "streamA_23:59:58.000000000.txt").write_text("0 0 0 1767311998.0 0 0 0\n")
    fits.writeto(
        deep_dir / "streamA_23:59:59.000000000.fits",
        np.zeros((2, 10, 12), dtype=np.float32),
        overwrite=True,
    )

    rep_a = tmp_path / "report_a.txt"
    rep_b = tmp_path / "report_b.txt"
    run_milk(f"mlt.tlmcheck '{deep_dir}' streamA '' '' '{rep_a}'", milk_env)
    kva = parse_kv(rep_a)
    assert int(kva["ncubes_a"]) == 3
    assert int(kva["nframes_a"]) == 1200
    assert int(kva["ndropped_a"]) == 3
    assert int(kva["norphan_txt_a"]) == 1
    assert int(kva["norphan_fits_a"]) == 1
    assert int(kva["has_acq_a"]) == 1
    assert abs(float(kva["fps_a"]) - 1000.0) < 1.0

    # Stream B has acq_zero=1 -> has_acq == 0 fallback to t_log, plus two-stream overlap check
    run_milk(f"mlt.tlmcheck '{deep_dir}' streamA '{deep_dir}' streamB '{rep_b}'", milk_env)
    kvb = parse_kv(rep_b)
    assert int(kvb["has_acq_b"]) == 0
    assert int(kvb["nframes_b"]) >= 1200
    assert float(kvb["overlap_sec"]) > 1.0

    # Also verify mlt.pca shortens >75-char input and output paths automatically via __mlt_short_path
    deep_pca = deep_dir / "pca_long_path"
    run_milk(f"mlt.pca '{deep_dir}' streamA '{deep_pca}' 2 cov", milk_env)
    assert (deep_pca / "modes.fits").is_file()

    # Test ROI + bin + 2D mask via raw C tlmload -> tlmsave -> tlmload round-trip (< 63-char paths)
    with tempfile.TemporaryDirectory(prefix="mlt_") as short_tmp:
        sp = Path(short_tmp)
        s_in = sp / "in"
        s_in.symlink_to(deep_dir)
        mask = np.zeros((5, 6), dtype=np.float32)  # matches ROI 0:11,0:9 binned 2x2 => 6x5!
        mask[1:4, 1:5] = 1.0
        mask_fits = sp / "mask.fits"
        fits.writeto(mask_fits, mask, overwrite=True)

        rt_dir = sp / "rt"
        run_milk(
            f"mlt.tlmload .dir='{s_in}' .stream=streamA .out=@L:sub "
            f".roi=0:11,0:9 .bin=2 .mask='{mask_fits}' .maxframes=200; "
            f"mlt.tlmsave sub sub_t '{rt_dir}' rtA; "
            f"@calc -v p0,n0,ep0 's.sub.xsize; s.sub.ysize; s.sub.kw.MLTEPOCH'; "
            f"echo \"SHAPE=$p0,$n0,$ep0\"",
            milk_env,
        )
        res_check = run_milk(
            f"mlt.tlmload .dir='{rt_dir}' .stream=rtA .out=@L:rt; "
            f"@calc -v p1,n1 's.rt.xsize; s.rt.ysize'; echo \"RT=$p1,$n1\"",
            milk_env,
        )
        assert "RT=12,200" in res_check.stdout

        # Compare pixel + timestamp round-trip
        rt_cube, rt_ts = load_stream_cubes(rt_dir, "rtA")
        raw_cube, raw_ts = load_stream_cubes(deep_dir, "streamA")
        assert rt_cube.shape == (200, 1, 12)
        cropped = raw_cube[:200, 0:10, 0:12]
        binned = cropped.reshape(200, 5, 2, 6, 2).mean(axis=(2, 4))
        expected_vec = binned[:, mask > 0.5]
        np.testing.assert_allclose(rt_cube[:, 0, :], expected_vec, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(rt_ts, raw_ts[:200], rtol=0.0, atol=1e-8)


def test_pca_all_solvers_and_proj(tmp_path, milk_env):
    """Compare mlt.pca (cov, gram, rsvd) + mlt.pcaproj against numpy.linalg.svd."""
    sdir = tmp_path / "synth_pca"
    run_milk(
        f"mlt.tlmsynth '{sdir}' 800 1000.0 1000.0 16 16 4 0.0 0.85 0.6 0.8 0.12 400 0 0",
        milk_env,
    )
    cube, _ = load_stream_cubes(sdir, "streamA")
    X = cube.reshape(cube.shape[0], -1).T  # (P, N) with astropy order flattened (y*nx+x)
    # Note: in milk column-major, (x, y) linearized index is y*nx + x, identical to C (y, x)!
    mu_np = X.mean(axis=1, keepdims=True)
    Xc = X - mu_np
    U_np, s_np, _ = np.linalg.svd(Xc, full_matrices=False)
    eig_np = (s_np**2) / (X.shape[1] - 1)

    for method, maxf in [("cov", 800), ("rsvd", 800), ("gram", 120)]:
        outdir = tmp_path / f"pca_{method}"
        run_milk(
            f"mlt.pca '{sdir}' streamA '{outdir}' 4 {method} 1 '' '' {maxf}",
            milk_env,
        )
        with fits.open(outdir / "modes.fits") as hm:
            modes = hm[0].data.reshape(4, -1).T  # (P, 4)
        with fits.open(outdir / "mean.fits") as hm:
            mean_mlt = hm[0].data.reshape(-1)
        eig_tbl = np.loadtxt(outdir / "eigval.txt", comments="#")
        eig_mlt = eig_tbl[:, 1]

        if maxf == 800:
            np.testing.assert_allclose(mean_mlt, mu_np[:, 0], rtol=1e-4, atol=1e-4)
            # Compare against numpy SVD mode-by-mode
            for m in range(4):
                cos_m = abs(np.dot(modes[:, m], U_np[:, m]))
                assert cos_m > 0.99, f"{method} mode {m} cosine={cos_m}"
            np.testing.assert_allclose(eig_mlt, eig_np[:4], rtol=2e-2, atol=1e-5)
        else:
            X_sub = X[:, :120]
            Xc_sub = X_sub - X_sub.mean(axis=1, keepdims=True)
            U_sub, s_sub, _ = np.linalg.svd(Xc_sub, full_matrices=False)
            eig_sub = (s_sub**2) / (120 - 1)
            for m in range(4):
                cos_m = abs(np.dot(modes[:, m], U_sub[:, m]))
                assert cos_m > 0.99, f"gram mode {m} cosine={cos_m}"
            np.testing.assert_allclose(eig_mlt, eig_sub[:4], rtol=1e-2, atol=1e-5)

        if method == "cov":
            proj_dir = tmp_path / "proj"
            run_milk(
                f"mlt.pcaproj '{sdir}' streamA '{outdir}/modes.fits' '{outdir}/mean.fits' '{proj_dir}'",
                milk_env,
            )
            c_mlt, _ = load_stream_cubes(proj_dir, "coeffs")
            c_mlt = c_mlt[:, 0, :]  # (N, 4)
            c_ref = (modes.T @ Xc).T
            np.testing.assert_allclose(c_mlt, c_ref, rtol=1e-3, atol=1e-3)


def test_latency_subframe_and_unequal_fps(tmp_path, milk_env):
    """Verify continuous sub-frame lag estimation on equal and unequal frame rates."""
    # Case A: 1000 Hz vs 1000 Hz with +2.35 ms lag and strong AR(1)=0.90 temporal color
    d_eq = tmp_path / "lat_eq"
    out_eq = tmp_path / "lat_eq_out"
    run_milk(
        f"mlt.tlmsynth '{d_eq}' 3000 1000.0 1000.0 12 12 4 0.00235 0.90 0.65 0.90 0.12 1500 0 0",
        milk_env,
    )
    run_milk(
        f"mlt.latency '{d_eq}' streamA '{d_eq}' streamB '{out_eq}' 30.0 2.0 0.05 4 8 4",
        milk_env,
    )
    kv_eq = parse_kv(out_eq / "latency_summary.txt")
    tau_eq = float(kv_eq["tau_ms"])
    assert abs(tau_eq - 2.35) < 0.12, f"Expected ~+2.35 ms, got {tau_eq}"
    assert float(kv_eq["peak_corr"]) > 0.25

    # Case B: Unequal rate 1000 Hz vs 500 Hz with negative lag -3.70 ms, via OS wrapper script!
    d_un = tmp_path / "lat_un"
    out_un = tmp_path / "lat_un_out"
    run_milk(
        f"mlt.tlmsynth '{d_un}' 3000 1000.0 500.0 12 12 4 -0.00370 0.90 0.65 0.85 0.12 1500 0 0",
        milk_env,
    )
    wrapper = MILK_INSTALLDIR / "bin" / "milk-mlt-latency"
    res = subprocess.run(
        [
            str(wrapper),
            str(d_un),
            "streamA",
            str(d_un),
            "streamB",
            str(out_un),
            "30.0",
            "2.0",
            "0.05",
            "4",
            "8",
            "4",
        ],
        env=milk_env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert res.returncode == 0, res.stderr
    kv_un = parse_kv(out_un / "latency_summary.txt")
    tau_un = float(kv_un["tau_ms"])
    assert abs(tau_un - (-3.70)) < 0.20, f"Expected ~-3.70 ms, got {tau_un}"


def test_cca_heldout_cv(tmp_path, milk_env):
    """Verify mlt.cca with lag=auto recovers injected canonical correlations on held-out test set."""
    sdir = tmp_path / "synth_cca"
    cdir = tmp_path / "cca_out"
    run_milk(
        f"mlt.tlmsynth '{sdir}' 3500 1000.0 1000.0 12 12 4 0.00235 0.90 0.65 0.82 0.10 1750 0 0",
        milk_env,
    )
    run_milk(
        f"mlt.cca '{sdir}' streamA '{sdir}' streamB '{cdir}' 4 12 12 1e-3 1e-3 auto 0.7",
        milk_env,
    )
    kv = parse_kv(cdir / "cca_summary.txt")
    assert abs(float(kv["lag_sec"]) * 1000.0 - 2.35) < 0.15

    rho_tbl = np.loadtxt(cdir / "cca_rho.txt", comments="#")
    rho_test = rho_tbl[:, 2]
    assert abs(rho_test[0] - 0.90) < 0.07, f"rho_test[0]={rho_test[0]}"
    assert abs(rho_test[1] - 0.65) < 0.09, f"rho_test[1]={rho_test[1]}"
    assert rho_test[2] < 0.20, f"noise mode rho_test[2]={rho_test[2]} too high"
    assert rho_test[3] < 0.20, f"noise mode rho_test[3]={rho_test[3]} too high"

    # Check forward spatial loading patterns against ground truth correlated subspace
    with fits.open(sdir / "truth_modesA.fits") as ha, fits.open(cdir / "cca_A_patterns.fits") as hc:
        true_a = ha[0].data[:2].reshape(2, -1).T
        est_a = hc[0].data[:2].reshape(2, -1).T
    cos_sub = subspace_cosines(true_a, est_a)
    assert np.min(cos_sub) > 0.90, f"CCA pattern subspace cosines={cos_sub}"

    u_var, _ = load_stream_cubes(cdir, "cca_U")
    v_var, _ = load_stream_cubes(cdir, "cca_V")
    assert u_var.shape[0] > 3000 and u_var.shape[2] == 4
    c0 = np.corrcoef(u_var[:, 0, 0], v_var[:, 0, 0])[0, 1]
    assert c0 > 0.80


def test_psd_and_coherence(tmp_path, milk_env):
    """Verify Welch PSD tone recovery and two-stream magnitude-squared coherence."""
    rng = np.random.default_rng(123)
    n = 2048
    fps = 1000.0
    t = np.arange(n, dtype=np.float64) / fps
    f_tone = 73.0
    tone = 2.5 * np.sin(2.0 * np.pi * f_tone * t)
    cube_a = tone[:, None, None] * np.ones((1, 4, 4)) + 0.15 * rng.normal(size=(n, 4, 4))
    cube_b = tone[:, None, None] * np.ones((1, 4, 4)) + 0.15 * rng.normal(size=(n, 4, 4))

    sdir = tmp_path / "synth_psd"
    pdir = tmp_path / "psd_out"
    write_synthetic_cube(sdir, "sigA", cube_a, t, fps)
    write_synthetic_cube(sdir, "sigB", cube_b, t, fps)

    run_milk(
        f"mlt.psd '{sdir}' sigA '{pdir}' 512 0.5 '{sdir}' sigB 0.0",
        milk_env,
    )
    tbl = np.loadtxt(pdir / "psd_mean.txt", comments="#")
    freqs = tbl[:, 0]
    psd = tbl[:, 1]
    coh = tbl[:, 2]

    ipk = int(np.argmax(psd))
    df = fps / 512.0
    assert abs(freqs[ipk] - f_tone) <= df + 0.5, f"Peak freq {freqs[ipk]} != {f_tone}"
    assert coh[ipk] > 0.85, f"Coherence at tone bin = {coh[ipk]}"
