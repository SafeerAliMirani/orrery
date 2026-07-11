"""Orrery Phase 1: integrate + fit one chunk of Gaia stars on Galileo100.

Merged version of the Phase 0 pipeline (01_integrate + 02_fit), one SLURM
array task per chunk. Reads the master CSV, processes rows [start, end),
writes one HDF5 fit shard with parameters, per-star errors, and energy QC.

  python integrate_fit.py --csv gaia_1m.csv --start 0 --end 31250 \
      --out fits/fit_000.h5

Algorithm identical to Phase 0 (validated: median 10.8 pc, p95 169 pc at
|t| = 500 Myr on 46k stars). See phase0/README.md.
"""

import argparse
import numpy as np
import astropy.units as u
import astropy.coordinates as coord
from astropy.table import Table
import h5py

import gala.dynamics as gd
import gala.potential as gp
from gala.integrate import DOPRI853Integrator

N_STEPS = 1000          # per direction, dt = 1 Myr: t spans -1000..+1000
K_R, K_Z, K_P = 4, 4, 3
CHUNK = 250             # fit vectorization chunk
INT_BATCH = 3000        # integration memory batch


def integrate_batch(t_rows):
    dist = (1000.0 / np.asarray(t_rows["parallax"])) * u.pc
    c = coord.SkyCoord(
        ra=np.asarray(t_rows["ra"]) * u.deg,
        dec=np.asarray(t_rows["dec"]) * u.deg,
        distance=dist,
        pm_ra_cosdec=np.asarray(t_rows["pmra"]) * u.mas / u.yr,
        pm_dec=np.asarray(t_rows["pmdec"]) * u.mas / u.yr,
        radial_velocity=np.asarray(t_rows["radial_velocity"]) * u.km / u.s,
    )
    gc = c.transform_to(coord.Galactocentric())
    w0 = gd.PhaseSpacePosition(pos=gc.cartesian.xyz, vel=gc.velocity.d_xyz)
    H = gp.Hamiltonian(gp.MilkyWayPotential2022())
    fwd = H.integrate_orbit(w0, dt=1.0 * u.Myr, n_steps=N_STEPS,
                            Integrator=DOPRI853Integrator)
    bwd = H.integrate_orbit(w0, dt=-1.0 * u.Myr, n_steps=N_STEPS,
                            Integrator=DOPRI853Integrator)
    fx = fwd.pos.xyz.to_value(u.kpc)
    bx = bwd.pos.xyz.to_value(u.kpc)
    pos = np.concatenate([bx[:, ::-1, :][:, :-1, :], fx], axis=1)
    pos = np.transpose(pos, (2, 1, 0)).astype(np.float32)
    e_f = fwd.energy().to_value(u.km ** 2 / u.s ** 2)
    e_b = bwd.energy().to_value(u.km ** 2 / u.s ** 2)
    e_all = np.concatenate([e_b, e_f], axis=0)
    de = (np.abs(e_all - e_all[0]).max(axis=0) / np.abs(e_all[0]))
    return pos, de.astype(np.float32)


def refine_freq(x, t, w_guess, half_width=1.2e-4, n_grid=33):
    grid = np.linspace(-half_width, half_width, n_grid)
    wg = w_guess[:, None] + grid[None, :]
    han = np.hanning(len(t))
    xw = (x * han[None, :]) * np.exp(-1j * w_guess[:, None] * t[None, :])
    D = np.exp(-1j * grid[:, None] * t[None, :])
    proj = np.einsum("nt,gt->ng", xw, D)
    pw = (proj * proj.conj()).real
    i = np.clip(np.argmax(pw, axis=1), 1, n_grid - 2)
    ar = np.arange(len(x))
    y0, y1, y2 = pw[ar, i - 1], pw[ar, i], pw[ar, i + 1]
    denom = y0 - 2 * y1 + y2
    off = np.where(np.abs(denom) > 0, 0.5 * (y0 - y2) / denom, 0.0)
    step = grid[1] - grid[0]
    return wg[ar, i] + np.clip(off, -1, 1) * step


def extract_lines(x, t, K):
    n, T = x.shape
    dt = t[1] - t[0]
    dw = 2 * np.pi / (t[-1] - t[0])
    han = np.hanning(T)
    res = x.copy()
    freqs = np.zeros((n, K))
    ar = np.arange(n)
    for k in range(K):
        F = np.fft.rfft(res * han[None, :], axis=1)
        mag = np.abs(F)
        mag[:, 0] = 0.0
        for j in range(k):
            bj = np.rint(freqs[:, j] * T * dt / (2 * np.pi)).astype(int)
            for o in range(-2, 3):
                mag[ar, np.clip(bj + o, 0, mag.shape[1] - 1)] = 0.0
        b = np.clip(np.argmax(mag, axis=1), 1, mag.shape[1] - 2)
        lg = np.log(np.maximum(mag, 1e-30))
        y0, y1, y2 = lg[ar, b - 1], lg[ar, b], lg[ar, b + 1]
        denom = y0 - 2 * y1 + y2
        off = np.where(np.abs(denom) > 0, 0.5 * (y0 - y2) / denom, 0.0)
        w0 = (b + np.clip(off, -1, 1)) * 2 * np.pi / (T * dt)
        w0 = np.maximum(w0, 0.25 * dw)
        w = refine_freq(res, t, w0)
        w = refine_freq(res, t, w, half_width=6e-6, n_grid=25)
        ec = np.cos(w[:, None] * t[None, :])
        es = np.sin(w[:, None] * t[None, :])
        a = 2.0 / T * np.einsum("nt,nt->n", res, ec)
        b2 = 2.0 / T * np.einsum("nt,nt->n", res, es)
        amp = np.hypot(a, b2)
        pha = np.arctan2(-b2, a)
        res = res - amp[:, None] * np.cos(w[:, None] * t[None, :] + pha[:, None])
        freqs[:, k] = w
    return freqs


def joint_lsq(x, t, freqs, with_t=False):
    n, T = x.shape
    K = freqs.shape[1]
    ph = freqs[:, :, None] * t[None, None, :]
    cs, sn = np.cos(ph), np.sin(ph)
    inter = np.empty((n, 2 * K, T))
    inter[:, 0::2] = cs
    inter[:, 1::2] = sn
    parts = [np.ones((n, 1, T))]
    if with_t:
        parts.append(np.broadcast_to(t[None, None, :], (n, 1, T)))
    G = np.concatenate(parts + [inter], axis=1)
    M = np.einsum("nct,ndt->ncd", G, G)
    bv = np.einsum("nct,nt->nc", G, x)
    C = M.shape[1]
    ridge = 1e-9 * np.trace(M, axis1=1, axis2=2) / C
    M[:, np.arange(C), np.arange(C)] += ridge[:, None]
    coef = np.linalg.solve(M, bv[:, :, None])[:, :, 0]
    model = np.einsum("nc,nct->nt", coef, G)
    base = 2 if with_t else 1
    a = coef[:, base::2]
    b = coef[:, base + 1::2]
    amp = np.hypot(a, b)
    pha = np.arctan2(-b, a)
    const = coef[:, 0]
    lin = coef[:, 1] if with_t else np.zeros(n)
    return const, lin, amp, pha, model


def line_sum(t, f, a, p):
    return np.einsum("nk,nkt->nt", a,
                     np.cos(f[:, :, None] * t[None, None, :] + p[:, :, None]))


def fit_coord(x, t, K, with_t=False):
    xc = x - x.mean(axis=1, keepdims=True)
    if with_t:
        slope = (xc * t[None, :]).sum(axis=1) / (t * t).sum()
        xc = xc - slope[:, None] * t[None, :]
    f = extract_lines(xc, t, K)
    const, lin, amp, pha, model = joint_lsq(x, t, f, with_t)
    for _ in range(2):
        resid = x - model
        for k in range(K):
            lk = amp[:, k][:, None] * np.cos(f[:, k][:, None] * t[None, :]
                                             + pha[:, k][:, None])
            f[:, k] = refine_freq(resid + lk, t, f[:, k],
                                  half_width=1.5e-5, n_grid=31)
        const, lin, amp, pha, model = joint_lsq(x, t, f, with_t)
    return const, lin, f, amp, pha, model


def reconstruct(t, Rg, fR, aR, pR, zc, fZ, aZ, pZ, ph0, Om, fP, aP, pP):
    R = Rg[:, None] + line_sum(t, fR, aR, pR)
    z = zc[:, None] + line_sum(t, fZ, aZ, pZ)
    phi = ph0[:, None] + Om[:, None] * t[None, :] + line_sum(t, fP, aP, pP)
    return np.stack([R * np.cos(phi), R * np.sin(phi), z], axis=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--end", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tbl = Table.read(args.csv, format="csv")[args.start:args.end]
    n_total = len(tbl)
    t = np.arange(-N_STEPS, N_STEPS + 1, dtype=np.float64)
    i5a, i5b = np.argmin(np.abs(t + 500)), np.argmin(np.abs(t - 500))

    keys = ["Rg", "fR", "aR", "pR", "zc", "fZ", "aZ", "pZ",
            "ph0", "Om", "fP", "aP", "pP",
            "err_med", "err_p95", "err500", "errmax", "dE_over_E"]
    out = {k: [] for k in keys}

    for b0 in range(0, n_total, INT_BATCH):
        rows = tbl[b0:b0 + INT_BATCH]
        pos, de = integrate_batch(rows)
        out["dE_over_E"].append(de)
        for c0 in range(0, len(rows), CHUNK):
            P = pos[c0:c0 + CHUNK].astype(np.float64)
            x, y, z = P[:, :, 0], P[:, :, 1], P[:, :, 2]
            R = np.hypot(x, y)
            phi = np.unwrap(np.arctan2(y, x), axis=1)
            Rg, _, fR, aR, pR, _ = fit_coord(R, t, K_R)
            zc, _, fZ, aZ, pZ, _ = fit_coord(z, t, K_Z)
            ph0, Om, fP, aP, pP, _ = fit_coord(phi, t, K_P, with_t=True)
            rec = reconstruct(t, Rg, fR, aR, pR, zc, fZ, aZ, pZ,
                              ph0, Om, fP, aP, pP)
            err = np.linalg.norm(rec - P, axis=2) * 1000.0
            out["err_med"].append(np.median(err, axis=1))
            out["err_p95"].append(np.percentile(err, 95, axis=1))
            out["err500"].append(np.maximum(err[:, i5a], err[:, i5b]))
            out["errmax"].append(err.max(axis=1))
            for k, v in [("Rg", Rg), ("fR", fR), ("aR", aR), ("pR", pR),
                         ("zc", zc), ("fZ", fZ), ("aZ", aZ), ("pZ", pZ),
                         ("ph0", ph0), ("Om", Om), ("fP", fP), ("aP", aP),
                         ("pP", pP)]:
                out[k].append(v)
        print(f"  rows {args.start + b0}..{args.start + b0 + len(rows)} done",
              flush=True)

    with h5py.File(args.out, "w") as f:
        for k in keys:
            f.create_dataset(k, data=np.concatenate(out[k], axis=0))
        f.create_dataset("source_id",
                         data=np.asarray(tbl["source_id"], dtype=np.int64))
        f.attrs["row_start"] = args.start
        f.attrs["row_end"] = args.end
        f.attrs["potential"] = "MilkyWayPotential2022"
        f.attrs["K"] = (K_R, K_Z, K_P)

    e5 = np.concatenate(out["err500"])
    print(f"shard {args.out}: {n_total} stars | err@500Myr median "
          f"{np.median(e5):.1f} pc, p95 {np.percentile(e5, 95):.0f} pc, "
          f"frac>200pc {(e5 > 200).mean():.3f}")


if __name__ == "__main__":
    main()
