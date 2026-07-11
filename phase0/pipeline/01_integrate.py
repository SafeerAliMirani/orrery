"""Phase 0: integrate Gaia DR3 6D stars +/-1 Gyr in MilkyWayPotential2022.

Reads the raw TAP CSV, converts ICRS to Galactocentric (astropy defaults),
integrates a row range with gala DOP853, writes one HDF5 shard with f32
positions sampled every 1 Myr and per-star energy conservation error.

Run per batch to keep each invocation short:
  python 01_integrate.py --csv <path> --start 0 --end 2000 --out shard_000.h5
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

N_STEPS = 1000  # per direction, dt = 1 Myr, so t spans -1000..+1000 Myr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--end", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    t = Table.read(args.csv, format="csv")[args.start : args.end]
    n = len(t)

    dist = (1000.0 / np.asarray(t["parallax"])) * u.pc  # SNR >= 10, 1/plx ok
    c = coord.SkyCoord(
        ra=np.asarray(t["ra"]) * u.deg,
        dec=np.asarray(t["dec"]) * u.deg,
        distance=dist,
        pm_ra_cosdec=np.asarray(t["pmra"]) * u.mas / u.yr,
        pm_dec=np.asarray(t["pmdec"]) * u.mas / u.yr,
        radial_velocity=np.asarray(t["radial_velocity"]) * u.km / u.s,
    )
    gc = c.transform_to(coord.Galactocentric())
    w0 = gd.PhaseSpacePosition(pos=gc.cartesian.xyz, vel=gc.velocity.d_xyz)

    pot = gp.MilkyWayPotential2022()
    H = gp.Hamiltonian(pot)

    fwd = H.integrate_orbit(w0, dt=1.0 * u.Myr, n_steps=N_STEPS,
                            Integrator=DOPRI853Integrator)
    bwd = H.integrate_orbit(w0, dt=-1.0 * u.Myr, n_steps=N_STEPS,
                            Integrator=DOPRI853Integrator)

    # xyz shape: (3, n_times, n_stars) in kpc
    fx = fwd.pos.xyz.to_value(u.kpc)
    bx = bwd.pos.xyz.to_value(u.kpc)
    # time axis: -1000..+1000 (backward reversed, drop duplicate t=0)
    pos = np.concatenate([bx[:, ::-1, :][:, :-1, :], fx], axis=1)
    pos = np.transpose(pos, (2, 1, 0)).astype(np.float32)  # (n, 2001, 3)

    # energy conservation over the full span, per star
    e_f = fwd.energy().to_value(u.km**2 / u.s**2)  # (n_times, n)
    e_b = bwd.energy().to_value(u.km**2 / u.s**2)
    e_all = np.concatenate([e_b, e_f], axis=0)
    de = (np.abs(e_all - e_all[0]).max(axis=0) / np.abs(e_all[0])).astype(np.float32)

    times = np.arange(-N_STEPS, N_STEPS + 1, dtype=np.float32)  # Myr

    with h5py.File(args.out, "w") as f:
        f.create_dataset("pos", data=pos, compression="lzf")
        f.create_dataset("t_myr", data=times)
        f.create_dataset("dE_over_E", data=de)
        f.create_dataset("source_id", data=np.asarray(t["source_id"], dtype=np.int64))
        f.attrs["row_start"] = args.start
        f.attrs["row_end"] = args.end
        f.attrs["potential"] = "MilkyWayPotential2022"
        f.attrs["frame"] = "astropy Galactocentric defaults"

    print(f"shard {args.out}: {n} stars, dE/E max {de.max():.2e}, "
          f"median {np.median(de):.2e}")


if __name__ == "__main__":
    main()
