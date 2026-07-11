"""Orrery Phase 1: aggregate QC over all fit shards, then pack the web
binaries (same 112 B/star format as Phase 0, validated there).

Run on the login node after the array finishes:
  python qc_export.py --fits fits --out web_shards --n-shards 16

Outputs: qc_report.txt, web_shards/params_XX.bin, web_shards/meta_XX.bin,
web_shards/ids_XX.bin, web_shards/manifest.json. Star order is the CSV row
order (random_index order), so every progressive loading stage is an
unbiased subsample of the whole.
"""

import argparse
import glob
import json
import os
import numpy as np
import h5py
from astropy.table import Table
import astropy.units as u

K_R, K_Z, K_P = 4, 4, 3


def model_constants():
    """vc(R) LUT and the Sun's own fitted series, for the app manifest."""
    import gala.dynamics as gd
    import gala.potential as gp
    from gala.integrate import DOPRI853Integrator
    from integrate_fit import fit_coord

    pot = gp.MilkyWayPotential2022()
    R_lut = np.linspace(0.5, 30.0, 64)
    q = np.zeros((3, 64))
    q[0] = R_lut
    vc = pot.circular_velocity(q * u.kpc).to_value(u.km / u.s)

    H = gp.Hamiltonian(pot)
    w0 = gd.PhaseSpacePosition(pos=[-8.122, 0, 0.0208] * u.kpc,
                               vel=[12.9, 245.6, 7.78] * u.km / u.s)
    fwd = H.integrate_orbit(w0, dt=1.0 * u.Myr, n_steps=1000,
                            Integrator=DOPRI853Integrator)
    bwd = H.integrate_orbit(w0, dt=-1.0 * u.Myr, n_steps=1000,
                            Integrator=DOPRI853Integrator)
    fx = fwd.pos.xyz.to_value(u.kpc)
    bx = bwd.pos.xyz.to_value(u.kpc)
    P = np.transpose(np.concatenate([bx[:, ::-1][:, :-1], fx], axis=1)[:, :, None],
                     (2, 1, 0))
    t = np.arange(-1000, 1001, dtype=np.float64)
    x, y, z = P[:, :, 0], P[:, :, 1], P[:, :, 2]
    R = np.hypot(x, y)
    phi = np.unwrap(np.arctan2(y, x), axis=1)
    Rg, _, fR, aR, pR, _ = fit_coord(R, t, K_R)
    zc, _, fZ, aZ, pZ, _ = fit_coord(z, t, K_Z)
    ph0, Om, fP, aP, pP, _ = fit_coord(phi, t, K_P, with_t=True)
    sun = {
        "Rg": float(Rg[0]), "ph0": float(ph0[0]), "Om": float(Om[0]),
        "zc": float(zc[0]),
        "lines": [[float(f_), float(a_), float(p_)] for (f_, a_, p_) in
                  [(fR[0, k], aR[0, k], pR[0, k]) for k in range(K_R)] +
                  [(fZ[0, k], aZ[0, k], pZ[0, k]) for k in range(K_Z)] +
                  [(fP[0, k], aP[0, k], pP[0, k]) for k in range(K_P)]],
    }
    lut = {"R_kpc": [round(float(r), 4) for r in R_lut],
           "vc_kms": [round(float(v), 2) for v in vc]}
    return lut, sun


def wrap(p):
    return (p + np.pi) % (2 * np.pi) - np.pi


def pack_f16pair(a, b):
    ah = np.float16(a).view(np.uint16).astype(np.uint32)
    bh = np.float16(b).view(np.uint16).astype(np.uint32)
    return (bh << 16) | ah


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fits", default="fits")
    ap.add_argument("--csv", default="data/gaia_1m.csv")
    ap.add_argument("--out", default="web_shards")
    ap.add_argument("--n-shards", type=int, default=16)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.fits, "fit_*.h5")))
    D = {}
    for k in ["Rg", "fR", "aR", "pR", "zc", "fZ", "aZ", "pZ",
              "ph0", "Om", "fP", "aP", "pP", "err500", "err_med",
              "dE_over_E", "source_id"]:
        D[k] = np.concatenate([h5py.File(f)[k][:] for f in files])
    n = len(D["Rg"])

    e5 = D["err500"]
    good = e5 <= 200.0
    lines = [
        f"stars: {n} from {len(files)} shards",
        f"err@500Myr: median {np.median(e5):.1f} pc, p95 "
        f"{np.percentile(e5, 95):.0f} pc, max {e5.max():.0f} pc",
        f"bad fraction (>200 pc): {(~good).mean()*100:.2f}%",
        f"good-set err@500Myr: median {np.median(e5[good]):.1f} pc, "
        f"p95 {np.percentile(e5[good], 95):.0f} pc",
        f"window-median error: median {np.median(D['err_med']):.1f} pc",
        f"energy drift dE/E: median {np.median(D['dE_over_E']):.1e}, "
        f"max {D['dE_over_E'].max():.1e}",
    ]
    report = "\n".join(lines)
    print(report)
    with open("qc_report.txt", "w") as f:
        f.write(report + "\n")

    # pack params (112 B/star)
    rec = np.zeros((n, 28), dtype=np.float32)
    rec[:, 0] = D["Rg"]
    rec[:, 1] = wrap(D["ph0"])
    rec[:, 2] = D["Om"]
    rec[:, 3] = D["zc"]
    w = 4
    entries = []
    for (fk, ak, pk, K) in [("fR", "aR", "pR", K_R), ("fZ", "aZ", "pZ", K_Z),
                            ("fP", "aP", "pP", K_P)]:
        for k in range(K):
            entries.append((D[fk][:, k], D[ak][:, k], wrap(D[pk][:, k])))
    entries.append((np.zeros(n), np.zeros(n), np.zeros(n)))
    for (f_, a_, p_) in entries:
        rec[:, w] = f_.astype(np.float32)
        rec[:, w + 1] = pack_f16pair(a_, p_).view(np.float32)
        w += 2

    tbl = Table.read(args.csv, format="csv")
    assert np.array_equal(np.asarray(tbl["source_id"]), D["source_id"])
    bp_rp = np.array([float(v) if str(v) != "NULL" else np.nan
                      for v in tbl["bp_rp"]])
    bp_rp = np.nan_to_num(bp_rp, nan=0.8)
    gmag = np.array([float(v) if str(v) != "NULL" else 13.0
                     for v in tbl["phot_g_mean_mag"]])
    meta = np.zeros((n, 4), dtype=np.float16)
    meta[:, 0] = bp_rp
    meta[:, 1] = gmag
    meta[:, 2] = np.minimum(e5, 60000).astype(np.float16)
    meta[:, 3] = good.astype(np.float16)
    ids = D["source_id"].astype(np.int64)

    per = (n + args.n_shards - 1) // args.n_shards
    shard_meta = []
    for s in range(args.n_shards):
        a, b = s * per, min((s + 1) * per, n)
        if a >= b:
            break
        rec[a:b].tofile(f"{args.out}/params_{s:02d}.bin")
        meta[a:b].tofile(f"{args.out}/meta_{s:02d}.bin")
        ids[a:b].tofile(f"{args.out}/ids_{s:02d}.bin")
        shard_meta.append({"shard": s, "n": int(b - a),
                           "params_bytes": int((b - a) * 112)})

    vc_lut, sun = model_constants()
    manifest = {
        "n_stars": int(n),
        "full_sample_count": 16673098,
        "vc_lut": vc_lut,
        "sun": sun,
        "bytes_per_star": 112,
        "layout": "f32 Rg, phi0, Om, zc; 12x(f32 freq, u32 f16amp|f16pha<<16); "
                  "lines 0-3 R, 4-7 z, 8-10 phi, 11 pad",
        "meta_layout": "f16 bp_rp, gmag, err500_pc, good_flag",
        "t_range_myr": [-500, 500],
        "fit_window_myr": [-1000, 1000],
        "potential": "gala MilkyWayPotential2022",
        "frame": "astropy Galactocentric defaults (R0=8.122 kpc)",
        "err500_median_pc": float(np.median(e5)),
        "err500_p95_pc": float(np.percentile(e5, 95)),
        "bad_frac_gt200pc": float((~good).mean()),
        "shards": shard_meta,
    }
    with open(f"{args.out}/manifest.json", "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"packed {n} stars into {len(shard_meta)} shards in {args.out}/")


if __name__ == "__main__":
    main()
