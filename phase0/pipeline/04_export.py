"""Phase 0 export: pack fit parameters into the GPU binary.

Per star, little-endian, 112 bytes:
  f32 Rg, f32 phi0, f32 Om, f32 zc                (16 B header)
  12 lines (4 R, 4 z, 3 phi + 1 zero pad):
    f32 freq, f16 amp, f16 phase                  (8 B each)
Order: R lines 0..3, z lines 0..3, phi lines 0..2, pad line.
Amp and phase quantized to f16; freqs and header stay f32.

Also writes meta.bin (f16 bp_rp, f16 gmag per star), a JSON manifest with
reconstruction checks (computed from the QUANTIZED params, so the browser
GPU path can be verified bit-honestly), and quantization QC.
"""

import glob
import json
import numpy as np
import h5py
from astropy.table import Table

K_R, K_Z, K_P = 4, 4, 3  # exported line counts (phi line 4 dropped: pad)

fits = sorted(glob.glob("shards/fit_s0*.h5"))
D = {}
for k in ["Rg", "fR", "aR", "pR", "zc", "fZ", "aZ", "pZ",
          "ph0", "Om", "fP", "aP", "pP", "err500", "source_id"]:
    D[k] = np.concatenate([h5py.File(f)[k][:] for f in fits])

n = len(D["Rg"])
tbl = Table.read("/sessions/trusting-brave-fermat/mnt/new project for cv/"
                 "Orrery/phase0/data/gaia_dr3_6d_raw.csv", format="csv")
assert np.array_equal(np.asarray(tbl["source_id"]), D["source_id"])

# wrap phases into [-pi, pi] before f16 quantization
def wrap(p):
    return (p + np.pi) % (2 * np.pi) - np.pi

rec = np.zeros((n, 28), dtype=np.float32)  # 112 B = 28 f32 words
rec[:, 0] = D["Rg"]
rec[:, 1] = wrap(D["ph0"])
rec[:, 2] = D["Om"]
rec[:, 3] = D["zc"]

def pack_f16pair(a, b):
    """two f16 values into one u32 word (a = low 16 bits, b = high)."""
    ah = np.float16(a).view(np.uint16).astype(np.uint32)
    bh = np.float16(b).view(np.uint16).astype(np.uint32)
    return (bh << 16) | ah

w = 4
lines = []
for (fk, ak, pk, K) in [("fR", "aR", "pR", K_R), ("fZ", "aZ", "pZ", K_Z),
                        ("fP", "aP", "pP", K_P)]:
    for k in range(K):
        lines.append((D[fk][:, k], D[ak][:, k], wrap(D[pk][:, k])))
lines.append((np.zeros(n), np.zeros(n), np.zeros(n)))  # pad to 12

for (f, a, p) in lines:
    rec[:, w] = f.astype(np.float32)
    rec[:, w + 1] = pack_f16pair(a, p).view(np.float32)
    w += 2
assert w == 28

rec.tofile("web_params.bin")

# meta: color + magnitude
bp_rp = np.array([float(v) if v != "NULL" else np.nan for v in tbl["bp_rp"]])
gmag = np.asarray(tbl["phot_g_mean_mag"], dtype=np.float64)
bp_rp = np.nan_to_num(bp_rp, nan=0.8)
meta = np.zeros((n, 2), dtype=np.float16)
meta[:, 0] = bp_rp
meta[:, 1] = gmag
meta.tofile("web_meta.bin")

# reconstruction from QUANTIZED params, mirroring WGSL exactly
def recon_quant(t):
    Rg = rec[:, 0].astype(np.float64)
    ph0 = rec[:, 1].astype(np.float64)
    Om = rec[:, 2].astype(np.float64)
    zc = rec[:, 3].astype(np.float64)
    R = Rg.copy(); Z = zc.copy(); PHI = ph0 + Om * t
    for li in range(12):
        f = rec[:, 4 + 2 * li].astype(np.float64)
        u32 = rec[:, 5 + 2 * li].view(np.uint32)
        amp = u32.astype(np.uint32) & 0xFFFF
        pha = (u32 >> 16) & 0xFFFF
        amp = amp.astype(np.uint16).view(np.float16).astype(np.float64)
        pha = pha.astype(np.uint16).view(np.float16).astype(np.float64)
        term = amp * np.cos(f * t + pha)
        if li < K_R:
            R += term
        elif li < K_R + K_Z:
            Z += term
        else:
            PHI += term
    return np.stack([R * np.cos(PHI), R * np.sin(PHI), Z], axis=-1)

# quantization QC against the f64 fit reconstruction at t = +/-500, 0
def recon_f64(t):
    def ls(f, a, p):
        return (a * np.cos(f * t + p)).sum(axis=1)
    R = D["Rg"] + ls(D["fR"], D["aR"], D["pR"])
    Z = D["zc"] + ls(D["fZ"], D["aZ"], D["pZ"])
    PHI = D["ph0"] + D["Om"] * t + ls(D["fP"][:, :K_P], D["aP"][:, :K_P],
                                      D["pP"][:, :K_P])
    return np.stack([R * np.cos(PHI), R * np.sin(PHI), Z], axis=-1)

qerr = []
for t in (-500.0, 0.0, 500.0):
    dq = np.linalg.norm(recon_quant(t) - recon_f64(t), axis=1) * 1000
    qerr.append(float(np.median(dq)))
print(f"quantization-only error at -500/0/+500 Myr, median pc: "
      f"{qerr[0]:.2f} / {qerr[1]:.2f} / {qerr[2]:.2f}")

# browser self-check block: quantized-param positions for 5 stars
chk_idx = [0, 1234, 9999, 23456, 46000]
checks = {}
for t in (-500.0, 0.0, 500.0):
    P = recon_quant(t)
    checks[str(int(t))] = [[round(float(v), 5) for v in P[i]] for i in chk_idx]

manifest = {
    "n_stars": int(n),
    "bytes_per_star": 112,
    "layout": "f32 Rg, phi0, Om, zc; 12x(f32 freq, u32 f16amp|f16pha<<16); "
              "lines 0-3 R, 4-7 z, 8-10 phi, 11 pad",
    "t_range_myr": [-500, 500],
    "fit_window_myr": [-1000, 1000],
    "potential": "gala MilkyWayPotential2022",
    "frame": "astropy Galactocentric defaults (R0=8.122 kpc)",
    "err500_median_pc": float(np.median(D["err500"])),
    "err500_p95_pc": float(np.percentile(D["err500"], 95)),
    "bad_frac_gt200pc": float((D["err500"] > 200).mean()),
    "check_star_indices": chk_idx,
    "check_positions_kpc": checks,
}
with open("web_manifest.json", "w") as f:
    json.dump(manifest, f, indent=1)
print(f"wrote web_params.bin ({rec.nbytes/1e6:.1f} MB), web_meta.bin, manifest")
