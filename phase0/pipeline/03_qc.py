"""Phase 0 QC: aggregate fit errors, check acceptance, characterize the
bad-fit tail, verify the solar orbit anchor, write the render-set flag."""

import glob
import numpy as np
import h5py

import astropy.units as u
import gala.dynamics as gd
import gala.potential as gp
from gala.integrate import DOPRI853Integrator

BAD_PC = 200.0

fits = sorted(glob.glob("shards/fit_s0*.h5"))
cols = {}
for k in ["err500", "err_med", "errmax", "Rg", "Om", "zc", "source_id"]:
    cols[k] = np.concatenate([h5py.File(f)[k][:] for f in fits])
aZ = np.concatenate([h5py.File(f)["aZ"][:] for f in fits])
aR = np.concatenate([h5py.File(f)["aR"][:] for f in fits])

n = len(cols["err500"])
e5 = cols["err500"]
good = e5 <= BAD_PC

print(f"stars: {n}")
print(f"ALL   err@500Myr: median {np.median(e5):.1f} pc, p95 {np.percentile(e5,95):.0f} pc, "
      f"max {e5.max():.0f} pc")
print(f"bad fraction (err500>{BAD_PC:.0f} pc): {(~good).mean()*100:.1f}%")
print(f"GOOD  err@500Myr: median {np.median(e5[good]):.1f} pc, "
      f"p95 {np.percentile(e5[good],95):.0f} pc")
print(f"GOOD  window-median: median {np.median(cols['err_med'][good]):.1f} pc")

# who is the tail: guiding radius, vertical + radial amplitude, rotation
zmax = np.abs(aZ).sum(axis=1)
ramp = np.abs(aR).sum(axis=1)
vphi = cols["Rg"] * cols["Om"] * 977.79  # kpc/Myr -> km/s
for name, m in [("good", good), ("bad", ~good)]:
    print(f"{name}: Rg med {np.median(cols['Rg'][m]):.2f} kpc | "
          f"zamp med {np.median(zmax[m])*1000:.0f} pc | "
          f"Ramp med {np.median(ramp[m])*1000:.0f} pc | "
          f"v_phi med {np.median(np.abs(vphi[m])):.0f} km/s")

# solar anchor: azimuthal period in MilkyWayPotential2022
pot = gp.MilkyWayPotential2022()
H = gp.Hamiltonian(pot)
w0 = gd.PhaseSpacePosition(pos=[-8.122, 0, 0.0208] * u.kpc,
                           vel=[12.9, 245.6, 7.78] * u.km / u.s)
o = H.integrate_orbit(w0, dt=0.5 * u.Myr, n_steps=4000,
                      Integrator=DOPRI853Integrator)
phi = np.unwrap(np.arctan2(o.pos.y.to_value(u.kpc), o.pos.x.to_value(u.kpc)))
T_sun = 2 * np.pi / np.abs(np.polyfit(o.t.to_value(u.Myr), phi, 1)[0])
print(f"Sun azimuthal period: {T_sun:.0f} Myr (expect ~212-230)")
vc = pot.circular_velocity([-8.122, 0, 0] * u.kpc)[0].to_value(u.km/u.s)
print(f"circular speed at R0: {vc:.0f} km/s (expect ~229)")

with h5py.File("shards/qc_flags.h5", "w") as f:
    f.create_dataset("source_id", data=cols["source_id"])
    f.create_dataset("good", data=good.astype(np.uint8))
    f.create_dataset("err500", data=e5.astype(np.float32))
