<div align="center">

# Orrery

**A million real stars, their orbits computed on a supercomputer, replayed in your browser.**

[![Live demo](https://img.shields.io/badge/live-orrery--dua.pages.dev-E8B34B?style=for-the-badge)](https://orrery-dua.pages.dev)
&nbsp;
![WebGPU](https://img.shields.io/badge/WebGPU-raw%20WGSL-1f6feb?style=for-the-badge)
![Orbits](https://img.shields.io/badge/orbits-HPC%20precomputed-db6d28?style=for-the-badge)
![Dependencies](https://img.shields.io/badge/dependencies-none-2ea043?style=for-the-badge)
![License](https://img.shields.io/badge/license-MIT-8957e5?style=for-the-badge)

</div>

Orrery takes real stars from ESA's Gaia mission, integrates their orbits on a supercomputer, and replays them in the browser on a timeline you can scrub across half a billion years. Every point is a measured star. Drag the timeline and the Sun's neighbourhood shears and phase mixes, all of it driven by a hand-written WebGPU pipeline that rebuilds each star's position from a tiny orbital fit every frame.

## What you are looking at

- **Real data.** ESA Gaia DR3. It shows 1,013,200 stars, a uniform random sample of the 16,673,098 with a full 3D position and velocity that pass strict quality cuts. Distances from inverted parallaxes, in a Galactocentric frame.
- **A million stars on the GPU.** Each orbit is stored as a short quasi-periodic series of about 112 bytes. A WGSL compute shader evaluates it for every star at the current time and writes the positions a render pass draws.
- **Orbits from a supercomputer.** Every orbit was integrated over two billion years of model time as a test particle in a fixed Galactic potential, on the CINECA Galileo100 cluster, then fitted to the series the browser reads.
- **Click any star.** A GPU pick returns the star nearest the cursor with its Gaia id, photometry, guiding radius, orbital period, and swings, and draws its orbit as a ribbon.
- **Honest by design.** The reconstruction error is measured and shown, and the orbits a short series cannot describe are flagged rather than hidden.

## How it works

1. `pull_gaia.sh` pulls about a million Gaia DR3 stars with full 6D phase space from the AIP archive mirror.
2. `integrate_fit.py` integrates each orbit plus and minus one billion years (gala's MilkyWayPotential2022) and fits a quasi-periodic series in cylindrical coordinates. A SLURM array runs it across 34 tasks on Galileo100, one HDF5 shard each.
3. `qc_export.py` checks the fit accuracy and packs the per-star binary plus a manifest into progressive shards.
4. The page loads the shards and runs the reconstruction, rendering, GPU picking, and orbit ribbons in raw WebGPU. No three.js, no framework, no build step.

## Run it

The app is static. Rebuilding the star data needs an HPC login node with gala, astropy, h5py, numpy:

```bash
bash phase1/hpc/pull_gaia.sh <data-dir>      # ~1M Gaia DR3 6D stars (AIP TAP mirror)
sbatch phase1/hpc/job_array.sbatch           # integrate + fit, one shard per task
python phase1/hpc/qc_export.py --fits fits --csv <data-dir>/gaia_1m.csv --out phase1/web/shards
```

Then serve the app, which fetches the shards, so file:// will not work:

```bash
cd phase1/web && python -m http.server 8000
```

You need a WebGPU browser, so desktop Chrome or Edge. The 1M-star shards ship with the live site and are gitignored here; point `window.ORRERY_SHARD_BASE` at a hosted copy if you would rather not rebuild them.

## Honest notes

- It is a model, not a live simulation. The stars are real, but their motion assumes a smooth, fixed Galaxy with no bar, spiral arms, gas, or self-gravity, and each star is a test particle.
- Ensemble motion is trustworthy; any single star's position half a billion years away is an estimate. The median error is 17 parsecs at 500 Myr, and about 4.85 percent of orbits are flagged as poor fits.
- What you see is the solar neighbourhood shearing and phase mixing, not tidal streams forming. The sample is the bright subset Gaia can fully measure, mostly within a few thousand light-years of the Sun.

## Author

Built by **Dr. Safeer Ali Mirani**, GPU / XR / real-time visualisation engineer (PhD).

[safeer.ali.mirani@gmail.com](mailto:safeer.ali.mirani@gmail.com) · [Portfolio](https://safeeralimirani.netlify.app) · [GitHub](https://github.com/SafeerAliMirani) · [LinkedIn](https://www.linkedin.com/in/safeeralimirani)

## License

MIT. Gaia data credit: ESA / Gaia / DPAC. Compute: CINECA (Galileo100).
