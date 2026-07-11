# Orrery

A million real Gaia stars, their orbits integrated on an HPC cluster and replayed in the browser on a scrubbable timeline. Raw WebGPU, no libraries.

Live: https://orrery-dua.pages.dev

## What it does

Every point is a real star from Gaia DR3 with a full 3D position and velocity. Their orbits are precomputed as test particles in a fixed Galactic potential, compressed to about 112 bytes each, and reconstructed on the GPU every frame. Scrub the timeline to watch the solar neighbourhood shear and phase mix over 500 million years; click a star for its dossier and orbit.

## Layout

```
phase0/            de-risk on 46k stars
  pipeline/        pull, integrate, fit, export, build
  web/             single-file viewer
phase1/
  hpc/             Galileo100 pipeline: pull, integrate + fit, SLURM array, QC/pack
  web/             the app (index.html, main.js)
```

## The data is not in the repo

The 1M-star shards (about 108 MB of binary) are served from Cloudflare Pages beside the app and are regenerable from the pipeline, so they are gitignored. Point `window.ORRERY_SHARD_BASE` at a hosted copy, or rebuild them.

## Rebuilding the data

On an HPC login node with gala, astropy, h5py, numpy:

```
bash phase1/hpc/pull_gaia.sh <data-dir>        # ~1M Gaia DR3 6D stars (AIP TAP mirror)
sbatch phase1/hpc/job_array.sbatch             # integrate + fit, one HDF5 shard per task
python phase1/hpc/qc_export.py --fits fits --csv <data-dir>/gaia_1m.csv --out phase1/web/shards
```

`integrate_fit.py` runs the orbit integration (gala MilkyWayPotential2022, plus and minus 1 Gyr) and the quasi-periodic fit; `qc_export.py` packs the per-star binary and the manifest.

## Running the app

Serve `phase1/web` over HTTP (it fetches shards, so file:// will not work) in a WebGPU browser:

```
cd phase1/web && python -m http.server 8000
```

## Stack

WebGPU / WGSL in the browser. Python (gala, astropy, numpy, h5py) offline. Data from Gaia DR3 via the AIP TAP mirror.

## Credits

Star data: ESA / Gaia / DPAC. Compute: CINECA Galileo100. Built by Safeer Ali Mirani.
