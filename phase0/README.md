# Orrery, Phase 0: de-risk

Goal: prove the two novel pieces of Orrery on a laptop-scale sample before spending
any HPC hours. (1) Compress real integrated orbits into a tiny per-star quasi-periodic
series. (2) Reconstruct positions from that series in a WGSL compute shader, scrubbed
in time. Both work.

## Result vs acceptance criteria

| Criterion | Target | Measured |
|---|---|---|
| Position error at t = +/-500 Myr, median | < ~20 pc | 10.8 pc |
| Position error at t = +/-500 Myr, 95th pct | < ~200 pc | 169 pc |
| Bad-fit fraction (kill if > 15-20%) | | 4.3% (err500 > 200 pc) |
| Rotation visually obvious | yes | yes: the solar-neighbourhood clump shears into a full annulus by t = +/-500 Myr |
| No shimmer or popping while scrubbing | yes | positions are a closed-form function of t; GPU output matches the CPU reference to 0.02 pc |

Quantization to the wire format (f16 amplitudes and phases) adds only ~0.45 pc median.
Anchors: circular speed at R0 is 229 km/s; the Sun's azimuthal period is 237 Myr,
consistent with its guiding radius (~8.7 kpc) in this potential; energy conservation
in the integrations is dE/E ~ 1e-14 median, 3e-7 worst.

The bad-fit tail is the expected population: hot orbits (median vertical amplitude
2.3 kpc vs 0.5 kpc for good fits). They stay in the render set here; Phase 1 flags
them and either restricts the render set or falls back to keyframes, with the
fraction disclosed.

## Data

46,197 real Gaia DR3 stars with full 6D phase space, pulled from the AIP mirror
(gaia.aip.de TAP, async, anonymous) after the ESA archive's shared anonymous quota
was full. Cuts: radial_velocity not null, parallax_over_error >= 10, ruwe < 1.4,
radial_velocity_error < 5, random_index < 5,000,000 (reproducible window).
Distances are 1/parallax (fine at this SNR). For Phase 1, register an ESA Cosmos
account for the bigger login quota, or chunk on random_index against AIP.

## Pipeline (in order)

- `pipeline/01_integrate.py` - ICRS to Galactocentric (astropy defaults), gala
  MilkyWayPotential2022, DOPRI853, +/-1 Gyr at 1 Myr output. Run in row batches;
  writes HDF5 shards with per-star energy drift.
- `pipeline/02_fit.py` - per coordinate (cylindrical R, z, phi): Hann-windowed FFT
  peak, local zoom-DFT frequency refinement (NAFF in spirit), deflation with peak
  masking, batched normal-equation joint refit, one re-refinement pass against the
  joint residual. K = 4 lines in R, 4 in z, 3 + secular in phi. The phi ramp
  (Omega t) must be removed before spectral extraction and refit jointly with the
  lines; getting this wrong costs two orders of magnitude in error.
- `pipeline/03_qc.py` - aggregate acceptance stats, tail characterization, solar
  anchors, render-set flags.
- `pipeline/04_export.py` - packs 112 B/star: f32 Rg, phi0, Omega, zc; 12 lines of
  f32 freq + packed f16 amp/phase (layout documented in the file and manifest).
  Emits web_params.bin, web_meta.bin (f16 bp_rp + gmag), web_manifest.json with
  5-star reference positions computed from the quantized params.
- `pipeline/05_build_web.py` - injects the binaries (base64) and manifest into
  `web/template.html`, producing the self-contained `web/orrery_phase0.html`.

## The page

`web/orrery_phase0.html` (7.2 MB, double-click to open, no server needed). Raw
WebGPU: a compute pass evaluates each star's series at time t (bitcast +
unpack2x16float, a dozen sincos) into a storage buffer; a render pass draws
instanced additive Gaussian sprites tinted by BP-RP. Time slider +/-500 Myr with
play speeds, orbit/zoom camera, exposure. On load it recomputes 5 reference stars
at t = -500/0/+500 and compares against the manifest: the badge shows pass/fail.

## Honest limits (carried into every later phase)

The potential is an assumed smooth model (no bar, no spiral arms, no GMCs, no
self-gravity); stars are test particles. The RV sample is the bright subset,
mostly within a few kpc of the Sun. Ensembles are meaningful; single stars are
indicative. Phase mixing is real; this does not show tidal streams forming.

## Phase 1 deltas

1M stars via SLURM array on Galileo100, same fit code per shard; progressive
shard loading (first ~128k interactive in seconds); HDR pipeline with tonemap
pass; click-to-pick with orbit ribbons; Sun marker and comoving mode; honesty
panel; DR4 rerun scheduled Dec 2026.
