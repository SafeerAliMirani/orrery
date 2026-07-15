# Orrery, Phase 1

Two halves. `hpc/` is the Galileo100 pipeline that produces the 1M-star data.
`web/` is the app; it loads the shards that pipeline packs.

## App (web/)

Raw WebGPU, no libraries, no build step. `python -m http.server` in `web/`
and open the page (it fetches shards, so file:// is not enough).

- Progressive streaming: renders from the first shard while the rest load;
  every loading stage is an unbiased subsample (stars stay in random_index
  order).
- Compute pass evaluates each star's quasi-periodic series at time t
  (112 B/star: bitcast f32 header + unpack2x16float lines, ~12 sincos).
- HDR path: additive Gaussian sprites into rgba16float, then a tonemap pass
  (1 - exp(-c), gamma) with an exposure control.
- Click-to-pick: a compute pass runs point-to-ray angle over all stars with
  atomicMin on a 12-bit-angle | 20-bit-index key (fits 1M stars), one 4-byte
  readback. The dossier is computed CPU-side from the same quantized
  parameters the GPU uses: source_id, G, BP-RP, current distance from the
  Sun, guiding radius, orbital period, radial and vertical swing,
  vphi - vc, fit error, Gaia archive link, and a poor-fit warning when the
  star is in the disclosed 4.3% tail.
- Orbit ribbon: on pick, a 512-point compute pass evaluates that star's
  closed-form path over +/-500 Myr into a line-strip buffer; alpha peaks
  near the current time. This is the signature interaction.
- Colour modes: BP-RP through a blackbody-ish ramp, or kinematics
  (vphi - vc(Rg), diverging map; the blue skew of the disc is real
  asymmetric drift, and retrograde halo stars saturate deep blue).
- Ride with the Sun: camera target follows the Sun's own fitted series
  (fit error ~11 pc, period 236.9 Myr) with a galactic-year counter.
  Markers: Sun ring, picked-star ring, Galactic-centre cross. WASD/QE pan,
  drag orbit, wheel zoom, space to play.
- Honesty panel: collapsible, includes the exact full-sample count
  (16,673,098) and the bad-fit fraction from the manifest.

## Data contract (shards/manifest.json)

`params_XX.bin` 112 B/star (f32 Rg, phi0, Omega, zc + 12 x (f32 freq,
u32 f16 amp | f16 phase << 16); lines 0-3 R, 4-7 z, 8-10 phi, 11 pad),
`meta_XX.bin` (4 x f16: bp_rp, gmag, err500_pc, good_flag), `ids_XX.bin`
(i64 source_id). The manifest carries vc(R) LUT and the Sun series so the
app has no hardcoded model numbers.

## Status

Live at https://orrery-dua.pages.dev. The app and all 1M-star shards are
served same-origin from Cloudflare Pages, so there is no separate data host
and no CORS to configure.

Remaining: visual tuning at 1M (sprite size, exposure defaults, maybe a
half-res HDR target), mobile and touch input, and the DR4 rerun when it
lands in December 2026.
