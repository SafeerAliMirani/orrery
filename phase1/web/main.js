"use strict";
/* Orrery Phase 1 app. Raw WebGPU, no libraries.
   Data: shards/manifest.json + params_XX.bin (112 B/star), meta_XX.bin
   (f16 bp_rp, gmag, err500, good), ids_XX.bin (i64 source_id).
   Galactic coords in buffers; world = (x, z, -y) for display, y up. */

const WORKGROUP = 64;
const PICK_MAX_ANGLE = 0.035;      // rad, click acceptance cone
const RIBBON_N = 512;
// override with window.ORRERY_SHARD_BASE to serve shards from elsewhere
const SHARD_BASE = ((typeof window !== "undefined" && window.ORRERY_SHARD_BASE)
  || "shards") + "/";

/* ---------------- WGSL ---------------- */

const CS_POS = `
struct U { t: f32, n: u32, pad0: f32, pad1: f32 }
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> params: array<u32>;
@group(0) @binding(2) var<storage, read_write> pos: array<vec4f>;

@compute @workgroup_size(${WORKGROUP})
fn cs(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= u.n) { return; }
  let b = i * 28u;
  var R   = bitcast<f32>(params[b]);
  var PHI = bitcast<f32>(params[b + 1u]) + bitcast<f32>(params[b + 2u]) * u.t;
  var Z   = bitcast<f32>(params[b + 3u]);
  for (var k = 0u; k < 12u; k = k + 1u) {
    let f = bitcast<f32>(params[b + 4u + 2u * k]);
    let ap = unpack2x16float(params[b + 5u + 2u * k]);
    let term = ap.x * cos(f * u.t + ap.y);
    if (k < 4u) { R += term; }
    else if (k < 8u) { Z += term; }
    else { PHI += term; }
  }
  pos[i] = vec4f(R * cos(PHI), R * sin(PHI), Z, 1.0);
}`;

const RS_STARS = `
struct VU { mvp: mat4x4f, right: vec4f, up: vec4f, size: f32, p0: f32, p1: f32, p2: f32 }
@group(0) @binding(0) var<uniform> vu: VU;
@group(0) @binding(1) var<storage, read> pos: array<vec4f>;
@group(0) @binding(2) var<storage, read> col: array<vec4f>;

struct VOut {
  @builtin(position) clip: vec4f,
  @location(0) uv: vec2f,
  @location(1) tint: vec3f,
}

@vertex
fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  var corners = array<vec2f, 4>(
    vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0), vec2f(1.0, 1.0));
  let c = corners[vi];
  let p = pos[ii].xyz;
  let world0 = vec3f(p.x, p.z, -p.y);
  let cd = col[ii];
  let s = vu.size * cd.w;
  let world = world0 + vu.right.xyz * (c.x * s) + vu.up.xyz * (c.y * s);
  var o: VOut;
  o.clip = vu.mvp * vec4f(world, 1.0);
  o.uv = c;
  o.tint = cd.rgb;
  return o;
}

@fragment
fn fs(v: VOut) -> @location(0) vec4f {
  let d2 = dot(v.uv, v.uv);
  let a = exp(-5.0 * d2);
  return vec4f(v.tint * a, 1.0);
}`;

const RS_TONEMAP = `
struct TU { exposure: f32, p0: f32, p1: f32, p2: f32 }
@group(0) @binding(0) var<uniform> tu: TU;
@group(0) @binding(1) var hdr: texture_2d<f32>;

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  var pts = array<vec2f, 3>(vec2f(-1.0, -3.0), vec2f(3.0, 1.0), vec2f(-1.0, 1.0));
  return vec4f(pts[vi], 0.0, 1.0);
}

@fragment
fn fs(@builtin(position) fc: vec4f) -> @location(0) vec4f {
  let c = textureLoad(hdr, vec2i(fc.xy), 0).rgb * tu.exposure;
  let mapped = vec3f(1.0) - exp(-c);
  let g = pow(mapped, vec3f(1.0 / 2.2));
  return vec4f(g, 1.0);
}`;

const CS_PICK = `
struct PU { ro: vec4f, rd: vec4f, n: u32, p0: u32, p1: u32, p2: u32 }
@group(0) @binding(0) var<uniform> pu: PU;
@group(0) @binding(1) var<storage, read> pos: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> best: atomic<u32>;

@compute @workgroup_size(${WORKGROUP})
fn cs(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= pu.n) { return; }
  let v = pos[i].xyz - pu.ro.xyz;
  let along = dot(v, pu.rd.xyz);
  if (along < 0.2) { return; }
  let perp = length(v - along * pu.rd.xyz);
  let ang = perp / along;
  if (ang > ${PICK_MAX_ANGLE}) { return; }
  // key: 12-bit quantized angle | 20-bit index (fits 1M stars)
  let q = min(u32(ang * 131072.0), 0xFFEu);
  atomicMin(&best, (q << 20u) | i);
}`;

const CS_RIBBON = `
struct RU { t0: f32, dt: f32, index: u32, n: u32 }
@group(0) @binding(0) var<uniform> ru: RU;
@group(0) @binding(1) var<storage, read> params: array<u32>;
@group(0) @binding(2) var<storage, read_write> line_out: array<vec4f>;

@compute @workgroup_size(${WORKGROUP})
fn cs(@builtin(global_invocation_id) gid: vec3u) {
  let s = gid.x;
  if (s >= ru.n) { return; }
  let t = ru.t0 + f32(s) * ru.dt;
  let b = ru.index * 28u;
  var R   = bitcast<f32>(params[b]);
  var PHI = bitcast<f32>(params[b + 1u]) + bitcast<f32>(params[b + 2u]) * t;
  var Z   = bitcast<f32>(params[b + 3u]);
  for (var k = 0u; k < 12u; k = k + 1u) {
    let f = bitcast<f32>(params[b + 4u + 2u * k]);
    let ap = unpack2x16float(params[b + 5u + 2u * k]);
    let term = ap.x * cos(f * t + ap.y);
    if (k < 4u) { R += term; }
    else if (k < 8u) { Z += term; }
    else { PHI += term; }
  }
  line_out[s] = vec4f(R * cos(PHI), R * sin(PHI), Z, t);
}`;

const RS_RIBBON = `
struct RV { mvp: mat4x4f, tnow: f32, p0: f32, p1: f32, p2: f32 }
@group(0) @binding(0) var<uniform> rv: RV;
@group(0) @binding(1) var<storage, read> line_in: array<vec4f>;

struct VOut { @builtin(position) clip: vec4f, @location(0) w: f32 }

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VOut {
  let p = line_in[vi];
  let world = vec3f(p.x, p.z, -p.y);
  var o: VOut;
  o.clip = rv.mvp * vec4f(world, 1.0);
  let dt = abs(p.w - rv.tnow);
  o.w = 0.10 + 0.55 * exp(-dt / 180.0);
  return o;
}

@fragment
fn fs(v: VOut) -> @location(0) vec4f {
  return vec4f(vec3f(1.0, 0.78, 0.35) * v.w, 1.0);
}`;

const RS_MARKERS = `
struct MU { mvp: mat4x4f, right: vec4f, up: vec4f, sun: vec4f, pick: vec4f }
@group(0) @binding(0) var<uniform> mu: MU;

struct VOut {
  @builtin(position) clip: vec4f,
  @location(0) uv: vec2f,
  @location(1) @interpolate(flat) kind: u32,
}

@vertex
fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  var corners = array<vec2f, 4>(
    vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0), vec2f(1.0, 1.0));
  let c = corners[vi];
  var center = vec3f(0.0);
  var s = 0.30;
  if (ii == 0u) { center = mu.sun.xyz; s = 0.22; }
  if (ii == 1u) { center = mu.pick.xyz; s = 0.30; }
  if (ii == 2u) { s = 0.45; }
  let world = center + mu.right.xyz * (c.x * s) + mu.up.xyz * (c.y * s);
  var o: VOut;
  o.clip = mu.mvp * vec4f(world, 1.0);
  if (ii == 1u && mu.pick.w < 0.5) { o.clip = vec4f(0.0, 0.0, -2.0, 1.0); }
  o.uv = c;
  o.kind = ii;
  return o;
}

@fragment
fn fs(v: VOut) -> @location(0) vec4f {
  let r = length(v.uv);
  if (v.kind == 2u) {
    // galactic centre: thin cross
    let ax = min(abs(v.uv.x), abs(v.uv.y));
    if (ax > 0.09 || r > 1.0) { discard; }
    return vec4f(0.85, 0.30, 0.25, 0.85);
  }
  // rings: sun (yellow), picked star (amber)
  if (r < 0.62 || r > 0.95) { discard; }
  if (v.kind == 0u) { return vec4f(1.0, 0.9, 0.4, 0.9); }
  return vec4f(1.0, 0.65, 0.2, 0.95);
}`;

/* ---------------- small math ---------------- */

function mat4mul(a, b) {
  const o = new Float32Array(16);
  for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) {
    let s = 0;
    for (let k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k];
    o[c * 4 + r] = s;
  }
  return o;
}
function persp(fovy, asp, near, far) {
  const f = 1 / Math.tan(fovy / 2), o = new Float32Array(16);
  o[0] = f / asp; o[5] = f; o[10] = far / (near - far); o[11] = -1;
  o[14] = near * far / (near - far);
  return o;
}
function lookAt(eye, at, up) {
  const z = norm3([eye[0] - at[0], eye[1] - at[1], eye[2] - at[2]]);
  const x = norm3(cross(up, z));
  const y = cross(z, x);
  const o = new Float32Array(16);
  o[0] = x[0]; o[4] = x[1]; o[8] = x[2];
  o[1] = y[0]; o[5] = y[1]; o[9] = y[2];
  o[2] = z[0]; o[6] = z[1]; o[10] = z[2];
  o[12] = -dot3(x, eye); o[13] = -dot3(y, eye); o[14] = -dot3(z, eye);
  o[15] = 1;
  return { m: o, x, y, z };
}
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
                         a[0] * b[1] - a[1] * b[0]];
const dot3 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const norm3 = a => { const l = Math.hypot(...a); return [a[0] / l, a[1] / l, a[2] / l]; };

function f16(h) {
  const s = (h & 0x8000) ? -1 : 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
  if (e === 0) return s * m * Math.pow(2, -24);
  if (e === 31) return m ? NaN : s * Infinity;
  return s * (1 + m / 1024) * Math.pow(2, e - 15);
}

function tintOf(bpRp) {
  const t = Math.max(0, Math.min(1, (bpRp + 0.3) / 3.0));
  const stops = [
    [0.62, 0.75, 1.00], [0.80, 0.87, 1.00], [1.00, 0.98, 0.95],
    [1.00, 0.86, 0.62], [1.00, 0.62, 0.35], [1.00, 0.45, 0.25]];
  const x = t * (stops.length - 1), i = Math.min(stops.length - 2, Math.floor(x));
  const f = x - i, A = stops[i], B = stops[i + 1];
  return [A[0] + f * (B[0] - A[0]), A[1] + f * (B[1] - A[1]), A[2] + f * (B[2] - A[2])];
}
function diverging(dv) {
  // blue (lagging / retrograde) .. white .. red (leading), +/-80 km/s
  const t = Math.max(-1, Math.min(1, dv / 80));
  if (t < 0) return [1 + t * 0.75, 1 + t * 0.55, 1.0];
  return [1.0, 1 - t * 0.60, 1 - t * 0.80];
}

/* ---------------- app state ---------------- */

const KMS_PER_KPCMYR = 977.792;
let device, ctx, fmt, canvas;
let manifest, nTotal = 0, alive = 0;
let paramsCPU;                 // Float32Array view of all params (CPU copy)
let paramsU32;                 // Uint32Array view of same buffer
let metaCPU;                   // Uint16Array
let idsCPU;                    // BigInt64Array
let bufParams, bufPos, bufColA, bufColB, bufRibbon;
let uPos, uStar, uTone, uPick, uRibGen, uRibDraw, uMark;
let pipePos, pipeStars, pipeTone, pipePick, pipeRibGen, pipeRibDraw, pipeMark;
let bgPos, bgStarsA, bgStarsB, bgPick, bgRibGen, bgRibDraw, bgMark, bgTone;
let bufPickResult, bufPickRead;
let hdrTex, hdrView;
let sunFn, vcOf, sunPeriod;
let picked = -1, pickPending = null, ribbonValid = false;

let t = 0, playing = false, lastTs = 0;
let az = -0.5, el = 0.95, dist = 38, target = [0, 0, 0];
let comoving = false, markers = true, cmode = 0;
let camBasis = null, curProj = null, curMvp = null;

const $ = id => document.getElementById(id);

// analytics: one hit per label per session, kept in memory
const trackFired = new Set();
function trackOnce(type, label) {
  const k = type + "/" + label;
  if (trackFired.has(k)) return;
  trackFired.add(k);
  if (window.track) window.track(type, label);
}

/* ---------------- data loading ---------------- */

async function loadManifest() {
  manifest = await (await fetch(SHARD_BASE + "manifest.json")).json();
  nTotal = manifest.n_stars;
  $("hmed").textContent = manifest.err500_median_pc.toFixed(0);
  $("hbad").textContent = (manifest.bad_frac_gt200pc * 100).toFixed(1) + "%";
  $("hcount").textContent = (manifest.full_sample_count || 0).toLocaleString();
  if (manifest.dev_data) $("devnote").style.display = "block";

  const full = (manifest.full_sample_count || 0).toLocaleString();
  $("a-shown").textContent = nTotal.toLocaleString();
  $("a-full").textContent = full;
  $("a-err").textContent = manifest.err500_median_pc.toFixed(0);
  $("a-bad").textContent = (manifest.bad_frac_gt200pc * 100).toFixed(1) + "%";

  const lut = manifest.vc_lut;
  vcOf = R => {
    const xs = lut.R_kpc, ys = lut.vc_kms;
    if (R <= xs[0]) return ys[0];
    if (R >= xs[xs.length - 1]) return ys[ys.length - 1];
    let i = 0;
    while (xs[i + 1] < R) i++;
    const f = (R - xs[i]) / (xs[i + 1] - xs[i]);
    return ys[i] + f * (ys[i + 1] - ys[i]);
  };
  const sun = manifest.sun;
  sunPeriod = 2 * Math.PI / Math.abs(sun.Om);
  sunFn = tt => {
    let R = sun.Rg, Z = sun.zc, PHI = sun.ph0 + sun.Om * tt;
    sun.lines.forEach(([f, a, p], k) => {
      const term = a * Math.cos(f * tt + p);
      if (k < 4) R += term; else if (k < 8) Z += term; else PHI += term;
    });
    return [R * Math.cos(PHI), R * Math.sin(PHI), Z];
  };

  allocate();

  const ab = new ArrayBuffer(nTotal * 112);
  paramsCPU = new Float32Array(ab);
  paramsU32 = new Uint32Array(ab);
  metaCPU = new Uint16Array(nTotal * 4);
  idsCPU = new BigInt64Array(nTotal);
}

async function streamShards() {
  const ab = paramsCPU.buffer;
  let off = 0;
  for (const sh of manifest.shards) {
    const s2 = String(sh.shard).padStart(2, "0");
    const [pb, mb, ib] = await Promise.all([
      fetch(`${SHARD_BASE}params_${s2}.bin`).then(r => r.arrayBuffer()),
      fetch(`${SHARD_BASE}meta_${s2}.bin`).then(r => r.arrayBuffer()),
      fetch(`${SHARD_BASE}ids_${s2}.bin`).then(r => r.arrayBuffer()),
    ]);
    new Uint8Array(ab, off * 112, sh.n * 112).set(new Uint8Array(pb));
    metaCPU.set(new Uint16Array(mb), off * 4);
    idsCPU.set(new BigInt64Array(ib), off);
    device.queue.writeBuffer(bufParams, off * 112, pb);
    fillColors(off, sh.n);
    off += sh.n;
    alive = off;
    $("load").innerHTML = `<b>${alive.toLocaleString()}</b> / ` +
      `${nTotal.toLocaleString()} stars streamed`;
  }
  $("load").innerHTML = `<b>${alive.toLocaleString()}</b> real stars`;
}

function fillColors(off, n) {
  const A = new Float32Array(n * 4), B = new Float32Array(n * 4);
  for (let j = 0; j < n; j++) {
    const i = off + j;
    const bpRp = f16(metaCPU[4 * i]), g = f16(metaCPU[4 * i + 1]);
    let w = Math.pow(10, -0.4 * (g - 12.0));
    w = Math.max(0.10, Math.min(3.5, w));
    const bright = 0.35 + 0.65 * Math.min(1.5, w);
    const size = 0.75 + 0.35 * Math.min(1.8, Math.sqrt(w));
    const tint = tintOf(bpRp);
    A.set([tint[0] * bright, tint[1] * bright, tint[2] * bright, size], 4 * j);
    const Rg = paramsCPU[i * 28], Om = paramsCPU[i * 28 + 2];
    // signed azimuthal speed minus the local circular speed; retrograde
    // stars land far on the blue side and saturate the map
    const dv = Rg * Om * KMS_PER_KPCMYR - vcOf(Rg);
    const kt = diverging(dv);
    B.set([kt[0] * bright, kt[1] * bright, kt[2] * bright, size], 4 * j);
  }
  device.queue.writeBuffer(bufColA, off * 16, A);
  device.queue.writeBuffer(bufColB, off * 16, B);
}

/* ---------------- gpu setup ---------------- */

function allocate() {
  bufParams = device.createBuffer({ size: nTotal * 112,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  bufPos = device.createBuffer({ size: nTotal * 16,
    usage: GPUBufferUsage.STORAGE });
  bufColA = device.createBuffer({ size: nTotal * 16,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  bufColB = device.createBuffer({ size: nTotal * 16,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  bufRibbon = device.createBuffer({ size: RIBBON_N * 16,
    usage: GPUBufferUsage.STORAGE });
  bufPickResult = device.createBuffer({ size: 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
  bufPickRead = device.createBuffer({ size: 4,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

  uPos = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  uStar = device.createBuffer({ size: 112, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  uTone = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  uPick = device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  uRibGen = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  uRibDraw = device.createBuffer({ size: 80, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  uMark = device.createBuffer({ size: 128, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

  const sm = s => device.createShaderModule({ code: s });
  pipePos = device.createComputePipeline({ layout: "auto",
    compute: { module: sm(CS_POS), entryPoint: "cs" } });
  pipePick = device.createComputePipeline({ layout: "auto",
    compute: { module: sm(CS_PICK), entryPoint: "cs" } });
  pipeRibGen = device.createComputePipeline({ layout: "auto",
    compute: { module: sm(CS_RIBBON), entryPoint: "cs" } });

  const addBlend = { color: { srcFactor: "one", dstFactor: "one" },
                     alpha: { srcFactor: "one", dstFactor: "one" } };
  const starsMod = sm(RS_STARS);
  pipeStars = device.createRenderPipeline({ layout: "auto",
    vertex: { module: starsMod, entryPoint: "vs" },
    fragment: { module: starsMod, entryPoint: "fs",
      targets: [{ format: "rgba16float", blend: addBlend }] },
    primitive: { topology: "triangle-strip" } });
  const ribMod = sm(RS_RIBBON);
  pipeRibDraw = device.createRenderPipeline({ layout: "auto",
    vertex: { module: ribMod, entryPoint: "vs" },
    fragment: { module: ribMod, entryPoint: "fs",
      targets: [{ format: "rgba16float", blend: addBlend }] },
    primitive: { topology: "line-strip" } });
  const toneMod = sm(RS_TONEMAP);
  pipeTone = device.createRenderPipeline({ layout: "auto",
    vertex: { module: toneMod, entryPoint: "vs" },
    fragment: { module: toneMod, entryPoint: "fs", targets: [{ format: fmt }] },
    primitive: { topology: "triangle-list" } });
  const markMod = sm(RS_MARKERS);
  pipeMark = device.createRenderPipeline({ layout: "auto",
    vertex: { module: markMod, entryPoint: "vs" },
    fragment: { module: markMod, entryPoint: "fs", targets: [{ format: fmt,
      blend: { color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha" },
               alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha" } } }] },
    primitive: { topology: "triangle-strip" } });

  bgPos = device.createBindGroup({ layout: pipePos.getBindGroupLayout(0), entries: [
    { binding: 0, resource: { buffer: uPos } },
    { binding: 1, resource: { buffer: bufParams } },
    { binding: 2, resource: { buffer: bufPos } }] });
  bgStarsA = device.createBindGroup({ layout: pipeStars.getBindGroupLayout(0), entries: [
    { binding: 0, resource: { buffer: uStar } },
    { binding: 1, resource: { buffer: bufPos } },
    { binding: 2, resource: { buffer: bufColA } }] });
  bgStarsB = device.createBindGroup({ layout: pipeStars.getBindGroupLayout(0), entries: [
    { binding: 0, resource: { buffer: uStar } },
    { binding: 1, resource: { buffer: bufPos } },
    { binding: 2, resource: { buffer: bufColB } }] });
  bgPick = device.createBindGroup({ layout: pipePick.getBindGroupLayout(0), entries: [
    { binding: 0, resource: { buffer: uPick } },
    { binding: 1, resource: { buffer: bufPos } },
    { binding: 2, resource: { buffer: bufPickResult } }] });
  bgRibGen = device.createBindGroup({ layout: pipeRibGen.getBindGroupLayout(0), entries: [
    { binding: 0, resource: { buffer: uRibGen } },
    { binding: 1, resource: { buffer: bufParams } },
    { binding: 2, resource: { buffer: bufRibbon } }] });
  bgRibDraw = device.createBindGroup({ layout: pipeRibDraw.getBindGroupLayout(0), entries: [
    { binding: 0, resource: { buffer: uRibDraw } },
    { binding: 1, resource: { buffer: bufRibbon } }] });
  bgMark = device.createBindGroup({ layout: pipeMark.getBindGroupLayout(0), entries: [
    { binding: 0, resource: { buffer: uMark } }] });
}

function makeHdr(w, h) {
  if (hdrTex) hdrTex.destroy();
  hdrTex = device.createTexture({ size: [w, h], format: "rgba16float",
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
  hdrView = hdrTex.createView();
  bgTone = device.createBindGroup({ layout: pipeTone.getBindGroupLayout(0), entries: [
    { binding: 0, resource: { buffer: uTone } },
    { binding: 1, resource: hdrView }] });
}

/* ---------------- picking ---------------- */

function requestPick(px, py) {
  if (!camBasis || alive === 0) return;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  const ndcX = (px / w) * 2 - 1, ndcY = 1 - (py / h) * 2;
  const tanF = Math.tan(0.45), asp = w / h;
  const d = norm3([
    -camBasis.z[0] + ndcX * tanF * asp * camBasis.x[0] + ndcY * tanF * camBasis.y[0],
    -camBasis.z[1] + ndcX * tanF * asp * camBasis.x[1] + ndcY * tanF * camBasis.y[1],
    -camBasis.z[2] + ndcX * tanF * asp * camBasis.x[2] + ndcY * tanF * camBasis.y[2]]);
  // world -> galactic: g = (w.x, -w.z, w.y)
  const roG = [camBasis.eye[0], -camBasis.eye[2], camBasis.eye[1]];
  const rdG = [d[0], -d[2], d[1]];
  pickPending = { roG, rdG };
}

let pickBusy = false;
async function runPick() {
  if (pickBusy) { pickPending = null; return; }
  pickBusy = true;
  const { roG, rdG } = pickPending;
  pickPending = null;
  device.queue.writeBuffer(uPick, 0, new Float32Array([...roG, 0, ...rdG, 0]));
  device.queue.writeBuffer(uPick, 32, new Uint32Array([alive, 0, 0, 0]));
  device.queue.writeBuffer(bufPickResult, 0, new Uint32Array([0xFFFFFFFF]));
  const enc = device.createCommandEncoder();
  const p = enc.beginComputePass();
  p.setPipeline(pipePick);
  p.setBindGroup(0, bgPick);
  p.dispatchWorkgroups(Math.ceil(alive / WORKGROUP));
  p.end();
  enc.copyBufferToBuffer(bufPickResult, 0, bufPickRead, 0, 4);
  device.queue.submit([enc.finish()]);
  await bufPickRead.mapAsync(GPUMapMode.READ);
  const v = new Uint32Array(bufPickRead.getMappedRange())[0];
  bufPickRead.unmap();
  pickBusy = false;
  if (v === 0xFFFFFFFF) { closeDossier(); return; }
  showDossier(v & 0xFFFFF);
}

function starPos(i, tt) {
  const b = i * 28;
  let R = paramsCPU[b], PHI = paramsCPU[b + 1] + paramsCPU[b + 2] * tt,
      Z = paramsCPU[b + 3];
  for (let k = 0; k < 12; k++) {
    const f = paramsCPU[b + 4 + 2 * k];
    const u = paramsU32[b + 5 + 2 * k];
    const a = f16(u & 0xFFFF), p = f16(u >>> 16);
    const term = a * Math.cos(f * tt + p);
    if (k < 4) R += term; else if (k < 8) Z += term; else PHI += term;
  }
  return [R * Math.cos(PHI), R * Math.sin(PHI), Z];
}

function showDossier(i) {
  picked = i;
  ribbonValid = false;
  trackOnce("demo_interaction", "picked-star");
  const b = i * 28;
  const Rg = paramsCPU[b], Om = paramsCPU[b + 2];
  let rAmp = 0, zAmp = 0;
  for (let k = 0; k < 4; k++) rAmp += Math.abs(f16(paramsU32[b + 5 + 2 * k] & 0xFFFF));
  for (let k = 4; k < 8; k++) zAmp += Math.abs(f16(paramsU32[b + 5 + 2 * k] & 0xFFFF));
  const bpRp = f16(metaCPU[4 * i]), g = f16(metaCPU[4 * i + 1]);
  const err = f16(metaCPU[4 * i + 2]), good = f16(metaCPU[4 * i + 3]) > 0.5;
  const sp = starPos(i, t), su = sunFn(t);
  const dNow = Math.hypot(sp[0] - su[0], sp[1] - su[1], sp[2] - su[2]);
  const dv = Rg * Om * KMS_PER_KPCMYR - Math.sign(Om) * vcOf(Rg);

  $("dtitle").textContent = "Gaia DR3 " + idsCPU[i].toString();
  $("dg").textContent = g.toFixed(2);
  $("dbprp").textContent = bpRp.toFixed(2);
  $("ddist").textContent = dNow < 1 ? (dNow * 1000).toFixed(0) + " pc"
                                    : dNow.toFixed(2) + " kpc";
  $("drg").textContent = Rg.toFixed(2) + " kpc";
  $("dper").textContent = (2 * Math.PI / Math.abs(Om)).toFixed(0) + " Myr";
  $("dramp").textContent = "±" + (rAmp).toFixed(2) + " kpc";
  $("dzamp").textContent = "±" + (zAmp * 1000).toFixed(0) + " pc";
  $("ddv").textContent = (dv >= 0 ? "+" : "") + dv.toFixed(0) + " km/s";
  $("derr").textContent = err.toFixed(0) + " pc";
  $("dwarn").textContent = good ? "" :
    "Poor fit: this orbit is not well described by a short series " +
    "(likely a hot or near-resonant orbit). Its path is indicative only.";
  $("dlink").href =
    "https://gaia.ari.uni-heidelberg.de/singlesource.html#id=" + idsCPU[i].toString();
  $("dossier").style.display = "block";
}
function closeDossier() { picked = -1; $("dossier").style.display = "none"; }

/* ---------------- frame ---------------- */

function frame(ts) {
  const dtSec = Math.min(0.1, (ts - lastTs) / 1000 || 0);
  lastTs = ts;
  if (playing) {
    t += parseFloat($("speed").value) * dtSec;
    if (t > 500) t = -500;
    $("tslider").value = t;
  }
  $("tval").firstChild.textContent = "t = " + t.toFixed(0) + " Myr";
  $("gyr").textContent = (t / sunPeriod).toFixed(2) + " galactic years";

  const w = Math.max(2, canvas.clientWidth * devicePixelRatio | 0);
  const h = Math.max(2, canvas.clientHeight * devicePixelRatio | 0);
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w; canvas.height = h;
    makeHdr(w, h);
  }

  // camera
  const su = sunFn(t);
  const sunWorld = [su[0], su[2], -su[1]];
  if (comoving) target = sunWorld;
  const eye = [target[0] + dist * Math.cos(el) * Math.cos(az),
               target[1] + dist * Math.sin(el),
               target[2] + dist * Math.cos(el) * Math.sin(az)];
  const la = lookAt(eye, target, [0, 1, 0]);
  camBasis = { ...la, eye };
  curProj = persp(0.9, w / h, 0.1, 600);
  curMvp = mat4mul(curProj, la.m);

  // uniforms
  device.queue.writeBuffer(uPos, 0, new Float32Array([t, 0, 0, 0]));
  device.queue.writeBuffer(uPos, 4, new Uint32Array([alive]));
  const sb = new Float32Array(28);
  sb.set(curMvp, 0);
  sb.set([la.x[0], la.x[1], la.x[2], 0], 16);
  sb.set([la.y[0], la.y[1], la.y[2], 0], 20);
  sb[24] = 0.055;
  device.queue.writeBuffer(uStar, 0, sb);
  device.queue.writeBuffer(uTone, 0,
    new Float32Array([Math.pow(10, parseFloat($("expo").value)) * 0.9, 0, 0, 0]));

  if (picked >= 0 && !ribbonValid) {
    device.queue.writeBuffer(uRibGen, 0,
      new Float32Array([-500, 1000 / (RIBBON_N - 1), 0, 0]));
    device.queue.writeBuffer(uRibGen, 8, new Uint32Array([picked, RIBBON_N]));
    ribbonValid = true;
  }
  const rb = new Float32Array(20);
  rb.set(curMvp, 0); rb[16] = t;
  device.queue.writeBuffer(uRibDraw, 0, rb);

  const mb = new Float32Array(32);
  mb.set(curMvp, 0);
  mb.set([la.x[0], la.x[1], la.x[2], 0], 16);
  mb.set([la.y[0], la.y[1], la.y[2], 0], 20);
  mb.set([...sunWorld, 1], 24);
  if (picked >= 0) {
    const pp = starPos(picked, t);
    mb.set([pp[0], pp[2], -pp[1], 1], 28);
  } else mb.set([0, 0, 0, 0], 28);
  device.queue.writeBuffer(uMark, 0, mb);

  // passes
  const enc = device.createCommandEncoder();
  if (alive > 0) {
    const cp = enc.beginComputePass();
    cp.setPipeline(pipePos);
    cp.setBindGroup(0, bgPos);
    cp.dispatchWorkgroups(Math.ceil(alive / WORKGROUP));
    if (picked >= 0) {
      cp.setPipeline(pipeRibGen);
      cp.setBindGroup(0, bgRibGen);
      cp.dispatchWorkgroups(Math.ceil(RIBBON_N / WORKGROUP));
    }
    cp.end();
  }
  const hp = enc.beginRenderPass({ colorAttachments: [{ view: hdrView,
    loadOp: "clear", storeOp: "store", clearValue: { r: 0, g: 0, b: 0, a: 1 } }] });
  if (alive > 0) {
    hp.setPipeline(pipeStars);
    hp.setBindGroup(0, cmode === 0 ? bgStarsA : bgStarsB);
    hp.draw(4, alive);
    if (picked >= 0) {
      hp.setPipeline(pipeRibDraw);
      hp.setBindGroup(0, bgRibDraw);
      hp.draw(RIBBON_N);
    }
  }
  hp.end();
  const canvasView = ctx.getCurrentTexture().createView();
  const tp = enc.beginRenderPass({ colorAttachments: [{ view: canvasView,
    loadOp: "clear", storeOp: "store", clearValue: { r: 0, g: 0, b: 0, a: 1 } }] });
  tp.setPipeline(pipeTone);
  tp.setBindGroup(0, bgTone);
  tp.draw(3);
  if (markers) {
    tp.setPipeline(pipeMark);
    tp.setBindGroup(0, bgMark);
    tp.draw(4, 3);
  }
  tp.end();
  device.queue.submit([enc.finish()]);

  if (pickPending) runPick();
  requestAnimationFrame(frame);
}

/* ---------------- input ---------------- */

function wireInput() {
  let drag = null, moved = 0;
  canvas.addEventListener("mousedown", e => { drag = [e.clientX, e.clientY]; moved = 0; });
  window.addEventListener("mouseup", e => {
    if (drag && moved < 4) requestPick(e.clientX, e.clientY);
    drag = null;
  });
  window.addEventListener("mousemove", e => {
    if (!drag) return;
    moved += Math.abs(e.clientX - drag[0]) + Math.abs(e.clientY - drag[1]);
    az -= (e.clientX - drag[0]) * 0.005;
    el = Math.max(-1.5, Math.min(1.5, el + (e.clientY - drag[1]) * 0.005));
    drag = [e.clientX, e.clientY];
  });
  canvas.addEventListener("wheel", e => {
    e.preventDefault();
    dist = Math.max(2, Math.min(160, dist * Math.exp(e.deltaY * 0.001)));
  }, { passive: false });
  window.addEventListener("keydown", e => {
    if (e.code === "Space") { e.preventDefault(); $("play").click(); }
    if (!camBasis) return;
    const step = dist * 0.03;
    const f = [-camBasis.z[0], 0, -camBasis.z[2]], r = camBasis.x;
    const fl = Math.hypot(f[0], f[2]) || 1;
    if (comoving && "KeyWKeyAKeySKeyDKeyQKeyE".includes(e.code)) toggleComove();
    if (e.code === "KeyW") { target[0] += f[0] / fl * step; target[2] += f[2] / fl * step; }
    if (e.code === "KeyS") { target[0] -= f[0] / fl * step; target[2] -= f[2] / fl * step; }
    if (e.code === "KeyA") { target[0] -= r[0] * step; target[2] -= r[2] * step; }
    if (e.code === "KeyD") { target[0] += r[0] * step; target[2] += r[2] * step; }
    if (e.code === "KeyQ") target[1] -= step;
    if (e.code === "KeyE") target[1] += step;
  });

  $("play").onclick = () => {
    playing = !playing;
    $("play").textContent = playing ? "Pause" : "Play";
  };
  $("tslider").oninput = () => {
    t = parseFloat($("tslider").value);
    trackOnce("demo_interaction", "scrubbed-timeline");
  };
  $("cmode").onchange = () => { cmode = parseInt($("cmode").value, 10); };
  $("comove").onclick = toggleComove;
  $("marks").onclick = () => {
    markers = !markers;
    $("marks").textContent = markers ? "on" : "off";
    $("marks").classList.toggle("on", markers);
  };
  $("dclose").onclick = closeDossier;

  const showAbout = v => {
    $("about").style.display = v ? "block" : "none";
    $("backdrop").style.display = v ? "block" : "none";
  };
  $("about-open").onclick = () => showAbout(true);
  $("about-close").onclick = () => showAbout(false);
  $("backdrop").onclick = () => showAbout(false);
  window.addEventListener("keydown", e => {
    if (e.code === "Escape") showAbout(false);
  });
}
function toggleComove() {
  comoving = !comoving;
  $("comove").textContent = comoving ? "on" : "off";
  $("comove").classList.toggle("on", comoving);
  if (!comoving) target = [...target];
  if (comoving) dist = Math.min(dist, 25);
}

/* ---------------- boot ---------------- */

async function main() {
  canvas = $("gpu");
  if (!navigator.gpu) {
    $("err").style.display = "grid";
    trackOnce("webgpu_unsupported", "no-webgpu");
    return;
  }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    $("err").style.display = "grid";
    $("errmsg").textContent = "No GPU adapter found.";
    trackOnce("webgpu_unsupported", "no-adapter");
    return;
  }
  device = await adapter.requestDevice();
  ctx = canvas.getContext("webgpu");
  fmt = navigator.gpu.getPreferredCanvasFormat();
  ctx.configure({ device, format: fmt, alphaMode: "opaque" });
  await loadManifest();
  wireInput();
  requestAnimationFrame(frame);            // render while shards stream in
  streamShards().catch(e => {
    $("load").textContent = "shard loading failed: " + e.message;
  });
}
main();
