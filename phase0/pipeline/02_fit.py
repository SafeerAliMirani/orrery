"""Phase 0: fit a truncated quasi-periodic series to each integrated orbit."""

import argparse
import numpy as np
import h5py

K_R, K_Z, K_P = 4, 4, 4
CHUNK = 250


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
        # mask bins around lines already taken so deflation leftovers
        # cannot be picked twice (duplicate columns break the solve)
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
    """Freqs fixed; batched normal-equation refit of const (+lin) and lines."""
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
    ap.add_argument("--shard", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", default=None)
    args = ap.parse_args()

    with h5py.File(args.shard, "r") as f:
        if args.rows:
            a, b = (int(v) for v in args.rows.split(":"))
            pos = f["pos"][a:b]
            sid = f["source_id"][a:b]
        else:
            pos = f["pos"][:]
            sid = f["source_id"][:]
        t = f["t_myr"][:].astype(np.float64)

    n = pos.shape[0]
    keys = ["Rg", "fR", "aR", "pR", "zc", "fZ", "aZ", "pZ",
            "ph0", "Om", "fP", "aP", "pP",
            "err_med", "err_p95", "err500", "errmax"]
    out = {k: [] for k in keys}
    i5a = np.argmin(np.abs(t + 500))
    i5b = np.argmin(np.abs(t - 500))

    for c0 in range(0, n, CHUNK):
        P = pos[c0:min(c0 + CHUNK, n)].astype(np.float64)
        x, y, z = P[:, :, 0], P[:, :, 1], P[:, :, 2]
        R = np.hypot(x, y)
        phi = np.unwrap(np.arctan2(y, x), axis=1)

        Rg, _, fR, aR, pR, _ = fit_coord(R, t, K_R)
        zc, _, fZ, aZ, pZ, _ = fit_coord(z, t, K_Z)
        ph0, Om, fP, aP, pP, _ = fit_coord(phi, t, K_P, with_t=True)

        rec = reconstruct(t, Rg, fR, aR, pR, zc, fZ, aZ, pZ, ph0, Om, fP, aP, pP)
        err = np.linalg.norm(rec - P, axis=2) * 1000.0
        out["err_med"].append(np.median(err, axis=1))
        out["err_p95"].append(np.percentile(err, 95, axis=1))
        out["err500"].append(np.maximum(err[:, i5a], err[:, i5b]))
        out["errmax"].append(err.max(axis=1))
        for k, v in [("Rg", Rg), ("fR", fR), ("aR", aR), ("pR", pR),
                     ("zc", zc), ("fZ", fZ), ("aZ", aZ), ("pZ", pZ),
                     ("ph0", ph0), ("Om", Om), ("fP", fP), ("aP", aP), ("pP", pP)]:
            out[k].append(v)

    with h5py.File(args.out, "w") as f:
        for k in keys:
            f.create_dataset(k, data=np.concatenate(out[k], axis=0))
        f.create_dataset("source_id", data=sid)

    e5 = np.concatenate(out["err500"])
    em = np.concatenate(out["err_med"])
    print(f"fit {args.out}: {n} stars | err@500Myr median {np.median(e5):.1f} pc, "
          f"p95 {np.percentile(e5, 95):.0f} pc, frac>200pc "
          f"{(e5 > 200).mean():.3f} | window-median median {np.median(em):.1f} pc")


if __name__ == "__main__":
    main()
