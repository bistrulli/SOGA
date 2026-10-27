"""CPU emulation of an NVBitFI transient campaign on PolyBench-GPU GEMM.

Mirrors the four NVBitFI steps:
  1. profile  -> dynamic G_GP sites per thread (ASSUMED mix, gemm_model.thread_sites)
  2. select   -> uniform dynamic instruction (thread, site) + uniform bit 0..31
  3. inject   -> XOR the bit into the destination register, replay the thread
                 with exact float32 semantics (FMUL / FFMA single rounding)
  4. classify -> gemm.cu check (percentDiff > 0.05 % on any element => SDC;
                 NaN passes), DUE for misaligned / out-of-bounds accesses.
Option B: every injection also logs the relative error of the corrupted
element, so MSK(eps) is available for any eps from the same campaign.

Structural sites (index / address / loop counter) use simple ASSUMED rules,
stated per class below; they are illustrative, SOGA does not model them.
"""
from __future__ import annotations

import numpy as np

from gemm_model import (ALL_CLASSES, ALPHA, ARITH_CLASSES, BETA, CLASS_ID, EPS_POLYBENCH,
                        Site, fma32, flip_f32, is_sdc, mul32, percent_diff,
                        thread_sites)

_C = CLASS_ID


def _accumulate(A_flat: np.ndarray, B_flat: np.ndarray, c0: np.ndarray,
                a_idx: np.ndarray, b_idx: np.ndarray, active: np.ndarray,
                cls: np.ndarray, kf: np.ndarray, bit: np.ndarray) -> np.ndarray:
    """Replay N threads in lockstep; inject the arithmetic flips at (cls, kf, bit)."""
    N, n = a_idx.shape
    rows = np.arange(N)
    a = A_flat[a_idx]
    b = B_flat[b_idx]
    c0 = c0.copy()

    m = cls == _C["LDG.a"]
    a[rows[m], kf[m]] = flip_f32(a[rows[m], kf[m]], bit[m])
    m = cls == _C["LDG.b"]
    b[rows[m], kf[m]] = flip_f32(b[rows[m], kf[m]], bit[m])
    m = cls == _C["LDG.c"]
    c0[m] = flip_f32(c0[m], bit[m])

    acc = mul32(BETA, c0)
    m = cls == _C["FMUL.beta"]
    acc[m] = flip_f32(acc[m], bit[m])

    t = mul32(ALPHA, a)
    m = cls == _C["FMUL.alpha"]
    t[rows[m], kf[m]] = flip_f32(t[rows[m], kf[m]], bit[m])

    ffma_k = np.where(cls == _C["FFMA"], kf, -1)
    for kk in range(n):
        acc = np.where(active[:, kk], fma32(t[:, kk], b[:, kk], acc), acc)
        m = ffma_k == kk
        if m.any():
            acc[m] = flip_f32(acc[m], bit[m])
    return acc


def _structural(site: Site, b: int, i: int, j: int, n: int,
                a_idx: np.ndarray, b_idx: np.ndarray, active: np.ndarray
                ) -> str | None:
    """Apply a structural fault in place. Returns a forced outcome or None (=replay).

    ASSUMED rules: buffers are 2^32-aligned (address bit b flips element-offset bit b-2);
    b < 2 => misaligned => DUE; any read/write outside its own buffer => DUE (illegal
    address); index faults => the own cell is never (correctly) written => SDC."""
    nn = n * n
    c = site.cls
    if c == "S2R/IMAD.idx":
        return "SDC"
    if c in ("IADD.ptr_a.hi", "IADD.ptr_b.hi"):
        return "MSK" if site.k == n else "DUE"           # never read again after the last block
    if c == "IADD.k":
        cnt = site.k
        new = np.int64(np.int32(np.uint32(cnt) ^ np.uint32(1 << b)))  # signed 32-bit counter
        remaining = max(0, n - int(new))
        if cnt + remaining > n:                           # extra iterations walk b past its buffer
            return "DUE"
        active[cnt + remaining:] = False                  # early exit: tail terms skipped
        return "MSK" if cnt == n else None
    if b < 2:
        return "DUE"                                      # misaligned address
    flip = 1 << (b - 2)
    if c == "IMAD.addr_c":
        new = (i * n + j) ^ flip
        return "DUE" if new >= nn else "SDC"              # writes another cell, own cell stale
    if c == "IMAD.addr_a":
        base = (i * n) ^ flip
        a_idx[:] = base + np.arange(n)
    elif c == "IMAD.addr_b":
        base = j ^ flip
        b_idx[:] = base + np.arange(n) * n
    elif c == "IADD.ptr_a.lo":
        if site.k == n:
            return "MSK"
        cur = (i * n + site.k) ^ flip
        a_idx[site.k:] = cur + np.arange(n - site.k)
    elif c == "IADD.ptr_b.lo":
        if site.k == n:
            return "MSK"
        cur = (site.k * n + j) ^ flip
        b_idx[site.k:] = cur + np.arange(n - site.k) * n
    else:
        raise ValueError(f"unknown structural class {c}")
    if a_idx.min() < 0 or a_idx.max() >= nn or b_idx.min() < 0 or b_idx.max() >= nn:
        return "DUE"
    return None


def run_campaign(A: np.ndarray, B: np.ndarray, C0: np.ndarray, n_inj: int,
                 unroll: int, seed: int, batch: int = 4000) -> dict[str, np.ndarray]:
    """Returns per-injection arrays: cls, k, bit, outcome@0.05% ('MSK'/'SDC'/'DUE'), pd (percent)."""
    n = A.shape[0]
    rng = np.random.default_rng(seed)
    sites = thread_sites(n, unroll)
    A_flat, B_flat, C0_flat = A.ravel(), B.ravel(), C0.ravel()

    cell_i = rng.integers(0, n, n_inj)
    cell_j = rng.integers(0, n, n_inj)
    site_ix = rng.integers(0, len(sites), n_inj)
    bits = rng.integers(0, 32, n_inj)

    out_cls = np.empty(n_inj, dtype=object)
    out_k = np.empty(n_inj, dtype=np.int64)
    outcome = np.empty(n_inj, dtype=object)
    pd_all = np.full(n_inj, np.nan)
    replayed_all = np.zeros(n_inj, dtype=bool)

    for s0 in range(0, n_inj, batch):
        sl = slice(s0, min(n_inj, s0 + batch))
        N = sl.stop - sl.start
        ii, jj, bb = cell_i[sl], cell_j[sl], bits[sl]
        a_idx = (ii[:, None] * n + np.arange(n)[None, :]).astype(np.int64)
        b_idx = (np.arange(n)[None, :] * n + jj[:, None]).astype(np.int64)
        active = np.ones((N, n), dtype=bool)
        cls = np.empty(N, dtype=np.int64)
        kf = np.zeros(N, dtype=np.int64)
        forced = np.empty(N, dtype=object)
        for r in range(N):
            site = sites[site_ix[sl][r]]
            cls[r] = CLASS_ID[site.cls]
            kf[r] = max(site.k, 0)
            out_cls[sl.start + r] = site.cls
            out_k[sl.start + r] = site.k
            forced[r] = None
            if site.cls not in ARITH_CLASSES:
                forced[r] = _structural(site, int(bb[r]), int(ii[r]), int(jj[r]), n,
                                        a_idx[r], b_idx[r], active[r])
        c0 = C0_flat[ii * n + jj]
        # golden: same threads, default addresses, no fault
        g_a = (ii[:, None] * n + np.arange(n)[None, :]).astype(np.int64)
        g_b = (np.arange(n)[None, :] * n + jj[:, None]).astype(np.int64)
        no = np.full(N, -1)
        golden = _accumulate(A_flat, B_flat, c0, g_a, g_b, np.ones((N, n), bool), no, kf, bb)
        replay = np.array([f is None for f in forced])
        safe_a = np.where(replay[:, None], a_idx, g_a)
        safe_b = np.where(replay[:, None], b_idx, g_b)
        arith_cls = np.where(replay, cls, -1)
        faulty = _accumulate(A_flat, B_flat, c0, safe_a, safe_b, active, arith_cls, kf, bb)
        pd = percent_diff(golden, faulty)
        for r in range(N):
            if forced[r] is not None:
                outcome[sl.start + r] = forced[r]
            else:
                pd_all[sl.start + r] = pd[r]
                replayed_all[sl.start + r] = True
                outcome[sl.start + r] = "SDC" if is_sdc(pd[r], EPS_POLYBENCH) else "MSK"
    return {"cls": out_cls, "k": out_k, "bit": bits, "outcome": outcome, "pd": pd_all,
            "replayed": replayed_all, "i": cell_i, "j": cell_j}


def rates_by_class(res: dict[str, np.ndarray], eps: float = EPS_POLYBENCH
                   ) -> dict[str, dict[str, float]]:
    """MSK/SDC/DUE fractions per class and overall; replayed sites re-classified at eps (option B)."""
    out: dict[str, dict[str, float]] = {}
    resd = res["outcome"].copy()
    rep = res["replayed"]
    resd[rep] = np.where(is_sdc(res["pd"][rep], eps), "SDC", "MSK")
    for name, m in [("ALL", np.ones(len(resd), bool)),
                    ("ARITH", np.isin(res["cls"], list(ARITH_CLASSES)))] + \
                   [(c, res["cls"] == c) for c in ALL_CLASSES]:
        tot = int(m.sum())
        if tot == 0:
            continue
        out[name] = {o: float((resd[m] == o).sum()) / tot for o in ("MSK", "SDC", "DUE")}
        out[name]["n"] = tot
    return out

