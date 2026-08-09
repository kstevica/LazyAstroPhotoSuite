"""LazyMoonSun stacking engines — global burst stack + multi-point lucky imaging.

Faithful port of ``SUN.stackBurst`` (LazyMoonSun.js:655-957) and ``SUN.stackMultiPoint``
(:1916-2317). Both consume frames through a ``load(frame) -> float array`` callable so a
whole burst never sits in RAM at once (the .js opens/closes each window per pass). All the
alignment is the FFT phase-correlation math in :mod:`.register`; there are no PI processes.

``frames`` is any list of opaque identifiers (paths, indices) understood by ``load``.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Sequence

import numpy as np

from . import register as reg


def _noop(_msg: str) -> None:
    pass


def _p999(a: np.ndarray) -> float:
    return float(np.quantile(a, 0.999))


def _same_shape(a: np.ndarray, h: int, w: int) -> bool:
    return a.shape[0] == h and a.shape[1] == w


# --------------------------------------------------------------------- global stack


def grade_frames(frames: Sequence, load: Callable, *, workdim: int = reg.WORKDIM,
                 log: Callable[[str], None] = _noop) -> List[dict]:
    """Grade sharpness + brightness for every readable frame (SUN.stackBurst [1/3])."""
    graded = []
    for i, f in enumerate(frames):
        try:
            img = load(f)
        except Exception:
            img = None
        if img is None:
            log(f"  frame {i} unreadable, skipped")
            continue
        wk, k = reg.working(img, workdim)
        g = reg.gradient(wk)
        graded.append({"f": f, "sharp": float(g.std()), "k": k, "p999": _p999(wk)})
    return graded


def _junk_gate(graded: List[dict], log: Callable[[str], None]) -> List[dict]:
    """Drop black/obstructed frames (SUN.stackBurst junk-frame gate): relative to the burst."""
    if not graded:
        return graded
    p9s = sorted(g["p999"] for g in graded)
    med = p9s[len(p9s) >> 1]
    lit = []
    for g in graded:
        if g["p999"] < 0.15 * med or (g["p999"] < 0.02 and g["p999"] < 0.5 * med):
            log(f"  dropped: no disc (p99.9 {g['p999']:.4f} vs burst median {med:.4f})")
        else:
            lit.append(g)
    return lit


def _register_pass(kept: List[dict], load: Callable, *, workdim: int, refdim: int,
                   max_drift: float, log: Callable[[str], None]) -> Optional[dict]:
    """Two-stage register + mean-accumulate ``kept`` against kept[0] (SUN.registerPass)."""
    ref_img = load(kept[0]["f"])
    if ref_img is None:
        return None
    ref_img = np.asarray(ref_img, dtype=np.float64)
    full_h, full_w = ref_img.shape[0], ref_img.shape[1]
    ref_wk, ref_k = reg.working(ref_img, workdim)
    ref = reg.make_ref(reg.gradient(ref_wk), ref_wk.shape[1], ref_wk.shape[0])

    # full-res refine reference: a REFDIM crop centred on the disc, prepared once
    refine = None
    ref_rect = None
    ctr = reg.disc_center(ref_wk)
    if ctr is not None:
        REF = int(min(refdim, full_w, full_h))
        fcx, fcy = ctr[0] / ref_k, ctr[1] / ref_k
        rx0 = int(round(min(max(0, fcx - REF / 2), full_w - REF)))
        ry0 = int(round(min(max(0, fcy - REF / 2), full_h - REF)))
        ref_rect = (rx0, ry0, REF)
        rc = reg.intensity(ref_img)[ry0:ry0 + REF, rx0:rx0 + REF]
        refine = reg.make_ref(reg.gradient(rc), REF, REF)
        log(f"Two-stage registration: refine window {REF}px at full res, disc-centred.")
    else:
        log("Refine window: no disc centre — coarse registration only.")

    acc = np.array(ref_img, dtype=np.float64, copy=True)
    n_acc = 1
    excluded: List[dict] = []

    for i in range(1, len(kept)):
        f = kept[i]
        img = load(f["f"])
        if img is None:
            continue
        img = np.asarray(img, dtype=np.float64)
        if not _same_shape(img, full_h, full_w):
            log(f"  frame differs in size, skipped")
            continue
        wk, k = reg.working(img, workdim)
        dx, dy = reg.measure_against(ref, reg.gradient(wk), wk.shape[1], wk.shape[0])
        drift = np.hypot(dx, dy)
        if drift > max_drift:
            log(f"  drift {drift:.1f}px exceeds {max_drift} — excluded outlier")
            excluded.append({"f": f, "dx": dx, "dy": dy})
            continue

        tdx, tdy = dx / k, dy / k                     # coarse full-res offset
        if refine is not None:
            cix, ciy = int(round(tdx)), int(round(tdy))
            rx0, ry0, REF = ref_rect
            tx0, ty0 = rx0 + cix, ry0 + ciy
            if tx0 >= 0 and ty0 >= 0 and tx0 + REF <= full_w and ty0 + REF <= full_h:
                tc = reg.intensity(img)[ty0:ty0 + REF, tx0:tx0 + REF]
                rdx, rdy = reg.measure_against(refine, reg.gradient(tc), REF, REF)
                if np.hypot(rdx, rdy) < 64:           # within the refine capture range
                    tdx, tdy = cix + rdx, ciy + rdy

        acc += reg.apply_shift(img, -tdx, -tdy)
        n_acc += 1
        log(f"  stacked offset ({-tdx:+.2f}, {-tdy:+.2f})")

    return {"acc": acc, "n_acc": n_acc, "excluded": excluded}


def stack_burst(frames: Sequence, load: Callable, *, keep: float = 0.50,
                workdim: int = reg.WORKDIM, refdim: int = reg.REFDIM,
                max_drift: float = 220.0, log: Callable[[str], None] = _noop) -> Optional[dict]:
    """Global engine: grade → junk-gate → cull → two-stage register → mean stack."""
    if len(frames) < 2:
        log("A burst needs at least 2 frames.")
        return None

    log(f"[1/3] Grading sharpness ({len(frames)} frames)...")
    graded = _junk_gate(grade_frames(frames, load, workdim=workdim, log=log), log)
    if len(graded) < 2:
        log("Fewer than 2 readable frames — nothing to stack.")
        return None

    graded.sort(key=lambda g: g["sharp"], reverse=True)
    n_keep = max(2, round(len(graded) * keep))
    kept = graded[:n_keep]
    log(f"[2/3] Keeping best {n_keep} of {len(graded)}. Reference chosen by sharpness.")

    passres = _register_pass(kept, load, workdim=workdim, refdim=refdim,
                             max_drift=max_drift, log=log)
    if passres is None:
        return None

    # consensus check: if MOST frames were drift-excluded and agree, the reference was junk
    exc = passres["excluded"]
    if len(exc) >= 2 and len(exc) > passres["n_acc"] - 1:
        mdx = sorted(e["dx"] for e in exc)[len(exc) >> 1]
        mdy = sorted(e["dy"] for e in exc)[len(exc) >> 1]
        cluster = [e["f"] for e in exc
                   if np.hypot(e["dx"] - mdx, e["dy"] - mdy) < 60]
        if len(cluster) >= 0.6 * len(exc):
            old_ref, new_ref = kept[0], cluster[0]
            log("advisor: reference was the outlier — re-registering to the consensus cluster.")
            kept2 = [new_ref] + [g for g in kept if g is not new_ref and g is not old_ref]
            passres = _register_pass(kept2, load, workdim=workdim, refdim=refdim,
                                     max_drift=max_drift, log=log)
            if passres is None:
                return None

    n_acc = passres["n_acc"]
    master = passres["acc"] / max(1, n_acc)
    log(f"[3/3] Integrated {n_acc} frame(s) (mean).")
    if n_acc < 2:
        log("WARNING: only the reference survived — this master is a single frame.")
    return {"master": np.clip(master, 0.0, 1.0), "n_graded": len(graded),
            "n_kept": n_keep, "n_stacked": n_acc}


# ------------------------------------------------------------------ multi-point


def _channel(img: np.ndarray, c: int) -> np.ndarray:
    return img[..., c] if img.ndim == 3 else img


def _n_channels(img: np.ndarray) -> int:
    return min(3, img.shape[2]) if img.ndim == 3 else 1


def stack_multipoint(frames: Sequence, load: Callable, master: np.ndarray,
                     params, *, workdim: int = reg.WORKDIM,
                     log: Callable[[str], None] = _noop) -> Optional[dict]:
    """Multi-point engine: per-AP lucky imaging crossfaded onto the global master."""
    AP = max(32, round(params.ap_size))
    STEP = max(16, round(AP / 2))
    SEARCH = max(2, round(params.ap_search))
    MAX_APS = max(20, round(params.ap_max))
    MIN_LIT, PAD = 0.40, 4

    master = np.asarray(master, dtype=np.float64)
    full_h, full_w = master.shape[0], master.shape[1]
    master_l = reg.intensity(master)
    k = min(1.0, workdim / max(full_w, full_h))
    master_wk = reg.zoom(master_l, k, order=1) if k < 1 else master_l
    mp999 = _p999(master_wk)
    lit_thr = 0.5 * mp999
    wk_fft = reg.make_ref(reg.grad_prep(master_wk, 24)["g"], master_wk.shape[1], master_wk.shape[0])

    # ---- AP grid on the lit surface, structure-gated ----
    log("[1/4] Alignment points...")
    candidates = []
    for y in range(0, full_h - AP + 1, STEP):
        for x in range(0, full_w - AP + 1, STEP):
            patch = master_l[y:y + AP, x:x + AP]
            if float((patch[::2, ::2] > lit_thr).mean()) < MIN_LIT:
                continue
            gp = reg.grad_prep(patch, 12)
            candidates.append({"rect": (x, y), "q": gp["q"], "ex": gp["ex"], "ey": gp["ey"]})
    if not candidates:
        log("No lit AP candidates — the global master stands.")
        return None
    q_med = sorted(c["q"] for c in candidates)[len(candidates) >> 1]
    aps, rej_1d, rej_flat = [], 0, 0
    for c in candidates:
        axis_ratio = min(c["ex"], c["ey"]) / max(1e-12, max(c["ex"], c["ey"]))
        if axis_ratio < 0.15:
            rej_1d += 1
            continue
        if c["q"] < 0.30 * q_med:
            rej_flat += 1
            continue
        aps.append(c)
    aps.sort(key=lambda a: a["q"], reverse=True)
    aps = aps[:MAX_APS]
    log(f"APs: {len(aps)} kept of {len(candidates)} lit "
        f"({rej_1d} 1D-edges, {rej_flat} featureless).")
    if len(aps) < 4:
        log("Too few trackable APs — the global master stands.")
        return None
    for a in aps:
        x, y = a["rect"]
        gp = reg.grad_prep(master_l[y:y + AP, x:x + AP], 12)
        a["fft"] = reg.make_ref(gp["g"], AP, AP)
        a["meas"] = []

    # ---- measure pass: global coarse + per-AP offsets ----
    log("[2/4] Measuring per-AP offsets...")
    frame_info: List[Optional[dict]] = []
    spread_sum, spread_max, spread_n = 0.0, 0.0, 0
    for f in frames:
        img = load(f)
        if img is None or not _same_shape(np.asarray(img), full_h, full_w):
            frame_info.append(None)
            continue
        img = np.asarray(img, dtype=np.float64)
        L = reg.intensity(img)
        wk = reg.zoom(L, k, order=1) if k < 1 else L
        if _p999(wk) < 0.15 * mp999:
            frame_info.append(None)
            continue
        gdx, gdy = reg.measure_against(wk_fft, reg.grad_prep(wk, 24)["g"], wk.shape[1], wk.shape[0])
        ix, iy = int(round(gdx / k)), int(round(gdy / k))
        dxs, dys = [], []
        for a in aps:
            x0, y0 = a["rect"]
            rx, ry = x0 + ix, y0 + iy
            if rx < 0 or ry < 0 or rx + AP > full_w or ry + AP > full_h:
                continue
            gp = reg.grad_prep(L[ry:ry + AP, rx:rx + AP], 12)
            mdx, mdy = reg.measure_against(a["fft"], gp["g"], AP, AP)
            if np.hypot(mdx, mdy) <= SEARCH:
                a["meas"].append({"f": f, "ox": ix + mdx, "oy": iy + mdy, "q": gp["q"]})
                dxs.append(mdx)
                dys.append(mdy)
        if len(dxs) > 3:
            dxs_a, dys_a = np.array(dxs), np.array(dys)
            spread = float(np.sqrt(((dxs_a - dxs_a.mean()) ** 2 + (dys_a - dys_a.mean()) ** 2).mean()))
            spread_sum += spread
            spread_max = max(spread_max, spread)
            spread_n += 1
        frame_info.append({"ix": ix, "iy": iy})

    # ---- per-AP selection (the Local estimator) ----
    kept_total = 0
    for a in aps:
        a["meas"].sort(key=lambda m: m["q"], reverse=True)
        nk = max(3, round(len(a["meas"]) * params.ap_keep))
        a["kept"] = a["meas"][:min(nk, len(a["meas"]))]
        kept_total += len(a["kept"])
    mean_kept = kept_total / len(aps)
    needed = {id(m["f"]): m["f"] for a in aps for m in a["kept"]}
    log(f"[3/4] Local selection: mean {mean_kept:.1f} frames per AP.")

    # ---- stack pass (raised-cosine crossfade) ----
    log("[4/4] Stacking patches...")
    yy = 0.5 * (1 - np.cos(2 * np.pi * (np.arange(AP) + 0.5) / AP))
    win = np.outer(yy, yy)
    nch = _n_channels(master)
    acc = [np.zeros((full_h, full_w)) for _ in range(nch)]
    wgt = np.zeros((full_h, full_w))
    stacked = 0
    for f in frames:
        if id(f) not in needed:
            continue
        img = load(f)
        if img is None:
            continue
        img = np.asarray(img, dtype=np.float64)
        for a in aps:
            sel = next((m for m in a["kept"] if m["f"] is f), None)
            if sel is None:
                continue
            x0, y0 = a["rect"]
            ox, oy = int(round(sel["ox"])), int(round(sel["oy"]))
            fx, fy = sel["ox"] - ox, sel["oy"] - oy
            sx, sy = x0 + ox - PAD, y0 + oy - PAD
            if sx < 0 or sy < 0 or sx + AP + 2 * PAD > full_w or sy + AP + 2 * PAD > full_h:
                continue
            for c in range(nch):
                patch = _channel(img, c)[sy:sy + AP + 2 * PAD, sx:sx + AP + 2 * PAD]
                patch = reg.apply_shift(patch, -fx, -fy)
                core = patch[PAD:PAD + AP, PAD:PAD + AP] * win
                acc[c][y0:y0 + AP, x0:x0 + AP] += core
            wgt[y0:y0 + AP, x0:x0 + AP] += win
        stacked += 1

    # ---- compose: acc/wgt where covered, global master elsewhere ----
    out = np.array(master, dtype=np.float64, copy=True)
    covered = wgt > 0.005
    safe_wgt = np.where(covered, wgt, 1.0)
    if out.ndim == 3:
        for c in range(nch):
            blended = acc[c] / safe_wgt
            out[..., c] = np.where(covered, blended, out[..., c])
    else:
        out = np.where(covered, acc[0] / safe_wgt, out)

    aniso_mean = (spread_sum / spread_n) if spread_n else 0.0
    report = _mp_report(AP, STEP, params, SEARCH, MAX_APS, len(aps), len(candidates),
                        rej_1d, rej_flat, mean_kept, stacked, aniso_mean, spread_max, spread_n)
    log(f"Multi-point done: {len(aps)} APs, {mean_kept:.1f} frames/AP, {stacked} stacked.")
    if spread_n:
        log(f"Anisoplanatism: mean {aniso_mean:.2f}px, max {spread_max:.2f}px over {spread_n} frames.")
    return {"master": np.clip(out, 0.0, 1.0), "n_aps": len(aps), "mean_kept": mean_kept,
            "n_stacked": stacked, "aniso_mean": aniso_mean, "aniso_max": spread_max,
            "report": report}


def _mp_report(AP, STEP, params, SEARCH, MAX_APS, n_aps, n_cand, rej_1d, rej_flat,
               mean_kept, stacked, aniso_mean, aniso_max, spread_n) -> str:
    """The mp_report.txt block (SUN.stackMultiPoint report append)."""
    lines = [
        f"dials: AP size {AP}, step {STEP}, keep/AP {params.ap_keep:.2f}, "
        f"search {SEARCH} px, cap {MAX_APS}",
        f"APs: {n_aps} kept of {n_cand} lit candidates ({rej_1d} 1D-edge, {rej_flat} featureless)",
        f"selection: mean {mean_kept:.1f} frames per AP; {stacked} frames stacked",
        (f"anisoplanatism: mean {aniso_mean:.2f} px, max {aniso_max:.2f} px over {spread_n} frames"
         if spread_n else "anisoplanatism: not measured"),
    ]
    return "\n".join(lines)
