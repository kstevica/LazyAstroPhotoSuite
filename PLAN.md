# LazyStretch → Python Port — Architecture & Porting PLAN

**Source of truth:** `reference/pixinsight/LazyStretch.js` — PixInsight LazyStretch **v1.4.1**, 4847 lines, SHA-1 `c997b739bc6406fe055fe793bede45b56c003273` (recorded in `reference/pixinsight/SOURCE.md`, synced 2026-08-04). The **implemented Python pipeline tracks v1.2.1 parity + the colour-cal fix**; the v1.2.1→v1.4.1 feature delta is the catch-up backlog in **§12**.
**Target:** `LazyStretchPy` — a standalone, native-desktop Python application. No web version, no web views. Ships as a pip package (primary) and a signed macOS `.app` (secondary), from one source tree.
**This document is a plan, not a spec dump.** It is grounded entirely in the eight structured source analyses. Where a step cannot be ported (the "AI wall"), it says so plainly and names the open substitute plus the honest quality gap.

---

## 1. Executive summary

LazyStretch is a statistics-driven, profile-driven, non-destructive stretching pipeline. It takes a **linear** astro master, classifies the object, and runs an ordered list of ~21 steps that ends in a permanent non-linear "look" tuned per object class and nudged by five sliders. A large share of the code — the MTF stretch math, class profiles, dust self-calibration, auto-assessment, masks, and the **PixelMath-expression** finishing steps — is pure numerical/data logic with **zero PixInsight dependency** and ports to numpy verbatim; that portion is the crown jewel. A **distinct** group of finishing steps are **stock PI processes with undocumented internal colour spaces** (ColorSaturation, CurvesTransformation, LocalHistogramEqualization, HDRMultiscaleTransform, MultiscaleMedianTransform) — these must be *reimplemented and visually calibrated against PI*, not diffed bit-for-bit (see §11). So "ports cleanly" ≠ "the whole pipeline"; it means the deterministic brain plus the true PixelMath steps.

**What ports cleanly (deterministic, golden-testable):**
- The entire numerical core: `Math.mtf`, `findMidtonesBalance`, `applyAutoStretch` (soft-knee STF), `measureDust`, `adaptiveFloorRaise`, `autoAssess`.
- All data tables: the 7-class profile table, slider→effective-value clamps, recipe ranges, class predicates, Messier/NGC/IC catalogs, palettes.
- Object identification *downstream of the solve*: catalog search, prominence ranking, `objectClass` classification, data-type detection.
- PixelMath-**expression** finishing (bit-exact, golden-testable): SCNR, background-level pull, dehaze, reduceCast, gradientCleanup blend, crop, narrowband combine, and every range/star-mask composite.
- Stock-**process** finishing (reimplement + visually calibrate, **not** bit-exact): contrast S-curve (Curves), saturation (ColorSaturation), emissionBoost (Sat+Curves), local contrast (LHE), HDR core (HDRMT), the MMT noise fallback.
- Recipes (`.lsrecipe` JSON) and per-object memory — already language-neutral; the port reads/writes byte-compatible files.

**What is a "wall" (proprietary AI / external DB — cannot be ported with parity):**
| PI process | Substitute | Honest parity |
|---|---|---|
| **BlurXTerminator** (deconv/sharpen) | GraXpert deconvolution; classical Richardson–Lucy | **~60–75%** — the weakest parity point in the whole pipeline. No true equivalent exists in 2026. |
| **SpectrophotometricColorCalibration** (Gaia photometry) | "SPCC-lite": astrometry.net solve + Gaia DR3 XP synthetic photometry (GaiaXPy/rgbloom) | Close for broadband **if** sensor QE + filter curves are supplied; otherwise a good-not-identical G2V/blackbody white reference. Hardest piece to build and to keep in sync. |
| **NoiseXTerminator** | GraXpert denoise; **or the script's own MultiscaleMedianTransform fallback** (median multiscale — *not* à-trous; ports at ~100% of PI's *degraded* mode) | **~80–90%** via GraXpert; **100%** of PI's fallback via the MMT path. |
| **StarXTerminator** (star split) | StarNet2 / Cosmic Clarity | **~75–90%** — the LazyStretch star *logic* (run on clone → `stars = orig − starless` → recombine with small-star gamma) is tool-agnostic and ports 100%; only the starless quality differs. |

> **GradientCorrection is NOT a wall.** It is a **stock PixInsight process** (feature-detected via `haveGradientCorrection` only because *older* PI builds lack it, `:3069/:3329-3349`) — not a plugin, not AI, no external DB. It **runs in Preview**. It belongs in the MEDIUM/portable tier (§3 step 5): approximate with GraXpert background-extraction, or fall back to the degree-4 ABE polynomial (`:2303-2323`) — the same fallback PI itself uses, which is bit-portable.

**Key scoping insight — Preview mode already *is* the portable AI-free subset.** In PI, Preview forces `doBXT = doNR = doStarReduce = starProtect = false` and never runs SPCC (all require `!preview`). Everything else — crop, ABE/GC, stretch, SCNR, HDR, background level, contrast, global saturation, cast reduction, emission boost, dehaze, gradient cleanup, local contrast — **runs in Preview**. That set is exactly the deterministic, no-plugin pipeline. So **v1's target is "the PI Preview pipeline, at full resolution, headless and in a GUI"** — a complete, faithful product on its own, with the four walled steps added later as optional external shell-outs that degrade gracefully when absent (mirroring PI's `haveBlurX`/`haveStarX`/… guards).

**Recommended v1 scope:** headless numpy core (validated against PI telemetry) → full AAA-free pipeline (the Preview subset) → PySide6 GUI → optional external-tool integration (ASTAP, StarNet2, GraXpert) → packaging. Ship pip + a notarized macOS `.app`.

---

## 2. Target architecture

The `.js` interleaves data, math, PI-process calls, and UI in one file. The port splits **along the change-frequency seam, not the feature seam**, so a typical PI edit (a number tweak) touches only a JSON data file, and a structural edit touches exactly one clearly-named module. The headless "brain" is pure numpy/scipy, importable and testable with no GUI and no PixInsight.

```
lazystretch/
  data/
    lazystretch_data.json     # THE shared data layer — all volatile numbers (§9)
    Messier.csv  NGC-IC.csv   # bundled DSO catalogs (converted once from PI's AdP copies)
    loader.py                 # loads + validates JSON; exposes typed ClassProfile / tiers / predicates
  stats/
    mtf.py                    # Math.mtf + findMidtonesBalance — the pinned core primitives
    measure.py                # median / avgDev(MAD-about-median) / mean; measureDust; autoAssess scoring
  stretch/
    autostretch.py            # applyAutoStretch: soft-knee STF + linked midtones MTF
  identify/
    solver.py                 # NON-PORTABLE SEAM: astropy.wcs fast path + external solver; emits result contract
    catalog.py                # DSOCatalog: CSV load, haversine, prominence, dedup, findInField/findByNameOrId
    classify.py               # objectClass rules + data-type (FILTER-keyword) classify
  palettes/
    combine.py                # mono + OSC narrowband channel algebra, driven by JSON maps
  pipeline/
    params.py                 # eff* resolution + all clamps (pure, data-driven)
    steps.py                  # each finishing step (SCNR, contrast, saturation, bg-level, dehaze, ...)
    runcore.py                # runPipelineCore: ordered step-list assembly + gating — the structural spine
  processes/                  # numpy/scipy reimplementations of PI *processes* (highest divergence risk)
    ht.py curves.py saturation.py abe.py rangemask.py scnr.py hdr.py lhe.py mmt.py
  external/                   # proprietary/non-deterministic wrappers, all feature-detected
    blurx.py noisex.py starx.py spcc.py gradientcorrection.py   # each: is_available() + run()/skip
  io/
    image_io.py               # FITS / TIFF / PNG read+write + XISF READ (write unverified, see §11); the I/O layer PI gave for free
    recipes.py                # byte-compatible .lsrecipe read/write
    memory.py                 # per-object memory + cross-session prefs → config file (platformdirs)
  gui/                        # PySide6 native GUI — depends INWARD only; never imported by the core
  objects/
    model.py                  # ClassProfile dataclass, LazyStretchParameters state object
  cli.py  app.py              # headless entry point + GUI entry point
```

**Separation of concerns (load-bearing):**
- **`data/` is the only place numbers live.** Every other module reads from `loader`. A profile/threshold tweak is a JSON-only edit — no Python change (§9).
- **The core runs fully headless.** `pipeline.runcore` operates on numpy arrays and returns numpy arrays. The GUI is a thin shell that feeds arrays in and renders arrays out. The golden-test harness never imports the GUI.
- **`external/` quarantines everything proprietary/non-deterministic** behind `is_available() + run()`. The pipeline degrades gracefully (PI's try/skip pattern), and the harness stubs these out.
- **`processes/` isolates the reimplementation risk** — each PI process reimplementation carries its own Layer-2 parity test.
- **`identify/solver.py` is the hard non-portable boundary.** Its output contract (`ra, dec, resolution, pixScaleAsec, focal, width, height, fovDeg, solved, fromExisting`) is identical to PI's `AstrometryEngine.makeResult`, so everything downstream is solver-agnostic.

**Image data model:** float32, normalized `[0,1]`, shape `(H, W)` mono or `(H, W, 3)` RGB. "channel-avg" statistics loop over channels and divide by N (a **linked** stretch — one transform for all channels). Reimplement PI's three stat primitives exactly: `median()` = `np.median`; **`avgDev()` = mean absolute deviation about the median** = `np.mean(np.abs(x - np.median(x)))` (NOT std, NOT MAD-median); `mean()` = `np.mean`.

**Channel-count subtlety (silent-divergence trap):** the source mixes two channel counts — `applyAutoStretch` and `backgroundLevel` average over `numberOfChannels` which **includes an alpha plane** (`:2086, :2485`), while `measureDust` and `autoAssess` use `numberOfNominalChannels` which **excludes alpha** (`:2165, :2549`). On an RGBA frame a naive port that averages "all channels" everywhere would fold the alpha plane into the stretch/background statistic and silently diverge from PI. **Port contract: strip alpha on load** (simplest, matches the common case). Only replicate PI's per-function channel choice if RGBA round-trip fidelity is ever required — in which case encode the nominal-vs-all choice per stat call.

---

## 3. The pipeline port — step-by-step table

21 steps, grouped by porting tier. Priority: **P0** = required for a faithful v1 (the Preview subset); **P1** = important finishing detail; **P2** = external/AI, post-v1.

### Tier: EASY (pure array algebra — port verbatim)
| # | Step | PI process | Python approach | Prio |
|---|---|---|---|---|
| 1a | Clone / work image | `ImageWindow.assign` | `arr.copy()` | P0 |
| 1b | Narrowband combine (mono) | ChannelCombination | `np.stack` per palette table | P0 |
| 1c | Narrowband combine (OSC sim) | PixelMath | `ha=R; o3=(G+B)/2`; assemble per palette | P0 |
| 4 | Crop edges | Crop | `arr[my:H-my, mx:W-mx]`; update WCS CRPIX | P0 |
| 10 | Remove green (SCNR) | SCNR AverageNeutral | `G = min(G, (R+B)/2)`, optional lightness restore | P0 |
| 13 | Contrast curve | CurvesTransformation | scipy `Akima1DInterpolator` through 4 points → LUT | P0 |
| 15 | Reduce background cast | PixelMath + RangeSelection | `C*0.7 + luma*0.3`, composited through bg mask | P1 |
| 16 | Enhance emission (Ha) | PixelMath/Sat/Curves | red-excess mask → sat + Akima S through mask | P1 |
| 21 | Reset screen STF | (display only) | **Drop** — no separate screen stretch in a standalone app | — |

### Tier: MEDIUM (reimplement + validate against PI output)
| # | Step | PI process | Python approach | Prio |
|---|---|---|---|---|
| 2 | Auto-assess (crop% + GC/ABE pick) | (math only) | 8 perimeter patches → per-channel STF → visual-luminance opposition score → thresholds | P0 |
| 5 | Background/gradient (linear) | **GradientCorrection** (stock PI, primary) / ABE (fallback) | GraXpert background-extraction for GC-grade results; `photutils.Background2D` or degree-4 `Polynomial2D` + sigma rejection as the portable fallback PI itself uses. **Runs in Preview — not an AI-wall step.** | P0 |
| 7 | Color calibration (BN+CC) | BackgroundNeutralization + ColorCalibration | BN: per-channel offset to target 0.001; CC: structure/star white reference + per-channel scale | P0 |
| 9 | **Auto-stretch (STF→HT)** | PixelMath + HistogramTransformation | **crown jewel** — soft-knee + MTF LUT (§4) | P0 |
| 11 | HDR core compression | HDRMultiscaleTransform | à-trous/starlet decomposition; compress large-scale residual on lightness, hue-preserved | P1 |
| 12 | Lower background to target | HT + RangeSelection + measureDust | MTF midtones pull (global or masked); adaptive floor raise | P0 |
| 14 | Saturation boost | ColorSaturation | HSV/CIELab chroma scale (approximation — calibrate vs PI). **Star-protected variant depends on the StarX wall** (`saturationMasked→buildStarMask`, :2704-2722) → falls back to global saturation when StarX absent | P0 |
| 17 | Neutralize haze (dehaze) | PixelMath + RangeSelection | 2-pass: soft veil subtraction + class-aware brownness desaturation | P1 |
| 18 | Gradient cleanup (post-stretch) | ABE degree-1 + PixelMath | robust plane fit, subtract, blend by strength | P1 |
| 19 | Local contrast | LocalHistogramEqualization | skimage CLAHE approximates; reimplement circular-window LHE for fidelity | P1 |

### Tier: WALL (proprietary AI / external DB — external shell-out, optional, skippable)
| # | Step | PI process | Substitute + gap | Prio |
|---|---|---|---|---|
| 3 | Color cal before crop (SPCC-early) | SPCC | SPCC-lite (solve + Gaia DR3 XP). Runs before crop because Crop deletes WCS | P2 |
| 6 | Deconvolution / sharpening | BlurXTerminator | GraXpert deconv (~60–75%) / RL. **Skipped in Preview** | P2 |
| 8 | Noise reduction | NoiseXTerminator | GraXpert (~80–90%) **or** portable MultiscaleMedianTransform fallback — median multiscale, *not* à-trous (~100% of PI fallback). **Skipped in Preview** | P2 (MMT path P1) |
| 20 | Star reduction | StarXTerminator | StarNet2 for starless; recombine math is trivial numpy. **Skipped in Preview** | P2 |

**Execution model to preserve:** the step list is **built first with all conditions resolved up front**, so a determinate `[i/total]` progress counter reflects exactly what will run. Each step runs in its own try/except and **continues on error** (PI's per-step fault isolation, which defends against deferred GPU-OOM from plugins). StarX and star-mask are **transactional**: run the remover on a clone, derive `stars = orig − starless`, leave the working image untouched until the final recombine, so a failure cleanly skips. The masking primitive is universal: `out = orig*(1-mask) + processed*mask` (or `1-mask` inverted).

**Mode — "Input already stretched (polish only)" (`inputStretched`).** A first-class pipeline *branch*, not just a toggle (`:3344-3383`). When set, the input is a finished, **non-linear / gamma-encoded** image (JPEG/TIFF) and the pipeline **skips: auto-stretch (step 9), ALL colour calibration — SPCC *and* BN+CC (steps 3 & 7) — and BlurX (step 6)** (`:3357-3358`), plus it disables the adaptive floor raise; it runs only the finishing steps (SCNR, HDR, bg-level, contrast, saturation, cast, emission, dehaze, gradient cleanup, local contrast, star reduction). This is the "fix a double-stretched image" path. **Implications:** `io/image_io` must accept non-linear 8/16-bit inputs (not only linear masters), and the golden harness must treat this as a **separate pipeline configuration** with its own fixtures.

---

## 4. The numerical core — the crown jewel

This is the first thing to build and the thing every golden test flows through. All pixel data is float `[0,1]`. Build these in `stats/mtf.py`, `stats/measure.py`, `stretch/autostretch.py` with the constants pinned from the source.

### 4.1 The two PixInsight primitives (must be bit-faithful)

**`Math.mtf(m, x)` — Midtones Transfer Function.** The single nonlinearity behind every stretch, background pull, and assessment score. A mismatch here fails *all* golden tests at once, so pin it and test it first.
```
x<=0 -> 0 ; x>=1 -> 1 ; x==m -> 0.5 ; else ((m-1)*x)/((2*m-1)*x - m)
```
Vectorized: compute the general branch, then `np.where(x<=0,0,...)`, `np.where(x>=1,1,...)`. `m ∈ (0,1)`.

**`Math.range(v, lo, hi)`** = `np.clip`.

### 4.2 Statistics semantics (easy to get wrong)
- `median()` = `np.median`.
- **`avgDev()` = mean absolute deviation about the median** = `np.mean(np.abs(x - np.median(x)))`. This is **not** std and **not** MAD-median. The whole stretch is built on avgDev (for a normal dist `avgDev ≈ 0.7979·σ`). The PI PLAN mentions Conejero's `median − 2.8·1.4826·MAD`; LazyStretch **deviates** and uses avgDev directly. Port avgDev, not `1.4826·MAD`.
- Regional/per-channel stats = array slicing + channel indexing.

### 4.3 `findMidtonesBalance(targetBkg, value) -> m ∈ [0,1]`
Bisection inverting the MTF — "what `m` maps `value` to `targetBkg`?"
```
if value<=0: return 0; if value>=1: return 1
v0 = clip(targetBkg,0,1); eps = 5e-5
[m0,m1] = [0,0.5] if value<v0 else [0.5,1]
loop: m=(m0+m1)/2; v=mtf(m,value); if |v-v0|<eps: return m; if v<v0: m1=m else m0=m
```
Port verbatim (keep the eps and bracket selection identical for parity). `scipy.optimize.brentq` is acceptable only if it matches to `atol 1e-4`.

### 4.4 `applyAutoStretch(img, shadowsClip<0, targetBkg)` — THE stretch
```
median = mean_c(median_c);  avgDev = mean_c(avgDev_c)
c0 = clip(median + shadowsClip*avgDev, 0, 1)          # shadowsClip is NEGATIVE, avgDev units
e  = max(0.30*avgDev, 1e-5)                            # soft-knee width, floored so e^2 is representable
# SOFT shadows knee (softplus-like), applied to ALL pixels ONLY IF 0<c0<1:
softclip(T) = ((T-c0) + sqrt((T-c0)^2 + e^2)) / (2*(1-c0))
softMed = kneeRan ? softclip(median) : median
m = findMidtonesBalance(targetBkg, softMed)
if 0<m<1 and m!=0.5: img = mtf(m, img)                 # linked, black=0 white=1, NO further clip
```
**Critical, do not "fix":** solve `m` from `softMed` (post-knee background), **not** from raw `median` — else background undershoots ~26% on dark stacks. The soft knee keeps sub-`c0` tones distinct in `(0, e/2]` instead of a hard wall.

### 4.5 `measureDust(img) -> {lift, cleanSky, typical}` — self-calibration
6×4 grid (24 patches), patch = 7% of W/H, centers span 8%→92%. Each patch value = channel-avg median inside its rect (skip patches <4px/side; return zeros if <4 valid). Sort ascending; `q(p)=vals[floor(p*(n-1))]`; `cleanSky=q(0.10)`, `typical=q(0.45)`, `lift=max(0, typical-cleanSky)`. Zero on a clean field by construction — that's why it generalizes. Measured on the **stretched** image.

`adaptiveFloorRaise(lift) = clip(lift, 0, 0.03)`. Applied only if `adaptiveFloor and !inputStretched and class ∈ {emission,generic}`: `effFloor = clip(effBgLevel + raise, 0.015, 0.40)`. `effFloor` then feeds the background pull **and** is the contrast-curve pivot.

### 4.6 `autoAssess(img)` — perimeter-opposition crop/GC picker
On the **linear** image (skip if W<128 or H<128). Per channel: `c0=clip(med-2.8*dev,0,1)`, `m=findMidtonesBalance(0.25, med-c0)`. `vlum(region)=mean_c( mtf(m_c, clip((median_region_c - c0_c)/(1-c0_c),0,1)) )`. Sample 8 perimeter patches (4 corners + 4 edge midpoints, size 0.09, inset 0.02). Sort the 8 vlum values: `vmin=R[0]` (keep darkest — real crop signal), **`vmax=R[len-2]`** (drop the single brightest to reject a lone off-center object halo). `gradient=(vmax-vmin)/vmax`. Tiers: `≥0.45 → crop 8% + GC`; `≥0.20 → crop 6% + GC`; else `crop 3% + ABE`.

### 4.7 Per-class effective values (the "look" resolution)
```
effBkg      = clip(prof.bkg      + brightAdj,    0.05, 0.50)
effSat      = clip(prof.sat      + satAdj,       0.00, 1.50)
effClip     = clip(prof.clip     + blackAdj,    -3.00, -0.20)
effBgLevel  = clip(prof.bgLevel  + bgAdj,        0.015, 0.40)
effContrast = clip(prof.contrast + contrastAdj,  0.00, 0.40)
effFloor    = clip(effBgLevel + adaptiveRaise,   0.015, 0.40)
```
`lc`, `starLevel`, `hdrLayers` are used directly from the profile (no slider). Model this as a pure function `resolve_effective(profile, sliders) -> Effective` — trivially unit-testable and identical across GUI/CLI.

### 4.8 Validation against PI (§9 Layer 1 + 2)
Validate the core on the sample masters under `/Users/kstevica/Dev/Astro/LazyStretch/example/` (M31, Horsehead, M42 sessions, M78/Witch Head, M45, Melotte 15 SHO trio, astrobackyard emission set). PI already **logs** `median/avgDev/c0/m` (from applyAutoStretch), `gradient/vmin/vmax/crop/GC` (autoAssess), `lift/cleanSky/typical/effFloor` (measureDust), and resolved `eff*`. A thin telemetry mode dumps these as JSON sidecars; Python is fed the **same measured raw stats** and must reproduce every derived number: exact for integer/class/tier outputs (crop%, GC-vs-ABE, class, palette), `atol 1e-4` for `m/c0/lift/eff*`.

---

## 5. Object model & data

### 5.1 Classes and the profile table
Seven classes: `galaxy, emission, reflection, planetary, globular, open, generic` (unknown → `generic`). The profile table is the single most sync-critical, most-tweaked artifact — it lives as **data**, not code (§9). Verbatim from `LS.profiles` (lines 948–956):

| class | bkg | clip | sat | contrast | bgLevel | lc | scnr | localC | bxt | nr | hdr | hdrLayers | starReduce | starLevel |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy | 0.16 | −1.30 | 0.45 | 0.12 | 0.13 | 0.10 | ✓ | ✓ | ✓ | ✓ | ✓ | **8** | ✓ | 0.60 |
| emission | 0.25 | −1.05 | 0.55 | 0.12 | 0.20 | 0.15 | ✓ | ✓ | ✓ | ✓ | ✓ | (6) | ✓ | 0.50 |
| reflection | 0.22 | −1.15 | 0.50 | 0.16 | 0.14 | 0.12 | ✓ | ✓ | ✓ | ✓ | ✗ | — | ✓ | 0.60 |
| planetary | 0.18 | −1.50 | 0.55 | 0.14 | 0.09 | 0.10 | ✓ | ✗ | ✓ | ✓ | ✓ | (6) | ✗ | 0.70 |
| globular | 0.18 | −1.60 | 0.30 | 0.10 | 0.10 | 0.08 | ✗ | ✗ | ✓ | ✓ | ✗ | — | ✗ | 1.00 |
| open | 0.20 | −1.50 | 0.35 | 0.10 | 0.10 | 0.08 | ✗ | ✗ | ✓ | ✓ | ✗ | — | ✗ | 1.00 |
| generic | 0.25 | −1.25 | 0.40 | 0.12 | 0.13 | 0.12 | ✓ | ✓ | ✓ | ✓ | ✗ | — | ✗ | 0.70 |

`hdrLayers` defaults to `None`; only galaxy sets 8 (others use 6 in `hdrCore`). The six bool fields (`scnr/localContrast/bxt/nr/hdr/starReduce`) drive the six pipeline toggle checkboxes via `applyClassProfileToOptions` — **changing class rewrites those toggles from the profile.** Model as a frozen `ClassProfile` dataclass loaded from JSON.

**Class predicates (data, coded as sets):**
- `adaptiveFloorAppliesTo = {emission, generic}` — floor raise only here.
- `maskedDarkenAppliesTo = {emission, reflection, generic, open}` — background darkening through faint-signal mask.
- `starProtectAppliesTo = {emission, reflection, planetary, generic}` — saturation through inverted star mask; galaxy/globular/open use global saturation.

### 5.2 Identification & plate-solving
Two collaborators: an `AstrometryEngine` (gets RA/Dec/resolution) and a `DSOCatalog` (solved center → ranked named list); `objectClass` maps the winner to a class.

- **Solve (the non-portable seam):** PI drives its internal ImageSolver. The port swaps in:
  - **Fast path (the common case):** existing WCS via `astropy.wcs.WCS(header)`; derive resolution from CD/CDELT, FOV from dims.
  - **Solve path:** **ASTAP** CLI (`astap -f img.fits -r <radius> -fov <deg> -wcs`, fully offline, self-contained star DBs) as primary; `astrometry.net` local `solve-field` or `astroquery` as alternatives.
  - Preserve the SPCC-before-crop ordering: Crop deletes the WCS, so re-solve after crop if color cal needs coordinates.
  - **The result object is the stable contract** (`ra, dec, resolution, pixScaleAsec, focal, width, height, fovDeg=sqrt(w²+h²)·res, solved, fromExisting`). Everything downstream is solver-agnostic.
- **Catalog search (fully portable):** `findInField(centerRa, centerDec, fovRadiusDeg)` keeps rows where `angularSep ≤ fovRadius + objRadius`, ranks by `prominence = 0.50·sizeScore + 0.28·magScore + 0.22·centerScore + 0.12·(Messier bonus)`, then dedupes by `round(ra*100)_round(dec*100)` bucket (collapses M42/NGC1976 twins). Haversine great-circle for separation. Show up to 12 candidates.
- **Classification (`objectClass`, pure data+regex, port verbatim in priority order):**
  1. Messier number → `messierClass` buckets (SNR M1 → `emission`).
  2. NGC/IC override → `reflectionCat` / `emissionCat` lists (runs **before** name test — fixes IC2118 "Witch Head" and IC1848 "Soul").
  3. Common-name keywords (order matters: reflection before the `/nebula/` catch-all).
  4. Geometric heuristic: `axisRatio≥2.2 and diameter<15 → galaxy`.
  5. Fallthrough → `generic`.

### 5.3 Narrowband & palettes
Data-type detection reads the FITS **FILTER** keyword (regex → narrowband/broadband/osc), falling back to channel geometry. `paletteKey` reduces a display string to `{SHO, HOS, HOO, None}`. Combine is pure channel algebra:
- **Mono:** SHO `R=SII|Ha, G=Ha, B=OIII`; HOS `R=Ha, G=OIII, B=SII|OIII`; HOO `R=Ha, G=OIII, B=OIII` (SII optional, falls back).
- **OSC sim:** `ha=$T[0]`, `o3=($T[1]+$T[2])/2`; SHO `R=ha, G=(ha+o3)/2, B=o3`; else HOO `R=ha, G=o3, B=o3`.
- Narrowband skips color calibration entirely and is **not cloned** (freshly built). Store a canonical palette-id internally with a display-label map; keep a shim so old recipes storing `"SHO (Hubble)"` still map.

### 5.4 Per-object memory & recipes
- **Per-object memory** restores the look a user dialed for a target. Key = `normalize(id)` = `upper().replace(/[^A-Z0-9]/,"")` (`"M 42"→"M42"`) — **port verbatim; it is the identity contract.** Saves **9 reals** (`satAdj, brightAdj, bgAdj, blackAdj, contrastAdj, cropPercent, gradientCleanup, dehaze, smallStars`) + **6 bools** (`useMask, starProtect, doHDR, doStarReduce, doLocalContrast, doNR`). Restore is guarded: only loads when saved data exists (Identify never destroys unsaved slider work), and layers **on top of** class defaults.
- **Recipes (`.lsrecipe`, plain JSON, ~700 B):** magic `lazystretch_recipe:1` + version + `objectClass` + `palette` + 9 reals + 18 bools. Load is **tolerant**: type-check, clamp to `recipeRanges`, ignore unknown keys (forward-compat is an explicit design goal — do NOT use strict schema validation). Preserve the nuance: loading a recipe sets class **without** re-firing the profile toggles, so the recipe's explicit toggles win.

### 5.5 What becomes shared data files
Everything in §5.1–5.3 that is a table (profiles, sliderMap, recipeRanges, predicates, autoAssess/measureDust constants, Messier/NGC/IC catalogs, palette maps) moves into **one** `data/lazystretch_data.json` — the single source of truth both languages read (§9). The DSO CSVs (Messier.csv + NGC-IC.csv) ship **with the app** (converted once from PI's AdP folder), not read from a PixInsight install; consider a k-d tree at load since NGC-IC is ~575 KB and the current search is a linear scan.

---

## 6. GUI plan

**Framework: PySide6 (Qt for Python).** Rationale: (1) **lowest sync cost** — PixInsight's PJSR GUI *is* a Qt binding, so the port maps nearly 1:1 (`Dialog→QDialog`, `Vertical/HorizontalSizer→QVBox/QHBoxLayout` with stretch factors, `GroupBox→QGroupBox`, `ComboBox→QComboBox`, `CheckBox→QCheckBox`, `NumericControl→QSlider+QDoubleSpinBox` composite, rich-text labels/tooltips→`QLabel(Qt.RichText)`). Every future PI widget has an obvious Qt counterpart. (2) **Licensing:** LGPLv3 — closed-source single-binary shipping is fine as long as Qt stays a replaceable dynamic lib (PyInstaller/Nuitka bundles satisfy this). This rules out PyQt6 (GPL/commercial). (3) **Image display:** `QGraphicsView + QGraphicsPixmapItem` gives turnkey zoom/pan of large astro bitmaps (optionally GPU via `QOpenGLWidget`) — the thing the port wants and the current code lacks. (4) **Threading:** `QThread + signals/slots` is the canonical way to run the numpy pipeline off the UI thread. Runner-up: wxPython (truly native Cocoa look) — pick only if native widgets outrank sync cost.

**Layout (mirrors PI):** top full-width banner; below it a horizontal split — **left** = two natural-width control columns of `QGroupBox`es (Target View, Identify, Output/class/palette/crop, Narrowband channels | Adjustments, Options); **right** = stretch-weighted vertical split, **preview pane (~70% height) over info+buttons (~30%)**. Resizable down to ~1200×780.

**Control-by-control mapping:**
| PI control | Qt widget | Behavior to preserve |
|---|---|---|
| Target View `ViewList` | `QComboBox` of open/loaded images | on-change → re-run `classify()`, rebuild palette combo |
| Identify button | `QPushButton` → worker | plate-solve/read-WCS → fill candidates → auto-set class → load memory |
| Object combo | editable `QComboBox` | up to 12 ranked candidates; free text accepted |
| Data type | read-only `QLabel` | set from FILTER/geometry |
| Object class | fixed `QComboBox` | on-change **overwrites 6 option checkboxes** from profile |
| Palette | dependent `QComboBox` | items rebuilt from data type; reset to items[0] |
| Crop % | `NumericControl` 0–10 (0.1) | slider 0–100 mapped to 0–10 |
| Auto crop+GC | `QCheckBox` | when on, makes crop slider + GC checkbox advisory-only |
| Ha/OIII/SII pickers | 3× `QComboBox` of images | mono narrowband inputs |
| 5 look sliders + 3 extra dials | 5+3 `NumericControl` | additive nudges (§4.7); extras = direct strengths |
| Remember / Forget / Reset / Save+Load recipe | checkbox + 4 buttons | memory + `.lsrecipe` I/O |
| Input already stretched (polish) | `QCheckBox` | **mode switch** — gates OFF stretch + all colour cal + BlurX + adaptive floor; enables the non-linear input path (§3) |
| ~22 Options checkboxes | `QCheckBox` grid | feature-availability probes annotate tooltips; may grey-out when absent |
| Preview frame | `QGraphicsView` | aspect-fit render of pipeline output array |
| detected/status labels | rich-text `QLabel` | detection summary + run status |

**State model:** a single mutable `LazyStretchParameters` object; each control's on-change handler writes directly into it (controls are the live source of truth flowing *in*). Reverse sync (params→controls) happens only at defined moments (class change, view change, reset, memory load, recipe load). **Preserve two couplings exactly:** (a) class change rewrites the 6 option checkboxes; (b) view change rebuilds the palette list and resets to items[0]. Persistence has four tiers → three port targets: cross-session prefs + per-object memory → config store (§5.4); recipes → `.lsrecipe`; the PI "New Instance" drag mechanism has **no native analog** — port as an optional export-settings file or drop it.

**Preview/Execute UX:** PI's `processEvents()` freeze is replaced by a `QThread` worker emitting `progress(i,total,name)` and `finished(ndarray)`. The pipeline already builds an ordered step list with per-step try/except and `[i/total] name` logging — this maps cleanly onto a determinate progress bar and a log pane. **Preview** = run the fast subset (BXT/NoiseX/StarX/starProtect/SPCC forced off) on a clone, render to the pane. **Execute** = run everything honored, at full res, and (unlike PI, which drops a window into a workspace) **write to a file and/or show the full-res result in the app's own view**. PI's "re-open dialog after Execute" loop becomes a normal non-modal main window that just stays open.

---

## 7. External tools & the AI wall

None of these are bundled into the pip wheel or the `.app`. They carry their own licenses, model files, and GPU runtimes (hundreds of MB). Resolve each at runtime with `shutil.which()` **plus a user-overridable path** stored in settings, and **degrade gracefully when absent** — exactly mirroring PI's `haveBlurX/haveNoiseX/haveStarX/haveSPCC/haveGradientCorrection` guards. Each lives in `external/<tool>.py` as `is_available() + run()/skip`.

| Need | Primary substitute | Invocation | Honest quality note |
|---|---|---|---|
| **Plate solve** | ASTAP | shell-out, read emitted `.wcs`/header via `astropy.wcs` | Fast, offline, self-contained DBs. Existing-WCS fast path needs no external tool. |
| **Background / gradient** | GraXpert `background-extraction` | pip `graxpert` or app shell-out; **photutils Background2D / degree-4 Polynomial2D fallback** | ~parity for typical light-pollution gradients; the ABE fallback is the same fallback PI uses. |
| **Deconvolution (BlurX wall)** | GraXpert deconvolution | pip/shell-out; RL (`skimage.restoration.richardson_lucy`) fallback | **~60–75% — the hard ceiling of the whole project.** Make it optional and skippable; RL is markedly worse (ringing). |
| **Noise reduction (NoiseX wall)** | GraXpert denoise | pip/shell-out; **portable MultiscaleMedianTransform fallback** (median multiscale, *not* à-trous): 6 layers, NR on the two finest only — layer 0 thr 3.0 / amt 0.60 / 2 iters, layer 1 thr 2.5 / amt 0.40 / 1 iter | ~80–90% via GraXpert; the MMT path is a faithful port of PI's *own* fallback (~100% of that degraded mode). Implement the **median** transform in `processes/mmt.py`, not a linear starlet — they produce different output. |
| **Star reduction (StarX wall)** | StarNet2 | shell-out (bundle the binary), then `stars=orig−starless`, `result=starless+starLevel·sign(d)·|d|^(1+smallStars)` | ~75–90%. The LazyStretch star *logic* ports 100%; only starless quality differs. |
| **Color cal (SPCC wall)** | SPCC-lite: solve → `astroquery.gaia` cone search → `photutils` DAOStarFinder → cross-match → GaiaXPy/rgbloom synthetic RGB → robust per-channel scale | multi-week build; network or local Gaia mirror | Approaches SPCC for broadband **with** sensor QE + filter curves; otherwise a good generic white reference. **Fallback = portable BN+ColorCalibration**, the honest default. Replicate the README warning: the CC fallback's structure white reference mutes all-Ha fields — prefer SPCC-lite on Ha-dominant frames and log the same warning. |

**Determinism note:** BlurX/NoiseX/StarX/SPCC substitutes are non-deterministic or approximate and are **excluded from pixel golden diffs** — the harness asserts only that they run, stay in `[0,1]`, and don't fully clip a channel.

---

## 8. Packaging & distribution

**Primary: pip / wheel to PyPI.** numpy, scipy, astropy all publish binary wheels for macOS universal2/arm64, win_amd64, and manylinux, so `pip install lazystretch` resolves the heavy native deps with zero bundling work — and it is the most **sync-friendly** channel (a new PI release is just a code change + version bump + `python -m build && twine upload`). Ship a GUI entry point (`[project.gui-scripts] lazystretch = "lazystretch.app:main"`); document `pipx install lazystretch` / `uv tool install lazystretch` (isolated venv avoids ABI clashes). Gate optional in-process AI/GPU behind extras: `[ai]` (ONNX/torch), `[cuda]` (GPU wheels) — kept out of the base install because the GPU-heavy work is delegated to external tools anyway.

**Secondary: one frozen binary for non-technical astrophotographers.** **PyInstaller onedir → signed/notarized `.app`** on macOS (and onedir/installer on Windows) — most mature freezer, best hook coverage for exactly this stack (numpy/scipy/astropy/Qt), fast builds. Built from the **same source tree** in a GitHub Actions matrix (macos + windows + ubuntu; no freezer cross-compiles), so the binary is a downstream CI artifact, never a fork. Keep **Nuitka** on the shelf as an optional later upgrade only if startup latency or anti-tamper matter. Avoid Briefcase (its scientific-wheel support lags exactly this core) and py2app (macOS-only, low activity).

**macOS signing/notarization (the biggest gotcha; does NOT apply to the pip install):**
- Hardened Runtime (`codesign --options runtime`) + entitlement `com.apple.security.cs.allow-unsigned-executable-memory` (CPython execs memory) + `--timestamp`.
- Sign **inside-out**, each nested dylib/so/framework individually — do **not** use `--deep`.
- **Use onedir/.app, NOT `--onefile`** on macOS (onefile unpacks to temp, fouls notarization/Gatekeeper, slows launch).
- PyInstaller drops `base_library.zip` in `Contents/MacOS`, which **fails notarization** — move it to `Contents/Resources`.
- Notarize with `notarytool` (altool retired), then `xcrun stapler staple`. Needs an Apple Developer ID ($99/yr). Ship universal2/arm64.

**External tools are never bundled** (§7): discovered via `shutil.which()` + user-set path in settings. This contract is identical across pip and frozen builds — decoupled from the packaging choice. Bundle only: the DSO CSVs, the wavelet/MMT fallback code, and (optionally) small ONNX models for an `[ai]` build.

---

## 9. The update / sync strategy (load-bearing)

The stated top-priority requirement: stay easy to keep in sync as the PI `.js` keeps evolving. The whole architecture is bent toward this. Grounded in the PI changelog, **~80% of edits are tunable-number tweaks** (profile softening, mask gains, thresholds), **occasionally** a structural change (new step, new param, reordering).

### 9.1 Shared data layer — one JSON, both languages read it, zero codegen
**Key leverage:** LazyStretch.js already parses external JSON at runtime (`JSON.parse(File.readLines(...))` for recipe load). So moving the data tables out of inline literals into a bundled **`shared/lazystretch_data.json`** and reading it at startup is a proven, low-risk PJSR pattern — no new capability needed. This inverts ownership: **data lives in JSON; both `.js` and Python load it.** A number tweak then propagates with literally zero re-porting. (Reject the reverse — a JS-literal extractor is fragile against comments, trailing commas, and the bool/real mix.)

> **Caveat — the payoff is conditional.** "Zero re-porting on a number tweak" only materializes once the upstream `.js` is (and stays) refactored to source these tables from `shared/lazystretch_data.json` at startup. On any release *before* that refactor lands, DATA-ONLY hunks must still be hand-mirrored into the JSON. This is viable precisely because the port author **also owns the PI script** — the JS-side refactor is a one-time investment, not an ongoing patch that must be re-applied each release.

**Contents (two tiers, adopt incrementally):**
- **Tier 1 (do first, highest ROI):** `profiles`, `classList`, `sliderMap` (adj→profile field + clamp), `recipeRanges`, `classPredicates` (the three sets), `autoAssess` (sigma −2.8, target 0.25, patch 0.09, inset 0.02, tiers, minFrame 128), `measureDust` (6×4, patch 0.07, span 0.08–0.92, q10/q45, cap 0.03), `catalogs` (messierClass / reflectionCat / emissionCat), `palettes` (byType / combineMono / combineOSC).
- **Tier 2 (eliminates the second-biggest drift source):** `stepParams` — the ~40 per-process magic numbers (ABE degrees/tolerances, LHE radius 64 / slope 1.5, SCNR 1.0, NoiseX 0.80/0.15, contrast toe 0.02 / shoulder 0.45 / cap 0.92, stretch knee 0.30, dehaze knee 0.15 / tint cap 0.7 / mask 0.70 / key ×6, emission red-excess ×5, reduceCast 0.30, rangeMask fuzz 0.15/smooth 12, highlights 0.35/24/0.75, prominence 0.50/0.28/0.22/0.12).

`schema_version` bumps **only on shape change**, never on a value change. Echo `source_pi_version`. Python loads into typed dataclasses. **Clamp the JSON too** — apply both clamp layers (recipeRanges on load, sliderMap min/max at eff\* resolution) so a hand-edited JSON can't push a dial out of range.

**Reuse the existing neutral formats:** `.lsrecipe` and per-object memory already use identical field names across both languages. The port reads/writes **byte-compatible `.lsrecipe`** files — a recipe shared from PI loads in Python and vice-versa. Free cross-tool feature *and* a ready-made parity fixture.

### 9.2 Parity map — `parity.yaml`, one row per PI symbol
Each row: PI symbol + line range → Python module + `last_synced_sha`. Reference baseline SHA = `c997b73…` (PI 1.4.1); the ported code is at `9257d8b…` (PI 1.2.1) parity plus the colour-cal fix, so the §12 rows are the visibly-stale delta. Bump a row's SHA only when that symbol's source lines change; rows still pointing at an older SHA than `SOURCE.md` are **visibly stale** — the single best "what needs attention" signal after a PI bump. Optionally store a per-symbol content hash so a diff classifier auto-flags changed rows. Three buckets: **ports cleanly** (pure math/data — golden-testable), **ports as reimplementation** (PI process → numpy, medium divergence risk, per-step Layer-2 test), **does not port** (solver, the four AI walls, GUI, Settings — parity only at the I/O boundary). **Flag prominently:** `Math.mtf` is a PJSR built-in whose exact formula is pinned in `stats/mtf.py`; every number flows through it, so a mismatch fails all golden tests at once.

### 9.3 Golden / regression harness — three layers (BXT/NoiseX/StarX are non-deterministic, so no naive "diff final PNG")
- **Layer 1 — pure-math golden (the core, fully deterministic).** A hidden telemetry mode in `LazyStretch.js` (env var / `#define`) dumps the numbers PI already logs (median/avgDev/c0/m, gradient/vmin/vmax/crop/GC, lift/cleanSky/typical/effFloor, resolved eff\*, objectClass + prominence, measured bg) as JSON sidecars. Python is fed the **same measured raw stats** and reruns `findMidtonesBalance` / `adaptiveFloorRaise` / autoAssess tiering / eff\* resolution / objectClass. Diff: exact for integer/class/tier outputs, `atol 1e-4` for bisection/real outputs.
- **Layer 2 — ported-process parity.** *Exact ports* (true PixelMath expressions — crop, SCNR, bg-level HT pull, dehaze, reduceCast, ABE poly, range/star-mask composites, gradient-cleanup blend): export the PI single-step output as 32-bit FITS on a fixture and diff against the Python step on identical input (`atol ~1e-3`; summary-stat diff <0.5%). *Near-exact ports* (CurvesTransformation-based — contrast S-curve): diff only **after** the spline is calibrated to PI's **AkimaSubsplines** (scipy's Akima differs subtly), at a looser tolerance. **Excluded from pixel diffs** (undocumented PI colour spaces or non-deterministic): **saturation (ColorSaturation)** and **emissionBoost** — which internally calls ColorSaturation + Curves (`LazyStretch.js:2838-2841`) and therefore inherits the same non-parity — plus GradientCorrection, HDRMT, LHE, StarX, BlurX, SPCC, NoiseX. For all excluded steps assert only invariants: output stays in `[0,1]`, no channel fully clipped, and (for saturation) chroma moves in the expected direction.
- **Layer 3 — end-to-end smoke (invariants only):** full Python pipeline per fixture asserts result is non-linear (median lifted toward class target), in `[0,1]`, no channel 100% clipped, original untouched.

**Fixtures** (already present under `/Users/kstevica/Dev/Astro/LazyStretch/example/`, good class spread): M31 (galaxy/OSC); Horsehead / M42 sessions / IC1396 / Tulip / Lagoon / Cocoon / Soul(IC1848) / Rosette / Orion (emission); M78 + Witch Head IC2118 (reflection — exercises reflectionCat); M45 (open + IFN — exercises maskedDarken-without-floor-raise); Melotte 15 Ha/OIII/SII trio (SHO/HOO/HOS combine). Masters are 100–580 MB: **commit the small telemetry/FITS-crop sidecars, not the masters.** Wiring: `make golden-export` (run PI telemetry on the Mac, refresh sidecars) + `pytest tests/golden` (Layers 1+2, runs anywhere, no PI). Store expected values as `tests/golden/<image>.expected.json` tagged with the parity SHA they were captured at.

### 9.4 Per-release update checklist
1. Drop new `LazyStretch.js` into `reference/pixinsight/`; update `SOURCE.md` (SHA-1, version, date) → new parity baseline.
2. `git diff` the `.js`; classify every hunk as **DATA-ONLY** (number/list/set) or **STRUCTURAL** (new step/param/class/palette, reorder, changed formula).
3. DATA-ONLY hunk → edit `lazystretch_data.json` to the new value. **No Python change.** Don't bump `schema_version`.
4. Shape change (new field/tier/predicate) → bump `schema_version`, update the loader/dataclass, add a migration note.
5. STRUCTURAL hunk → edit the mapped Python module in `parity.yaml`. New parameter → replicate the **6-touchpoint fan-out** (defaults, persistence, per-object memory list, recipe bool/range list, param serialization, GUI control).
6. New/reordered step → update `pipeline/runcore` step assembly to match PI ordering, preserving all gate conditions (preview flags, isColor, narrowband, inputStretched, class predicates).
7. Bump `last_synced_sha` on every changed parity row; leave untouched rows stale.
8. On the Mac with PI: run telemetry export over fixtures, regenerate + commit sidecars.
9. `pytest tests/golden` (Layers 1+2); investigate any out-of-tolerance diff (Python bug vs intended output change → update sidecar + note it).
10. Layer 3 smoke on all fixtures.
11. Verify a `.lsrecipe` from the new PI version loads in Python (unknown keys ignored, values clamped) and vice-versa.
12. Update the port CHANGELOG: PI version synced, values changed, modules re-ported, stale rows, expected-output updates.

### 9.5 Module boundaries that isolate volatility
`data/` is the only home for numbers. `external/` quarantines non-determinism behind `is_available()+run()`. `processes/` isolates reimplementation risk (own Layer-2 tests). `runcore.py` is the one structural file that changes on a reorder — it builds an explicit ordered step list so a new step is a one-line insert and the list is introspectable for the parity map. `solver.py` keeps PI's `makeResult` output contract so the backend is swappable. `gui/` depends inward only — the whole pipeline runs headless so the harness never needs the GUI.

### 9.6 Versioning — three independent numbers, never conflated
1. **PI script version** (what the port is synced *against*) — in `SOURCE.md` + `source_pi_version`.
2. **`schema_version`** of the shared JSON — shape changes only.
3. **Python port semver** — per port release.
Keep a corpus of `.lsrecipe` files (one per class + edge cases: out-of-range, unknown future keys, missing keys) as cross-version, cross-language conformance fixtures; assert PI-written ⟷ Python-read round-trips survive every PI bump.

---

## 10. Phased roadmap

**Current status (2026-08-04):** P0, P1-core, P2, P3 (incl. masked step variants), P4 (native PySide6 GUI),
**P5 (external-tool integration), and P7 (full v1.4.1 catch-up: Chroma NR, dark-lane gradient, Remove stars,
starsAdj, dehaze-gradient, curated recipes, Deepen second-pass, Advisor/Analyze-Frame)** are **implemented and
passing** (117 tests; verified against the source, on real M31 + Melotte-15-SHO + Horsehead masters, via
fake-tool shell-out round-trips, and adversarially per new process). P5 wires StarNet
(reduction + star-protected saturation + star mask), GraXpert (denoise + background), and a local classical
Richardson-Lucy deconvolution (opt-in) — all feature-detected with graceful degradation. A colour-calibration
fix removed a violet cast on neutral OSC masters (BN+CC was re-tinting the background; now white-balances the
signal above background + re-neutralises — regression-tested). **Reference synced to PI v1.4.1 (2026-08-04);
the implemented pipeline is at v1.2.1 parity — the v1.2.1→v1.4.1 feature delta is the §12 catch-up backlog.**
**The Python pipeline now tracks v1.4.1 feature parity** (minus DATE-OBS + star census, both minor).

**P1 (golden calibration) — started 2026-08-04.** The `example/out/` folder is the golden set: PixInsight
v1.4.1 recipes + finished PNGs for ~10 targets. A reusable harness (`calibration/golden.py`) runs the port
with PI's recipe and compares the object core against PI's render. **First calibration — HDR core (M42):** the
port's `hdr_core` was *destroying* fine detail (log/multiplicative reconstruction shrank absolute detail with
the dimmed base → a flat/washed core). Rewrote it as a **linear base/detail HDR** — compress the base toward
its median, add the fine detail back at full absolute strength (`_HDR_BASE_COMPRESSION=0.65`,
`_HDR_DETAIL_BOOST=1.10`). On the M42 core, detailRMS 0.029 → 0.050 (target 0.041) and the Trapezium core now
shows pink HII + dust structure like PI's reference; locked by `tests/test_hdr_calibration.py`. A **10-target
survey** (`calibration/survey.py`) then ranked the port-vs-PI gaps: #1 was **highlight clipping** (the port
hard-clipped 0.2-1.2% of pixels to pure white, PI keeps near-white ≈0%). Fixed with a **highlight roll-off**
(`processes/highlights.py`, soft per-channel top-end knee, run last) — near-white ~halved on every target and
no pixel reaches pure white; residual gap is star-count (needs StarNet/GraXpert). **#3 saturation** — the port
scaled chroma linearly in RGB (over-boosting saturated colours + clipping, under-boosting subtle ones, PI uses
CIE L*c*h*); reimplemented as a **CIE Lab** chroma scale (`processes/tone.py`, perceptually uniform, round-trip
exact), removing the over/under extremes. Remaining P1: `measure_noise` vs MRS, RangeSelection falloff,
tone/brightness. Bigger lever = install StarNet + GraXpert (closes the star-colour/near-white/core-noise
confounds). Also remaining: full **SPCC-lite** (scaffolded); P6 packaging.

| Phase | Milestone | Concrete done-criterion |
|---|---|---|
| **P0** | **Shared data layer + core primitives** | `lazystretch_data.json` (Tier 1) + `loader` typed dataclasses; `stats/mtf.py` (`Math.mtf`, `findMidtonesBalance`) pass Layer-1 golden at `atol 1e-4` on all fixtures' telemetry. |
| **P1** | **Headless stretch core validated** | `applyAutoStretch`, `measureDust`, `adaptiveFloorRaise`, `autoAssess`, `resolve_effective` reproduce every PI-logged derived number (exact for tiers/class; `atol 1e-4` for reals) across the fixture set. This is the crown-jewel gate. |
| **P2** | **I/O + identification** | Read/write FITS (astropy), 16-bit TIFF (tifffile), **XISF read** (`xisf`; write only if a writer is confirmed, else output FITS/TIFF), PNG; existing-WCS fast path via astropy.wcs; `catalog` + `objectClass` + data-type classify pass golden on catalog fixtures; ASTAP shell-out solves at least one fixture. |
| **P3** | **Full AAA-free pipeline (the Preview subset)** | Headless `runcore` runs the complete deterministic pipeline (crop, ABE/GC-via-GraXpert-or-fallback, BN+CC, stretch, SCNR, HDR, bg-level, contrast, saturation, cast, emission, dehaze, gradient cleanup, local contrast, narrowband combine) end-to-end on every fixture; passes Layer-2 parity on the exact-port steps and Layer-3 smoke on all. **Faithful v1 product exists, CLI-only.** |
| **P4** | **Native GUI** | PySide6 app with full control mapping, QThread worker + determinate progress, `QGraphicsView` preview (zoom/pan), the two coupling behaviors, recipe + per-object memory persistence. Preview renders the fast subset; Execute writes full-res output. |
| **P5** | **External AI integration** | `external/` wrappers for StarNet2 (star reduction + star mask), GraXpert (denoise + deconv + background), SPCC-lite (solve + Gaia XP). All feature-detected, graceful-degrading, excluded from pixel golden diffs. Honest quality notes surfaced in the UI. |
| **P6** | **Packaging & distribution** | `pip install lazystretch` works from PyPI (pipx/uv documented); CI matrix produces a **signed, notarized macOS `.app`** (hardened runtime, inside-out sign, base_library.zip relocated, stapled) and a Windows onedir. External tools resolved via `shutil.which` + settings path. |

---

## 11. Risks & open questions

- **The `Math.mtf` definition is the linchpin.** It is a PJSR built-in not defined in the source. The standard MTF formula is pinned here, but it must be confirmed against the PI install before P0 closes — every stretch/background/assess number flows through it, so a mismatch fails all golden tests simultaneously.
- **SPCC is the single hardest-to-sync piece.** SPCC-lite is multi-week, needs a Gaia mirror or network, and degrades without sensor QE + filter curves. Ship BN+CC as the honest default; treat SPCC-lite as an explicitly-labeled approximation. Decide whether to bundle a Gaia DR3 XP subset or require network.
- **BlurXTerminator has no real substitute (~60–75% ceiling).** This is the parity gap most likely to disappoint an experienced imager. Make deconvolution optional and clearly labeled; set expectations in the UI.
- **ColorSaturation / LocalHistogramEqualization / HDRMultiscaleTransform** operate in PI-specific spaces with undocumented internals. HSV/CIELab saturation, CLAHE, and à-trous HDR are approximations needing visual calibration against PI output — budget iteration; they are excluded from strict pixel golden diffs.
- **Telemetry mode in the `.js` must be added and maintained.** The Layer-1/2 harness depends on it. It should stay a small, guarded addition that does not alter normal behavior; keep it in sync with any logging changes across PI releases.
- **6-touchpoint fan-out on new parameters** is the biggest structural re-port hazard. The checklist and parity map mitigate it, but a missed touchpoint (e.g. per-object memory or recipe range) silently drops persistence — worth a lint/test that asserts all four persistence tiers agree on the parameter set.
- **Non-RFC CSV parsing in PI** (naive `split(",")`) will differ from a real CSV reader on quoted commas in common names. Use a proper reader but preserve the exact column mapping; add a test that the parsed catalog matches PI's row count/fields.
- **Full-resolution performance.** PI Preview downscales; the port targets full-res interactive preview. à-trous/HDR/LHE on 100–580 MB frames may need tiling, downsampled preview, or GPU. Open question: preview-at-reduced-res vs full-res with a progress bar.
- **"New Instance" drag has no native analog.** Decide: export-settings file vs drop entirely. Low stakes, but a documented decision.
- **DSO catalog freshness.** Bundled Messier/NGC-IC decouple from PI, but PI may update its AdP catalogs; the sync process should re-export them when they change (part of the per-release checklist, currently implicit).
- **XISF write support is unverified.** The PyPI `xisf` package is reliable for **reading** XISF, but round-trip *write* support in the Python ecosystem is historically limited. Verify before committing to "write XISF" (§8/P2); if a writer isn't available, scope XISF to read-only and write results as FITS/TIFF (fully covered), or budget a small custom XISF writer. Low stakes for a stretching tool whose canonical output is TIFF/PNG.

---

## 12. v1.2.1 → v1.4.1 catch-up backlog

The reference was synced from PI **v1.2.1** to **v1.4.1** on 2026-08-04 (3542 → 4847 lines).
This followed the §9.4 checklist and validated the §9.1 design: **all data-layer changes propagated with
zero re-porting**, and the structural work is isolated below.

### 12.1 Data layer — DONE (propagated, no code logic)
- **Profiles and `classList` are unchanged** across 1.2.1→1.4.1 — no profile edits.
- New dials/params synced into `data/lazystretch_data.json` and added as inert fields on `Parameters`
  (recipes/memory round-trip today; the *behaviour* is the backlog below):
  - `recipeRanges`: `chromaNR [0,1]`, `deepen [0,1]`, `starsAdj [-0.3, 0.5]`
  - `persistedReals` += `chromaNR, deepen, starsAdj`  (9 → 12)
  - `recipeBools` += `darkLaneGC` (after `useGradientCorrection`), `removeStars` (after `doStarReduce`)  (18 → 20)
- `schema_version` stays **1** — these are value/list additions, not a shape change (design intent).
- **Corroboration:** 1.4.0 added an Advisor message (`LS.adviseSPCC`, `Pipe.colorCalibrate` red-suppression
  detection at `:3227`) that flags exactly the "CC mis-whites Ha-dominated frames" failure the port already
  fixed in `processes/colorcal.py`. The fix is aligned with upstream's own diagnosis.

### 12.2 Feature backlog — STRUCTURAL (to port for full v1.4.1 parity)

**P7 — DONE (2026-08-04, both parts):** Part 1 — Chroma NR (`processes/chromanr.py`), dark-lane gradient
(`processes/darklane.py`), Remove stars (`external/starx.remove_stars` + `stars_layer`), `starsAdj`, the
`dehaze` gradient arg, curated recipes (`objects/presets.py`). Part 2 — **Deepen second-pass**
(`processes/deepen.py`: `deepen_stretch` + `restore_source_highlights` with blue-halo/hue-of-gain restore)
and the **Advisor / Analyze-Frame** engine (`objects/analyze.py`: `measure_noise` + `analyze_view` →
lines + recommendations). All wired into `runcore`, CLI (`--deepen`, `--analyze`), and GUI (Deepen dial,
Analyze Frame + Apply-recommendations buttons); adversarially verified (one order-of-ops bug in the restore
mask fixed); tested (117 passing). **Remaining minor:** DATE-OBS (#9, astropy already handles header epoch)
and the star census (needs a star detector; emits "census unavailable"). Status below: ✅ done, ⏳ remaining.

| ok | # | Feature (param) | PI source | Tier | Python plan |
|---|---|---|---|---|---|
| ✅ | 1 | **`dehaze` gains a `gradient` arg** | `Pipe.dehaze(view,strength,floor,cls,gradient)` :3796 | **easy** | signature threaded (gradient-gated LP-tint refinement is a calibration TODO) |
| ✅ | 2 | **`starsAdj`** star-reduction nudge | `effLevel = starLevel + starsAdj` | **easy** | `runcore`: `eff_star = clip(starLevel + starsAdj, 0, 1)` |
| ✅ | 3 | **Chroma NR** (`chromaNR`) | `Pipe.chromaNR` :4037 | **medium** | `processes/chromanr.py` — median-multiscale on RGB-mean chroma (approx CIE a*/b*), luminance kept |
| ✅ | 4 | **Dark-lane gradient** (`darkLaneGC`) | `Pipe.darkLaneModel` :2985 / `darkLaneGradient` :3138 | **medium** | `processes/darklane.py` — 40-zone darkest-anchor deg-2 fit, subtract excess-over-min |
| ✅ | 5 | **Remove stars** (`removeStars`) | :4147-4214 | **medium** | `external/starx.remove_stars` → starless result + `stars_layer` output (saved as `<out>_stars`) |
| ✅ | 6 | **Deepen / second pass** (`deepen`) — *1.4.1 headline* | inputStretched path + `restoreSourceHighlights` :4224 | **medium-hard** | `processes/deepen.py` — 30% global MTF + 70% masked lift; restore rides source 30%, blends bright+blue-halo, damps blue-dominant gains |
| ✅ | 7 | **Advisor / Analyze Frame** | `LS.analyzeView` :2633, `LS.measureNoise` :2554 | **medium** | `objects/analyze.py` — MAD-based `measure_noise` + `analyze_view` (all rec branches verbatim); CLI `--analyze`, GUI Analyze/Apply buttons; star census skipped |
| ✅ | 8 | **Curated starting recipes** | preset recipe objects `:960-1005` | **easy (data)** | `objects/presets.py` (`curated_for` + `apply_preset`); auto-applied on Identify in CLI + GUI |
| ⏳ | 9 | **DATE-OBS solver fix** | solver metadata | **easy** | use `DATE-OBS` when solving (`identify/solver.py`) — astropy WCS already handles header epoch; minor |
| — | — | **Step snapshots** | `LS.snapshotWindows` :2880 | **N/A / low** | PI window-management to close stray tool windows — no analog in the port |

### 12.3 Suggested sequencing
Group the **easy** items (1, 2, 8, 9) into a quick pass, then the **medium** processing steps (3 Chroma NR,
4 dark-lane, 5 remove-stars) which are self-contained numpy/StarX additions, then **6 Deepen** (the largest —
a new nonlinear second-pass with protection), and finally **7 Advisor** as a GUI phase. None of these change
the existing v1.2.1-parity pipeline output; they are additive. Do this **catch-up (call it P7)** alongside or
before P1 golden validation, so the golden telemetry is captured against v1.4.1 rather than 1.2.1.
