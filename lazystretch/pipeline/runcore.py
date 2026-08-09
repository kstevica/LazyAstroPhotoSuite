"""The pipeline spine — headless port of ``runPipelineCore`` (LazyStretch.js:3193-3496).

Builds the ordered step list up front (so a determinate ``[i/total]`` progress reflects
exactly what will run), gates each step by the same conditions PI uses (preview flags,
isColor, narrowband, inputStretched, class predicates), and runs each in its own
try/except, continuing on error — PI's per-step fault isolation.

External "wall" tools (``Tools``: GraXpert, StarNet, SPCC-lite) are feature-detected and
used when installed, else the pipeline degrades: GraXpert denoise -> MMT; GraXpert
background -> ABE polynomial; StarNet reduction/star-protect -> skipped/global; SPCC ->
BN+CC. Preview forces the wall OFF (the deterministic, no-plugin subset). Deconvolution
has no open CLI equivalent, so BlurX -> optional classical Richardson-Lucy (off by default).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

from ..data.loader import LazyStretchData, get_data
from ..external import Tools
from ..identify.classify import palette_key
from ..objects.model import Parameters
from .ledger import INFO as LEDGER_INFO, MEASURED as LEDGER_MEASURED, Ledger
from ..palettes.combine import combine_narrowband, combine_osc_narrowband
from ..pipeline.params import (
    Effective,
    adaptive_floor_applies_to,
    effective_floor,
    masked_darken_applies_to,
    resolve_effective,
    star_protect_applies_to,
)
from ..processes import (
    background,
    backgroundlevel,
    brightcore,
    chromanr,
    colorcal,
    crop,
    darklane,
    deconv,
    deepen as deepen_mod,
    finishing,
    halo,
    highlights,
    masks,
    multiscale,
    scnr as scnr_mod,
    shadowanchor,
    tone,
    transparency as transparency_mod,
)
from ..stats.assess import adaptive_floor_raise, auto_assess, measure_dust
from ..stretch.autostretch import StretchResult, apply_auto_stretch


@dataclass
class PipelineResult:
    image: np.ndarray
    eff: Effective
    object_class: str
    eff_crop: float
    eff_gc: bool
    stretch: Optional[StretchResult] = None
    stars_layer: Optional[np.ndarray] = None   # from Remove stars: the extracted stars
    steps_run: List[str] = field(default_factory=list)
    steps_skipped: List[str] = field(default_factory=list)
    log: List[str] = field(default_factory=list)
    ledger: Optional[Ledger] = None            # PROC ledger: computed + pinned values


def _is_rgb(a: np.ndarray) -> bool:
    return a.ndim == 3 and a.shape[2] == 3


def run_pipeline(
    image: Optional[np.ndarray],
    params: Parameters,
    *,
    preview: bool = False,
    data: "LazyStretchData | None" = None,
    tools: Optional[Tools] = None,
    solve_result=None,
    pins: Optional[dict] = None,
    log: Optional[Callable[[str], None]] = None,
    progress: Optional[Callable[[int, int, str], None]] = None,
) -> PipelineResult:
    """Run the headless pipeline. Returns the finished image + a run report."""
    if data is None:
        data = get_data()
    if tools is None:
        tools = Tools.resolve()
    logs: List[str] = []
    ledger = Ledger(pins)          # PROC ledger: records every decision, applies pin overrides

    def _log(msg: str):
        logs.append(msg)
        if log:
            log(msg)

    # --- preview gating (js:3196-3201) ---
    do_bxt = params.doBXT and not preview
    do_nr = params.doNR and not preview
    do_star = params.doStarReduce and not preview
    star_pro = params.starProtect and not preview

    # --- narrowband detection (js:3205-3215) ---
    nb_key = palette_key(params.palette)
    nb_mono = nb_key is not None and params.ha is not None and params.oiii is not None
    view_is_color = image is not None and _is_rgb(np.asarray(image))
    nb_osc = nb_key is not None and not nb_mono and view_is_color
    narrowband = nb_mono or nb_osc

    # --- build the working image (js:3255-3281) ---
    if nb_mono:
        target = combine_narrowband(nb_key, params.ha, params.oiii, params.sii)
        _log(f"Combined mono narrowband {nb_key}")
    elif nb_osc:
        target = combine_osc_narrowband(np.asarray(image, dtype=np.float64), nb_key)
        _log(f"Combined OSC-simulated narrowband {nb_key}")
    elif image is None:
        raise ValueError("no input image (and no narrowband channels assigned)")
    else:
        target = np.asarray(image, dtype=np.float64).copy()

    # Original input kept as the source reference for the Deepen highlight-restore.
    source_ref = np.asarray(image, dtype=np.float64).copy() if image is not None else None

    is_color = _is_rgb(target)
    cls = params.object_class
    prof = data.profile_for(cls)
    eff = resolve_effective(prof, params.slider_dict(), data)
    # PROC ledger: surface (and let the user pin) the resolved "look" numbers.
    ledger.record("Run", "mode", "preview" if preview else "execute", kind=LEDGER_INFO)
    ledger.record("Profile", "object class", cls, kind=LEDGER_INFO)
    eff = Effective(
        bkg=ledger.record("Profile", "stretch background", eff.bkg),
        sat=ledger.record("Profile", "saturation boost", eff.sat),
        clip=ledger.record("Profile", "black point (avgDev units)", eff.clip),
        bgLevel=ledger.record("Profile", "background floor", eff.bgLevel),
        contrast=ledger.record("Profile", "contrast strength", eff.contrast),
    )

    # --- auto-assess crop % + GradientCorrection (js:3286-3306) ---
    eff_crop = float(params.cropPercent)
    eff_gc = params.useGradientCorrection
    assess_gradient = 0.0
    if params.autoAssess:
        try:
            a = auto_assess(target, data)
            if a is not None:
                eff_crop, eff_gc = a.cropPercent, a.useGC
                assess_gradient = a.gradient
                _log(f"Auto: crop {eff_crop:.0f}%, {'GradientCorrection' if eff_gc else 'ABE'} "
                     f"(gradient {a.gradient:.2f}, darkest {a.vmin:.3f}, brightest {a.vmax:.3f})")
            else:
                _log(f"Auto: frame too small to assess — using manual crop {eff_crop:.0f}%")
        except Exception as e:  # pragma: no cover - defensive, mirrors PI
            _log(f"Auto-assess failed: using manual crop — {e}")
    eff_crop = float(ledger.record("Crop", "edge crop %", eff_crop))
    eff_gc = bool(ledger.record("Background", "use GradientCorrection", eff_gc))

    ctx = {"img": target, "eff_floor": eff.bgLevel, "color_cal_done": False, "stretch": None}
    steps: List = []

    def add(name: str, fn: Callable[[], None]):
        steps.append((name, fn))

    # --- SPCC before crop (needs solve + astroquery/GaiaXPy; never in preview) ---
    spcc_early = (eff_crop > 0 and params.doColorCal and is_color and not narrowband
                  and not params.inputStretched and params.preferSPCC and not preview
                  and tools.spcc.is_available()
                  and solve_result is not None and getattr(solve_result, "solved", False))
    if spcc_early:
        def _spcc():
            out = tools.spcc.calibrate(ctx["img"], solve_result)
            if out is not None:
                ctx["img"] = out
                ctx["color_cal_done"] = True
                _log("   via SPCC-lite (before crop)")
            else:
                _log("   SPCC-lite deferred; BN + ColorCalibration will run after crop")
        add("Color calibration (SPCC before crop — crop deletes the WCS)", _spcc)
    elif params.preferSPCC and not preview and is_color and not narrowband and not params.inputStretched:
        _log("-- SPCC (before crop) skipped (needs a solve + GaiaXPy) — BN+CC will run after crop")

    # --- crop (js:3330-3333) ---
    if eff_crop > 0:
        def _crop():
            ctx["img"] = crop.crop_edges(ctx["img"], eff_crop / 100.0)
        add(f"Crop {eff_crop:.1f}% off each edge", _crop)

    # --- background / gradient (js:3336-3343) ---
    if params.doBgExtract:
        if eff_gc:
            def _bg():
                if tools.graxpert.is_available():
                    ctx["img"] = tools.graxpert.background_extraction(ctx["img"])
                    _log("   via GraXpert background-extraction")
                else:
                    ctx["img"] = background.background_correct(ctx["img"])
                    _log("   via ABE polynomial (GraXpert not installed)")
            add("Background / gradient correction", _bg)
        else:
            def _bg():
                ctx["img"] = background.background_extract(ctx["img"])
            add("Background / gradient extraction (ABE)", _bg)

    # --- dark-lane gradient (v1.4.x): deg-2 model anchored on dark lanes, for
    #     no-empty-sky wall-to-wall fields where ABE/GC over-flatten (js:4509-4511) ---
    if params.darkLaneGC and not params.inputStretched:
        def _darklane():
            ctx["img"] = darklane.dark_lane_gradient(ctx["img"])
        add("Dark-lane gradient model", _darklane)

    # --- deconvolution (BlurX wall -> optional classical RL) (js:3344-3354) ---
    if params.inputStretched:
        _log("-- Deconvolution skipped (input already stretched)")
    elif preview:
        _log("-- Deconvolution skipped (preview: BlurX off)")
    elif do_bxt:
        if params.useClassicalDeconv:
            def _dec():
                ctx["img"] = deconv.richardson_lucy(ctx["img"])
            add("Deconvolution (classical Richardson-Lucy — weak BlurX substitute)", _dec)
        else:
            _log("-- Deconvolution skipped (no BlurX; enable classical RL to approximate)")

    # --- color calibration (js:3355-3368) ---
    if narrowband:
        _log("-- Color calibration skipped (narrowband)")
    elif params.inputStretched:
        _log("-- Color calibration skipped (input already stretched)")
    elif params.doColorCal and is_color:
        def _cc():
            if ctx["color_cal_done"]:
                _log("   already calibrated (SPCC ran before crop)")
                return
            ctx["img"] = colorcal.color_calibrate(ctx["img"])
        add("Color calibration (BN + ColorCalibration)", _cc)
    elif params.doColorCal and not is_color:
        _log("-- Color calibration skipped (mono image)")

    # --- noise reduction is applied POST-stretch (after Deepen, below): the NN denoisers
    #     (DeepSNR / GraXpert) are trained on stretched data, unlike NoiseX's linear slot. ---

    # --- auto-stretch (js:3376-3384) ---
    if params.inputStretched:
        _log("-- Auto-stretch skipped (input already stretched)")
    else:
        def _stretch():
            out, res = apply_auto_stretch(ctx["img"], eff.clip, eff.bkg, data)
            ctx["img"] = out
            ctx["stretch"] = res
            ledger.record("Auto-stretch", "measured median", res.median, kind=LEDGER_MEASURED)
            ledger.record("Auto-stretch", "measured avgDev", res.avgDev, kind=LEDGER_MEASURED)
            ledger.record("Auto-stretch", "black point c0", res.c0, kind=LEDGER_MEASURED)
            ledger.record("Auto-stretch", "midtones m", res.m, kind=LEDGER_MEASURED)
            _log(f"   median={res.median:.4f} avgDev={res.avgDev:.4f} -> "
                 f"blackPoint={res.c0:.4f} midtones={res.m:.4f}")
        add("Auto-stretch (STF -> HistogramTransformation)", _stretch)

    # --- Deepen (v1.4.1): deterministic 2nd-pass lift, highlights protected (js:4554-4608) ---
    if params.deepen > 0:
        def _deepen():
            ctx["img"] = deepen_mod.deepen_stretch(ctx["img"], params.deepen)
        add(f"Deepen ({100 * params.deepen:.0f}% additional stretch, highlights protected)", _deepen)

    # --- noise reduction (post-stretch): DeepSNR > GraXpert > MMT. Relocated from PI's
    #     linear slot (js:3369) because the NN denoisers operate on stretched images. ---
    if preview and params.doNR:
        _log("-- Noise reduction skipped (preview: off)")
    elif do_nr:
        masked = params.useNRMask
        def _nr():
            if tools.deepsnr.is_available():
                processed = tools.deepsnr.denoise(ctx["img"]); via = "DeepSNR"
            elif tools.graxpert.is_available():
                processed = tools.graxpert.denoise(ctx["img"]); via = "GraXpert"
            else:
                processed = multiscale.noise_reduction_mmt(ctx["img"]); via = "MMT fallback"
            ctx["img"] = masks.background_masked(ctx["img"], processed) if masked else processed
            _log(f"   via {via}{' (background-masked)' if masked else ''}")
        add("Noise reduction", _nr)

    # --- nonlinear stage ---
    if params.doSCNR and is_color:
        def _scnr():
            ctx["img"] = scnr_mod.scnr(ctx["img"])
        add("Remove green (SCNR)", _scnr)

    if params.doHDR:
        layers = int(ledger.record("HDR", "layers", prof.hdrLayers if prof.hdrLayers else 6))
        def _hdr():
            ctx["img"] = multiscale.hdr_core(ctx["img"], layers)
        add(f"HDR core compression ({layers} layers)", _hdr)

    # --- lower background to target, with the adaptive floor (js:3401-3420) ---
    mask_darken = params.useMask and masked_darken_applies_to(cls, data)

    def _bglevel():
        eff_floor = eff.bgLevel
        if params.adaptiveFloor and not params.inputStretched and not adaptive_floor_applies_to(cls, data):
            _log(f"   adaptive floor: skipped for {cls} class (compact-object halo would inflate lift)")
        elif params.adaptiveFloor and not params.inputStretched:
            d = measure_dust(ctx["img"], data)
            raise_amt = adaptive_floor_raise(d.lift, data)
            eff_floor = effective_floor(eff.bgLevel, raise_amt, data)
            _log(f"   adaptive floor: dust lift {d.lift:.4f} "
                 f"(clean {d.cleanSky:.3f} -> typical {d.typical:.3f}) -> +{raise_amt:.3f} "
                 f"({eff.bgLevel:.2f} -> {eff_floor:.2f})")
        eff_floor = float(ledger.record("Background", "effective floor", eff_floor))
        ctx["eff_floor"] = eff_floor
        if params.useMask and not mask_darken:
            _log(f"   global darkening (faint-signal mask skipped for {cls} class)")
        if mask_darken:
            ctx["img"] = masks.background_level_masked(ctx["img"], eff_floor)
        else:
            ctx["img"] = backgroundlevel.background_level(ctx["img"], eff_floor)

    add(f"Lower background to ~{eff.bgLevel:.2f}{' (masked)' if mask_darken else ''}", _bglevel)

    # --- contrast (pivot = the possibly-raised floor) ---
    def _contrast():
        ctx["img"] = tone.contrast(ctx["img"], ctx["eff_floor"], eff.contrast)
    add(f"Contrast curve ({eff.contrast:.2f})", _contrast)

    # --- saturation (star-protected via a StarNet mask when available) ---
    if is_color:
        star_protect_here = star_pro and star_protect_applies_to(cls, data)
        def _sat():
            processed = tone.saturation(ctx["img"], eff.sat)
            if star_protect_here and tools.starx.is_available():
                sm = tools.starx.star_mask(ctx["img"])
                ctx["img"] = masks.apply_masked(ctx["img"], processed, sm, invert=True)
                _log("   star-protected (StarNet star mask)")
            else:
                if star_pro and not star_protect_here:
                    _log(f"   global saturation (star-protect skipped for {cls} class)")
                elif star_pro:
                    _log("   global saturation (StarNet not installed)")
                ctx["img"] = processed
        add(f"Saturation boost ({eff.sat:.2f})"
            + (" star-protected" if star_protect_here else ""), _sat)

    if is_color and params.reduceCast:
        def _cast():
            ctx["img"] = finishing.reduce_cast(ctx["img"])
        add("Reduce background color cast", _cast)

    if is_color and params.enhanceEmission:
        def _emis():
            ctx["img"] = finishing.emission_boost(ctx["img"])
        add("Enhance emission (Ha)", _emis)

    if is_color and params.dehaze > 0:
        def _dehaze():
            ctx["img"] = finishing.dehaze(ctx["img"], params.dehaze, eff.bgLevel, cls, assess_gradient)
        add(f"Neutralize haze ({params.dehaze:.2f})", _dehaze)

    if params.gradientCleanup > 0:
        def _gclean():
            ctx["img"] = finishing.gradient_cleanup(ctx["img"], params.gradientCleanup)
        add(f"Gradient cleanup ({params.gradientCleanup:.2f})", _gclean)

    if params.doLocalContrast:
        def _lhe():
            if params.protectCores:
                ctx["img"] = masks.local_contrast_masked(ctx["img"], prof.lc)
            else:
                ctx["img"] = tone.local_contrast(ctx["img"], prof.lc)
        add("Local contrast (LHE)" + (" core-protected" if params.protectCores else ""), _lhe)

    # --- chroma noise reduction (v1.4.x): MMT on chrominance only (js:4679-4681) ---
    if is_color and params.chromaNR > 0:
        def _chroma():
            ctx["img"] = chromanr.chroma_nr(ctx["img"], params.chromaNR)
        add(f"Chroma noise reduction ({params.chromaNR:.2f})", _chroma)

    # --- star handling: Remove stars (starless output) OR Star reduction (js:4702-4723) ---
    do_remove = params.removeStars and not preview
    if preview and (params.doStarReduce or params.removeStars):
        _log("-- Star handling skipped (preview: StarX off)")
    elif do_remove:
        if tools.starx.is_available():
            def _remove():
                starless, stars = tools.starx.remove_stars(ctx["img"])
                ctx["img"] = starless
                ctx["stars_layer"] = stars
                _log(f"   stars removed; stars layer kept (mean {float(stars.mean()):.5f})")
            add("Remove stars (starless result + stars layer)", _remove)
        else:
            _log("-- Remove stars skipped (StarNet not installed)")
    elif do_star:
        if tools.starx.is_available():
            eff_star = float(np.clip(prof.starLevel + params.starsAdj, 0.0, 1.0))
            eff_star = float(ledger.record("Stars", "reduction level", eff_star))
            def _starred():
                ctx["img"] = tools.starx.reduce_stars(ctx["img"], eff_star, params.smallStars)
                _log(f"   via StarNet (level {eff_star:.2f}"
                     f"{f' [class {prof.starLevel:.2f}{params.starsAdj:+.2f}]' if params.starsAdj else ''}, "
                     f"small stars {params.smallStars:.2f})")
            add(f"Star reduction (level {eff_star:.2f})", _starred)
        else:
            _log("-- Star reduction skipped (StarNet not installed)")

    # --- Halo Tamer (v1.5.1): suppress dual-band/LP reflection rings around bright stars.
    #     Auto scan + per-channel feathered-annulus subtraction, after star reduction so the
    #     ring is maximally exposed. Profile-driven (milkyway opts out). Detection is heavy
    #     (star census + radial profiles), so it runs on execute, not preview. ---
    ledger.record("Halo Tamer", "enabled", params.haloTamer, kind=LEDGER_INFO)
    if params.haloTamer and preview:
        _log("-- Halo Tamer skipped (preview: ring scan off)")
    elif params.haloTamer:
        def _halo():
            ctx["img"] = halo.halo_tamer(ctx["img"], log=_log)
        add("Halo Tamer (reflection ring suppression)", _halo)

    # --- restore source highlights: last step of a Deepen 2nd pass (js:4729-4733) ---
    if params.inputStretched and params.deepen > 0 and not narrowband and source_ref is not None:
        def _restore():
            ctx["img"] = deepen_mod.restore_source_highlights(
                ctx["img"], source_ref, eff_crop / 100.0, params.deepen)
        add("Restore source highlights (Deepen 2nd pass)", _restore)

    # --- core transparency (P1 calibration): large-scale local contrast masked to the
    #     bright nebulosity so dust lanes deepen and the core reads as a transparent
    #     structured cloud (like PI) rather than an opaque veil. Runs before the highlight
    #     roll-off so the roll-off then dims the (now-structured) veil. ---
    if params.transparency > 0:
        def _transparency():
            ctx["img"] = transparency_mod.core_transparency(ctx["img"], params.transparency)
        add(f"Core transparency ({params.transparency:.2f})", _transparency)

    # --- dim core (P1 calibration): mask the large bright veil and multiplicatively lower
    #     its luminosity. Unlike the highlight roll-off (which asymptotes at a ceiling), this
    #     can take an over-bright core well below it, structure + colour preserved. Opt-in. ---
    if params.dimCore > 0:
        def _dimcore():
            ctx["img"] = brightcore.dim_bright_core(ctx["img"], params.dimCore)
        add(f"Dim core ({params.dimCore:.2f})", _dimcore)

    # --- shadow anchor (P1 calibration): pull the (NN-denoised) milky sky floor toward
    #     PI's deep black point without dimming cores. Low-end mirror of the highlight
    #     roll-off; self-limiting, reflection/galaxy-gated, and identity at/above the sky
    #     median so cores are untouched. Acts on the sub-background shadows only (disjoint
    #     from the roll-off's >0.82 band). See calibration/survey.py (blk was the dominant
    #     execute-mode gap once the tools closed the noise gap). ---
    def _shadow():
        ctx["img"] = shadowanchor.shadow_anchor(ctx["img"], cls, log=_log)
    add("Shadow anchor (black-point calibration)", _shadow)

    # --- highlight roll-off (P1 calibration): always last. A soft top-end knee so no
    #     channel hard-clips to pure white — matches PI's controlled highlights and
    #     compensates for accumulated clipping (esp. LHE). Not a PI step; see
    #     calibration/survey.py (the biggest port-vs-PI gap was highlight clipping). ---
    def _rolloff():
        knee = float(ledger.record("Highlights", "roll-off knee",
                                   highlights.knee_for_dial(params.highlights)))
        ctx["img"] = highlights.highlight_rolloff(ctx["img"], knee)
        _log(f"   dial {params.highlights:.2f} -> knee {knee:.3f}")
    add(f"Highlight roll-off (dial {params.highlights:.2f})", _rolloff)

    # --- run the steps, isolated, continue-on-error (js:3465-3480) ---
    total = len(steps)
    steps_run: List[str] = []
    steps_skipped: List[str] = []
    for i, (name, fn) in enumerate(steps):
        if progress:
            progress(i + 1, total, name)
        _log(f"[{i + 1}/{total}] {name}")
        try:
            fn()
            steps_run.append(name)
        except Exception as e:
            steps_skipped.append(name)
            _log(f"   skipped: {e} (continuing)")

    return PipelineResult(
        image=np.clip(ctx["img"], 0.0, 1.0),
        eff=eff,
        object_class=cls,
        eff_crop=eff_crop,
        eff_gc=eff_gc,
        stretch=ctx.get("stretch"),
        stars_layer=ctx.get("stars_layer"),
        steps_run=steps_run,
        steps_skipped=steps_skipped,
        log=logs,
        ledger=ledger,
    )
