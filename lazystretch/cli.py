"""Command-line entry point — the headless LazyStretch pipeline.

Examples
--------
    lazystretch master.fits --class galaxy -o out.tif
    lazystretch osc.fits --identify -o out.tif
    lazystretch --ha Ha.fits --oiii OIII.fits --sii SII.fits --palette SHO \
                --class emission -o sho.tif
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from .data.loader import get_data
from .external import Tools
from .identify.catalog import get_catalog, object_label
from .identify.classify import object_class as classify_object
from .identify.solver import solve
from .io.image_io import LoadedImage, load_image, save_image
from .io.recipes import apply_recipe, load_recipe, save_recipe
from .objects.model import Parameters
from .objects.presets import apply_preset, curated_for
from .pipeline.runcore import run_pipeline


def _mono(path: str) -> np.ndarray:
    li = load_image(path)
    d = li.data
    return d[..., 0] if d.ndim == 3 else d


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="lazystretch",
        description="Standalone LazyStretch — statistics-driven, object-aware stretching.",
    )
    p.add_argument("input", nargs="?", help="input master (FITS/TIFF/PNG/XISF); omit for mono narrowband")
    p.add_argument("-o", "--output", help="output path (default: <input>_LazyStretch.tif)")
    p.add_argument("--class", dest="object_class", default="generic",
                   help="object class (galaxy|emission|reflection|planetary|globular|open|generic)")
    p.add_argument("--identify", action="store_true",
                   help="plate-solve + catalog lookup to auto-set the object class")
    p.add_argument("--preview", action="store_true", help="fast preview (AI wall off) — the portable subset")
    p.add_argument("--palette", default="", help="palette, e.g. 'SHO (Hubble)', 'HOO', 'HOS'")
    p.add_argument("--ha", help="mono Ha master (narrowband combine)")
    p.add_argument("--oiii", help="mono OIII master")
    p.add_argument("--sii", help="mono SII master (optional)")
    p.add_argument("--bit-depth", type=int, default=16, choices=(8, 16), help="TIFF/PNG output depth")
    p.add_argument("--astap-path", help="path to the ASTAP executable (for --identify solving)")
    p.add_argument("--starnet-path", help="path to StarNet (else $LAZYSTRETCH_STARNET or PATH)")
    p.add_argument("--graxpert-path", help="path to GraXpert (else $LAZYSTRETCH_GRAXPERT or the .app)")
    p.add_argument("--deepsnr-path", help="path to DeepSNR (else $LAZYSTRETCH_DEEPSNR or PATH)")
    p.add_argument("--rcastro-path", help="path to the RC-Astro CLI 'rc-astro' binary "
                   "(else $LAZYSTRETCH_RCASTRO or PATH); enables real BlurX/StarX/NoiseX")
    p.add_argument("--no-gpu", action="store_true", help="disable GPU for external tools")
    p.add_argument("--deconv", action="store_true",
                   help="classical Richardson-Lucy deconvolution (weak BlurX substitute)")
    p.add_argument("--tools", action="store_true", help="print detected external tools and exit")
    p.add_argument("-q", "--quiet", action="store_true", help="don't echo the pipeline log")

    # dials (default None so a curated preset is not clobbered by an unset dial)
    p.add_argument("--sat", type=float, dest="satAdj")
    p.add_argument("--bright", type=float, dest="brightAdj")
    p.add_argument("--bg", type=float, dest="bgAdj")
    p.add_argument("--black", type=float, dest="blackAdj")
    p.add_argument("--contrast", type=float, dest="contrastAdj")
    p.add_argument("--crop", type=float, dest="cropPercent", help="manual crop %% (implies --no-auto-assess)")
    p.add_argument("--dehaze", type=float, dest="dehaze")
    p.add_argument("--gradient-cleanup", type=float, dest="gradientCleanup")
    p.add_argument("--chroma-nr", type=float, dest="chromaNR", help="chroma noise reduction (0..1)")
    p.add_argument("--stars", type=float, dest="starsAdj", help="star-reduction nudge (-0.3..0.5)")
    p.add_argument("--deepen", type=float, dest="deepen",
                   help="2nd-pass additional stretch 0..1 (for --input-stretched results)")
    p.add_argument("--highlights", type=float, dest="highlights",
                   help="highlights dial 0..1 (0=dialed down/dim cores+stars, 1=bright/untouched)")
    p.add_argument("--transparency", type=float, dest="transparency",
                   help="core-transparency dial 0..1 (large-scale local contrast in the bright core)")
    p.add_argument("--dim-core", type=float, dest="dimCore",
                   help="dim-core dial 0..1 (mask the large bright veil + lower its luminosity)")
    p.add_argument("--star-shrink", type=float, dest="starShrink",
                   help="software star reduction 0..1 (thin the star carpet; no StarNet)")
    p.add_argument("--snr-protect", type=float, dest="snrProtect",
                   help="SNR-protect 0..1 using a LazyStack noise map (protect signal in NR, "
                        "damp local contrast in pure-noise regions)")
    p.add_argument("--meteor-strength", type=float, dest="meteorStrength",
                   help="composite preserved meteor trails 0..1 (needs a LazyStack meteor layer)")
    p.add_argument("--structure", type=float, dest="structure",
                   help="mid-scale (arm/dust) structure enhancement 0..~1 (galaxy default 0.6; 0=off)")
    p.add_argument("--deepen-bg", type=float, dest="deepenBackground",
                   help="de-veil 0..1: colour-preserving deepen of a milky background floor (0=off)")
    p.add_argument("--nightscape-brightness", type=float, dest="nightscapeBrightness",
                   help="nightscape foreground brightness 0..1 (needs a LazyStack nightscape master)")
    p.add_argument("--analyze", action="store_true", help="print frame analysis + recommendations and exit")

    # toggles (store the opposite of the default where useful)
    p.add_argument("--no-auto-assess", action="store_true")
    p.add_argument("--no-bg-extract", action="store_true")
    p.add_argument("--gc", action="store_true", help="use GradientCorrection (ABE-poly fallback)")
    p.add_argument("--no-color-cal", action="store_true")
    p.add_argument("--no-scnr", action="store_true")
    p.add_argument("--no-local-contrast", action="store_true")
    p.add_argument("--hdr", dest="hdr", action="store_true", default=None)
    p.add_argument("--no-hdr", dest="hdr", action="store_false")
    p.add_argument("--reduce-cast", action="store_true")
    p.add_argument("--enhance-emission", action="store_true")
    p.add_argument("--no-develop-foreground", action="store_true",
                   help="nightscape: composite the foreground as-is (it's already developed)")
    p.add_argument("--dark-lane", action="store_true", help="dark-lane gradient model (no-empty-sky fields)")
    p.add_argument("--remove-stars", action="store_true", help="full star removal (starless + <out>_stars)")
    p.add_argument("--no-preset", action="store_true", help="don't apply the curated recipe on --identify")
    p.add_argument("--load-recipe", help="apply a .lsrecipe file (before CLI dial/toggle overrides)")
    p.add_argument("--save-recipe", help="write the resolved settings to a .lsrecipe file and exit")
    p.add_argument("--input-stretched", action="store_true", help="polish-only mode (non-linear input)")
    p.add_argument("--debug-background", action="store_true",
                   help="save the estimated gradient/background model to <out>_background.tif")
    return p


def _identify(args, loaded: Optional[LoadedImage]):
    """Return (object_class, solve_result, object_id) from a solve, or (--class, None, None)."""
    data = get_data()
    if not (args.identify and loaded is not None):
        return args.object_class, None, None
    res = solve(loaded, astap_path=args.astap_path)
    if not res.solved:
        print("Plate solve did not succeed — using --class")
        return args.object_class, res, None
    cands = get_catalog().find_in_field(res.ra, res.dec, res.fov_radius_deg)
    if not cands:
        print("Solved, but no catalog object in field — using --class")
        return args.object_class, res, None
    top = cands[0].obj
    cls = classify_object(top, data)
    print(f"Identified: {object_label(top)}  ->  class '{cls}' "
          f"(solved {'from header WCS' if res.fromExisting else 'with ASTAP'})")
    return cls, res, top.id


def _make_params(args, object_class: str, preset: Optional[dict] = None) -> Parameters:
    """Build params: class defaults -> curated preset -> explicit CLI overrides."""
    p = Parameters.for_object(object_class)
    if preset:
        apply_preset(p, preset)                      # curated > class defaults
    if getattr(args, "load_recipe", None):
        apply_recipe(p, load_recipe(args.load_recipe), get_data())   # recipe > curated

    # explicit CLI dials (None == unset, so a preset/recipe value survives)
    for name in ("satAdj", "brightAdj", "bgAdj", "blackAdj", "contrastAdj",
                 "dehaze", "gradientCleanup", "chromaNR", "starsAdj", "deepen", "highlights",
                 "transparency", "dimCore", "starShrink", "snrProtect", "meteorStrength", "structure",
                 "deepenBackground", "nightscapeBrightness"):
        v = getattr(args, name)
        if v is not None:
            setattr(p, name, v)
    if args.cropPercent is not None:
        p.cropPercent = args.cropPercent
        p.autoAssess = False
    if args.no_auto_assess:
        p.autoAssess = False
    if args.no_bg_extract:
        p.doBgExtract = False
    if args.gc:
        p.useGradientCorrection = True
    if args.no_color_cal:
        p.doColorCal = False
    if args.no_scnr:
        p.doSCNR = False
    if args.no_local_contrast:
        p.doLocalContrast = False
    if args.hdr is not None:
        p.doHDR = args.hdr
    if args.reduce_cast:
        p.reduceCast = True
    if args.enhance_emission:
        p.enhanceEmission = True
    if args.no_develop_foreground:
        p.developForeground = False
    if args.dark_lane:
        p.darkLaneGC = True
    if args.remove_stars:
        p.removeStars = True
    if args.input_stretched:
        p.inputStretched = True
    if getattr(args, "debug_background", False):
        p.debugBackground = True
    if args.deconv:
        p.useClassicalDeconv = True
    if args.palette:
        p.palette = args.palette
    return p


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv)

    tools = Tools.resolve(graxpert_path=args.graxpert_path, starnet_path=args.starnet_path,
                          deepsnr_path=args.deepsnr_path, rcastro_path=args.rcastro_path,
                          gpu=not args.no_gpu)
    if args.tools:
        print("Detected external tools:")
        for name, ok in tools.status().items():
            print(f"  [{'x' if ok else ' '}] {name}")
        return 0

    ha = oiii = sii = None
    if args.ha or args.oiii or args.sii:
        if not (args.ha and args.oiii):
            print("error: narrowband combine needs at least --ha and --oiii", file=sys.stderr)
            return 2
        ha, oiii = _mono(args.ha), _mono(args.oiii)
        sii = _mono(args.sii) if args.sii else None

    loaded = None
    image = None
    if args.input:
        loaded = load_image(args.input)
        image = loaded.data

    if image is None and ha is None:
        print("error: provide an input image or --ha/--oiii narrowband channels", file=sys.stderr)
        return 2

    object_class, solve_result, object_id = _identify(args, loaded)
    preset = None
    if object_id and not args.no_preset:
        rec = curated_for(object_id)
        if rec:
            preset = rec["settings"]
            print(f"Curated recipe: {rec['name']} — {rec['why']}")
    params = _make_params(args, object_class, preset)
    params.ha, params.oiii, params.sii = ha, oiii, sii
    if args.input and params.snrProtect > 0:                 # SNR-protect needs the master's noise map
        from .processes.snrmask import load_coverage_map, load_noise_map
        params.snr_noise_map = load_noise_map(args.input)
        params.snr_coverage_map = load_coverage_map(args.input)
        if params.snr_noise_map is None:
            print("note: --snr-protect set but no LazyStack noise map found next to the input "
                  "(lazystack_master_noise.npy) — SNR-protect will be skipped")
    if args.input and getattr(params, "meteorStrength", 0.0) > 0:   # composite preserved meteor trails
        from .lazystack.meteors import load_meteor_layer
        params.meteor_layer = load_meteor_layer(args.input)

    if args.save_recipe:
        save_recipe(args.save_recipe, params, get_data())
        print(f"Wrote recipe {args.save_recipe}")
        return 0

    if args.analyze:
        from .objects.analyze import analyze_view
        if image is None:
            print("--analyze needs an input image.", file=sys.stderr)
            return 2
        res = analyze_view(image, params, get_data())
        print("\n".join(res.lines))
        if res.recommendations:
            print("\nRecommendations (apply with the matching flags):")
            for r in res.recommendations:
                print(f"  {r['key']} = {r['val']}  — {r['label']}")
        return 0

    if not args.quiet:
        active = [n for n, ok in tools.status().items() if ok]
        print("External tools: " + (", ".join(active) if active else "none detected "
              "(GraXpert->MMT, StarNet->skip, SPCC->BN+CC)"))

    def _progress(i, total, name):
        print(f"[{i}/{total}] {name}")

    result = run_pipeline(image, params, preview=args.preview, tools=tools,
                          solve_result=solve_result,
                          progress=None if args.quiet else _progress)

    if not args.quiet:
        for line in result.log:
            if line.startswith(("   ", "--", "Auto", "Combined", "Identified")):
                print("  " + line.strip())

    # output path
    if args.output:
        out_path = args.output
    else:
        stem = Path(args.input).stem if args.input else "narrowband"
        out_path = str(Path(args.input).with_name(f"{stem}_LazyStretch.tif")) if args.input \
            else f"{stem}_LazyStretch.tif"

    save_image(out_path, result.image, bit_depth=args.bit_depth)
    print(f"\nWrote {out_path}  ({result.image.shape}, class '{result.object_class}', "
          f"{len(result.steps_run)} steps run, {len(result.steps_skipped)} skipped)")
    if result.stars_layer is not None:
        stars_path = str(Path(out_path).with_name(Path(out_path).stem + "_stars" + Path(out_path).suffix))
        save_image(stars_path, result.stars_layer, bit_depth=args.bit_depth)
        print(f"Stars layer -> {stars_path}")
    if result.background_model is not None:
        import numpy as _np
        bg = result.background_model
        bg = (bg - float(_np.min(bg))) / (float(_np.ptp(bg)) or 1.0)   # normalize for viewing
        bg_path = str(Path(out_path).with_name(Path(out_path).stem + "_background" + Path(out_path).suffix))
        save_image(bg_path, bg, bit_depth=args.bit_depth)
        print(f"Background model -> {bg_path}  (inspect: it should NOT show galactic structure)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
