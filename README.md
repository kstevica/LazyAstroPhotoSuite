# LazyStretchPy

A standalone, native-desktop **Python** port of the PixInsight **LazyStretch** script — the same
statistics-driven, object-aware stretching pipeline, running without PixInsight. No web version, no web
views. Ships as a pip package (primary) and a signed macOS `.app` (secondary).

> **Status:** P0–P5 complete — headless numerical core, image I/O, target identification, the full
> **AAA-free pipeline** (incl. masked step variants), a native PySide6 desktop GUI, **and external-tool
> integration** (StarNet, GraXpert, classical deconvolution; SPCC-lite scaffolded). Runs end-to-end on real
> masters via CLI *and* GUI, using the wall tools when installed and degrading gracefully when not.
> **Synced to PI v1.4.1** — the full **P7 catch-up is done** (Chroma NR, dark-lane gradient, Remove stars,
> `starsAdj`, dehaze-gradient, curated recipes, **Deepen second-pass**, and the **Advisor / Analyze-Frame**),
> all wired into the pipeline/CLI/GUI, adversarially verified, and tested (117 passing). The Python pipeline
> now tracks v1.4.1 feature parity. **P1 (golden calibration) is underway** — a harness (`calibration/`)
> compares the port against PixInsight's reference renders; first fix: the **HDR core** now reveals nebula-core
> structure (M42 Trapezium) instead of flattening it. Remaining: more P1 steps (saturation/LHE/noise),
> **SPCC-lite**, **P6** packaging. See **[PLAN.md](PLAN.md)**.

## Quick start (dev)

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev,gui]"
.venv/bin/python -m pytest -q          # 93 passing, 1 skipped (PI-telemetry placeholder)
lazystretch-gui                        # launch the desktop app
lazystretch --tools                    # show which external tools are detected
```

## External tools (optional — the "AI wall")

None are bundled. LazyStretch finds them by **explicit path → env var → PATH**, and degrades gracefully
when absent (GraXpert→MMT noise reduction, StarNet→skip, SPCC→BN+CC).

| Tool | Provides | Point LazyStretch at it |
|---|---|---|
| **StarNet++ / StarNet2** | star reduction, star-protected saturation | `--starnet-path /path/to/starnet++` or `export LAZYSTRETCH_STARNET=…` |
| **GraXpert** | noise reduction, background/gradient (GC) | `--graxpert-path /path/to/graxpert` or `export LAZYSTRETCH_GRAXPERT=…` |
| **ASTAP** | plate solving (`--identify`) | `--astap-path /path/to/astap` |
| classical **Richardson–Lucy** | weak BlurX substitute (no install) | `--deconv` (off by default) |
| **SPCC-lite** | Gaia color calibration | `pip install ".[spcc]"` — scaffolded; currently defers to BN+CC |

### CLI — run the pipeline on a master

```bash
# RGB / OSC master, class chosen manually
lazystretch master.fits --class galaxy -o out.tif

# auto-identify the target (existing WCS, or ASTAP if installed), then process
lazystretch osc.fits --identify -o out.tif

# mono narrowband SHO/HOO/HOS combine from three masters
lazystretch --ha Ha.fits --oiii OIII.fits --sii SII.fits --palette "SHO (Hubble)" \
            --class emission -o sho.tif

# fast preview (the deterministic subset — AI wall off), tweak the dials
lazystretch master.fits --class emission --preview --sat 0.1 --bright 0.03 -o preview.png
```

### Library — headless core

```python
import numpy as np
from lazystretch import get_data, resolve_effective, apply_auto_stretch
from lazystretch.io import load_image, save_image
from lazystretch.objects import Parameters
from lazystretch.pipeline.runcore import run_pipeline

img = load_image("master.fits").data                 # float [0,1], (H,W) or (H,W,3)
result = run_pipeline(img, Parameters.for_object("galaxy"), preview=True)
save_image("out.tif", result.image)
```

## What runs (the AAA-free "Preview" pipeline)

Crop → background/gradient (ABE poly / GC-fallback) → colour calibration (BN+CC) → auto-stretch →
SCNR → HDR core → background level (+ adaptive floor) → contrast → saturation → cast/emission/dehaze →
gradient cleanup → local contrast. The four **wall** steps (BlurX / NoiseX / StarX / SPCC) are
feature-detected and skipped when absent (NoiseX falls back to the portable MMT); they are integrated in
P5. Masked step variants are a P3 follow-up (they fall back to the global variant with a logged note).

## Layout

```
LazyStretchPy/
├── PLAN.md                     ← the architecture & porting plan (start here)
├── README.md                   ← this file
└── reference/
    └── pixinsight/             ← READ-ONLY source of truth (do not edit)
        ├── LazyStretch.js      ← PI LazyStretch v1.2.1, 3542 lines
        ├── README.md  PLAN.md  ← PI-side docs, copied verbatim
        └── SOURCE.md           ← exact version + SHA-1 + isolation date
```

## The core idea

The Python port keeps the **PixInsight `.js` as the source of truth** and is engineered to track it with
minimal effort as it keeps evolving. The seam is drawn along *change frequency*, not features:

- **Tunable numbers** (class profiles, thresholds, slider gains, palettes) live in **one shared JSON** that
  both the `.js` and Python read — so a number tweak propagates with zero re-porting.
- A **parity map** links each PI symbol to its Python module + the PI SHA it was last synced against.
- A **golden/regression harness** diffs Python against telemetry exported from PI on the sample images.
- A **per-release update checklist** turns "PI changed" into a mechanical procedure.

See PLAN.md §9 for the full sync strategy and §10 for the phased roadmap.

## Updating to a new PI version (short form)

1. Drop the new `LazyStretch.js` into `reference/pixinsight/` and update `SOURCE.md` (version + SHA-1).
2. `git diff` the `.js`; classify each hunk **DATA-ONLY** vs **STRUCTURAL**.
3. DATA-ONLY → edit the shared JSON (no Python change). STRUCTURAL → edit the mapped Python module.
4. Regenerate golden sidecars from PI, run `pytest tests/golden`, bump the synced SHA.

(Full 12-step checklist in PLAN.md §9.4.)
