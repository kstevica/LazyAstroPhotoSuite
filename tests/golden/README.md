# Golden fixtures (Layer-1 parity)

These `*.expected.json` files are the deterministic parity contract between the Python
port and PixInsight LazyStretch (PLAN §9.3). The harness that consumes them is
`tests/test_golden.py`. Commit only these small JSON sidecars — **never** the astro
masters (`.gitignore` blocks `*.fits/*.fit/*.xisf/*.tif`).

## `kind: "mtf_vectors"` (shipped)

Closed-form MTF values + find-midtones round-trips, derived from the pinned formula
independently of `stats/mtf.py`. This locks the crown-jewel primitive today, with no
PixInsight required. Once `Math.mtf` is confirmed against a real PI install (PLAN §11),
extend it with PI-dumped `(m, x) -> y` values.

## `kind: "pi_telemetry"` (add later)

Statistics **measured by PixInsight** that the Python core must reproduce from the same
raw inputs. LazyStretch already logs the numbers it derives (`median/avgDev/c0/m` from
`applyAutoStretch`; `gradient/vmin/vmax/crop` from `autoAssess`; `lift/cleanSky/typical`
from `measureDust`; resolved `eff*`). Add a small **guarded telemetry mode** to
`LazyStretch.js` (env var / `#define`) that dumps these as a JSON sidecar per master,
then reshape into:

```json
{
  "kind": "pi_telemetry",
  "image": "M42_NGC1977",
  "atol": 1e-4,
  "stretch": [
    { "median": 0.0623, "avgDev": 0.0121, "shadowsClip": -1.05, "targetBkg": 0.25,
      "expect": { "c0": 0.0496, "m": 0.1873 } }
  ],
  "effective": [
    { "class": "emission", "sliders": { "satAdj": 0.1 },
      "expect": { "bkg": 0.25, "sat": 0.65, "clip": -1.05, "bgLevel": 0.20, "contrast": 0.12 } }
  ]
}
```

The harness feeds `stretch[*]` into `solve_stretch(median, avgDev, shadowsClip,
targetBkg)` and `effective[*]` into `resolve_effective(...)`, asserting the derived
numbers match within `atol` (default `1e-4`). Because these are fed the *same measured
raw stats*, they need no image at test time and run anywhere — no PixInsight, no masters.

## Workflow (per PLAN §9.4)

1. On the Mac with PI: run the telemetry export over the sample masters
   (`../../../LazyStretch/example/`), producing the sidecars.
2. Drop the `*.expected.json` here, tagged with the PI SHA they were captured at.
3. `.venv/bin/python -m pytest tests/golden` — runs everywhere, no PI needed.
