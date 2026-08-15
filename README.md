# LazyAstroPhotoSuite (LAPS)

**An end-to-end astrophotography workflow in one native desktop app — from raw subs to a
finished image, and beyond.** Stack your lights into a clean master, stretch it automatically
with an object-aware pipeline, hand-finish it in a Lightroom-style darkroom, turn the still
into a 3-D fly-through video, and process Sun & Moon lucky-imaging bursts — all offline, no
subscriptions.

- **Website:** [kstevica.com/laps](https://kstevica.com/laps)
- **Contact:** kstevica@gmail.com
- Native **macOS** and **Windows** apps

---

## The suite

LAPS is a launcher that frames the whole pipeline — **Build the master → Process the image**,
with **Sun & Moon** and **Fly-through** alongside:

| Tool | What it does |
|------|--------------|
| **LazyStack** | Calibrate, register and integrate a folder of subs into a clean master (edge-crop, coverage map, per-pixel noise/SNR map, meteor detection). |
| **LazyStretch** | Automated, statistics-driven, **object-aware** stretching + finishing for deep-sky masters — auto-classifies the target, picks sensible defaults, and keeps every value live and adjustable. |
| **LazyDevelop** | Hand-finish a stretched master: curves, levels, colour, selective colour, wavelet clarity, HDR, crop (drag to refine), deconvolution, star reduction, **star diffraction spikes**, and one-click XTerminator (BlurX / StarX / NoiseX) with graceful open-source fallbacks. |
| **LazyFlight** | Turn a finished still into a faithful 3-D **fly-through video** — parallax + volumetric glow, stars that fly toward the camera, per-star spike/streak control, up to 3840-px and 150 Mbps. |
| **LazyMoonSun** | Lucky-imaging burst stacking + finishing for the **Sun and Moon** (global + multi-point alignment, auto-classify, Sun/Moon presets). |

Everything is designed around one idea: **get you to a great result with minimal effort**,
while keeping every knob within reach when you want control.

---

## Install

Download the ready-to-run app from **[kstevica.com/laps](https://kstevica.com/laps)**:

- **macOS** — `LazyAstroPhotoSuite.app` (unzip, drag to Applications, open).
- **Windows** — `LazyAstroPhotoSuite.exe` (unzip the folder, run the `.exe`).

No Python or dependencies to install — the app is self-contained.

> New to it? Start with the **[Usage & Tutorial guide](USAGE.md)**.

---

## Optional: external AI tools

LAPS works fully on its own with built-in, portable algorithms. If you already use any of
these, LAPS **auto-detects** them and prefers them; otherwise it degrades gracefully:

| Tool | Used for | How LAPS finds it |
|------|----------|-------------------|
| **RC-Astro CLI** (`rc-astro`) | BlurXTerminator, StarXTerminator, NoiseXTerminator | on `PATH`, or `LAZYSTRETCH_RCASTRO` |
| **StarNet++ / StarNet2** | star removal / reduction | `--starnet-path`, or `LAZYSTRETCH_STARNET` |
| **GraXpert** | background / gradient + denoise | `--graxpert-path`, or `LAZYSTRETCH_GRAXPERT` |
| **ASTAP** | plate solving (Identify Target) | `--astap-path` |

None are bundled or required. Without them, LazyDevelop's BX / NX / SX fall back to
Richardson-Lucy deconvolution, multiscale-median denoise, and morphological star reduction.
For video export, LAPS bundles its own **ffmpeg**, so LazyFlight works out of the box.

---

## Command line (optional)

The stretch pipeline is also scriptable for batch work:

```bash
# process an RGB / OSC master; class chosen automatically or with --class
lazystretch master.fits --class galaxy --output result.tif

# also save the full run log next to the result
lazystretch master.fits --log-file run.log
```

Run `lazystretch --help` for the full flag list.

---

## Build from source (developers)

```bash
python -m pip install -e ".[gui,video,bg,tools]"
python -m lazystretch.gui.app          # launch the suite
python -m pytest -q                    # run the test suite
```

Packaging the standalone macOS `.app` / Windows `.exe` is documented in
[`build/README.md`](build/README.md) (PyInstaller + a GitHub Actions Windows workflow).

---

## About

LazyAstroPhotoSuite is built by **Stevica Kuharski**.
Questions, bugs, feature ideas → **kstevica@gmail.com** · **[kstevica.com/laps](https://kstevica.com/laps)**
