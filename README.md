# LazyAstroPhotoSuite (LAPS)

**An end-to-end astrophotography workflow in one native desktop app — from raw subs to a
finished image, and beyond.** Stack your lights into a clean master, stretch it automatically
with an object-aware pipeline, hand-finish it in a Lightroom-style darkroom, turn the still
into a 3-D fly-through video, and process Sun & Moon lucky-imaging bursts — all offline, no
subscriptions.

- **Website:** [kstevica.com/laps](https://kstevica.com/laps)
- **Source:** [github.com/kstevica/LazyAstroPhotoSuite](https://github.com/kstevica/LazyAstroPhotoSuite)
- **Contact:** kstevica@gmail.com
- Native **macOS** (Apple Silicon) and **Windows** apps
- **Source-available** — free for any non-commercial use (see [License](#license))

---

## Download

Ready-to-run builds for the latest release:

- **macOS (Apple Silicon)** — [LazyAstroPhotoSuite-macos-arm64.zip](https://github.com/kstevica/LazyAstroPhotoSuite/releases/latest/download/LazyAstroPhotoSuite-macos-arm64.zip)
  → unzip, drag `LazyAstroPhotoSuite.app` to Applications. First launch is unsigned, so
  right-click → **Open** (or `xattr -dr com.apple.quarantine LazyAstroPhotoSuite.app`).
- **Windows (x64)** — [LazyAstroPhotoSuite-windows-x64.zip](https://github.com/kstevica/LazyAstroPhotoSuite/releases/latest/download/LazyAstroPhotoSuite-windows-x64.zip)
  → unzip the folder, run `LazyAstroPhotoSuite.exe`.

No Python or dependencies to install — the app is self-contained. All builds are on the
[Releases page](https://github.com/kstevica/LazyAstroPhotoSuite/releases).

> New to it? Start with the **[Usage & Tutorial guide](USAGE.md)**.

---

## The suite

LAPS is a launcher that frames the whole pipeline — **Build the master → Process the image**,
with **Sun & Moon** and **Fly-through** alongside:

| Tool | What it does |
|------|--------------|
| **LazyStack** | Calibrate, register and integrate a folder of subs into a clean master (edge-crop, coverage map, per-pixel noise/SNR map, meteor detection). Optional **experimental Amplified-signal mode**: keep soft frames via frequency-split weights, a measured photon-transfer noise model, dither-validated pattern removal, and a 2× fine grid when undersampled. |
| **LazyStretch** | Automated, statistics-driven, **object-aware** stretching + finishing for deep-sky masters — auto-classifies the target, picks sensible defaults, and keeps every value live and adjustable. Includes a **Significance stretch** that holds each region's brightness to what its measured SNR statistically supports. |
| **LazyDevelop** | Hand-finish a stretched master: curves, levels, colour, selective colour, wavelet clarity, HDR, crop (drag to refine), deconvolution, star reduction, **star diffraction spikes**, and one-click XTerminator (BlurX / StarX / NoiseX) with graceful open-source fallbacks. Open any run straight from LazyStretch's history. |
| **LazyNightscape** | Foreground-locked Milky Way: stack a fixed-tripod sky on the sky stars only and keep a sharp foreground (paintable horizon segmentation), composited over the deep sky in LazyStretch. |
| **LazyFlight** | Turn a finished still into a faithful 3-D **fly-through video** — parallax + volumetric glow, stars that fly toward the camera, per-star spike/streak control, up to 3840-px and 150 Mbps. |
| **LazyMoonSun** | Lucky-imaging burst stacking + finishing for the **Sun and Moon** (global + multi-point alignment, auto-classify, Sun/Moon presets). |

Everything is designed around one idea: **get you to a great result with minimal effort**,
while keeping every knob within reach when you want control.

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

## Build from source

The code is public — clone it, run it, hack on it:

```bash
git clone https://github.com/kstevica/LazyAstroPhotoSuite.git
cd LazyAstroPhotoSuite
python -m pip install -e ".[gui,video,bg,tools]"
python -m lazystretch.gui.app          # launch the suite
python -m pytest -q                    # run the test suite
```

Packaging the standalone macOS `.app` (Apple Silicon) / Windows `.exe` is documented in
[`build/README.md`](build/README.md) — PyInstaller, with GitHub Actions workflows that build
both and attach them to a Release on every `v*` tag.

---

## License

LazyAstroPhotoSuite is **source-available** under the
[PolyForm Noncommercial License 1.0.0](LICENSE): free to use, study, modify and share for any
**non-commercial** purpose. **Commercial use requires a separate licence from the author** —
if you want to use LAPS commercially, contact **kstevica@gmail.com**.

(This is not an OSI "open source" licence, because it restricts commercial use — but the full
source is public and free for hobbyists, students, researchers and non-profits.)

---

## About

LazyAstroPhotoSuite is built by **Stevica Kuharski**.
Questions, bugs, feature ideas → **kstevica@gmail.com** · **[kstevica.com/laps](https://kstevica.com/laps)**
· **[GitHub](https://github.com/kstevica/LazyAstroPhotoSuite)**
