# Building LazyAstroPhotoSuite as a standalone macOS app

`build/build_macos.sh` compiles the whole suite into a self-contained
`LazyAstroPhotoSuite.app` with [PyInstaller](https://pyinstaller.org) — no Python install
required on the target machine.

## Why PyInstaller (not Nuitka)

Nuitka builds a working bundle right up until astropy's **unit parser**: astropy builds its
unit grammar at runtime with PLY, which needs real module docstrings, runtime frame
introspection, and a writable table directory. Nuitka's C-compilation strips/relocates all
of that, so unit strings fail with *"'m / (s)' did not parse as unit: Unable to build
parser"*. PyInstaller keeps modules as **bytecode** and ships a mature **astropy hook**, so
PLY (and scipy/skimage/photutils/rawpy) work exactly as they do unfrozen.

## One-time setup

Use a dedicated build venv so the everyday `.venv` is untouched. A non-Apple CPython
(Homebrew/python.org) also gives a cleaner bundle:

```bash
/opt/homebrew/bin/python3.11 -m venv .venv-build
.venv-build/bin/python -m pip install --upgrade pip
.venv-build/bin/python -m pip install -e ".[gui,video,bg,tools,build]"
```

The `build` extra pulls PyInstaller, imageio-ffmpeg (bundled ffmpeg), rawpy and astroalign.

## Build

```bash
./build/build_macos.sh
```

- Output: `build/dist/LazyAstroPhotoSuite.app` (gitignored).
- Builds in a few minutes; PyInstaller caches analysis between runs.
- The `.app` is **arm64**, matching the build Python.
- **Minimum macOS** follows the build Python's deployment target (Homebrew 3.11 → macOS 26+).
  For older macOS, build against a **python.org** CPython (universal2, targets macOS 11+).

## What ends up inside

PyInstaller's hooks bundle numpy/scipy/astropy/photutils/astroalign/rawpy/PySide6 (with the
Qt cocoa platform + JPEG imageformats plugins). `--add-data` ships the launcher assets +
`lazystretch_data.json` + the DSO catalog CSVs at their `Path(__file__)`-relative paths.
`--collect-all imageio_ffmpeg` bundles the ffmpeg binary for **offline LazyFlight export**
(the app still prefers a system `ffmpeg` on `PATH` first). The app-menu name comes from the
bundle's `CFBundleName` = **LazyAstroPhotoSuite**, with a themed icon (`build/make_icon.py`).

**Not bundled, by design** — the external "AI wall" tools stay external and feature-detected;
the app degrades gracefully without them. Install separately if wanted: the RC-Astro CLI
(`rc-astro` — BlurX/StarX/NoiseX), StarNet, GraXpert, DeepSNR, ASTAP.

## Distributing to other Macs (signing + notarization)

Unsigned, the `.app` runs locally but other users hit Gatekeeper. To distribute:

```bash
codesign --deep --force --options runtime --timestamp \
  --sign "Developer ID Application: <you> (<TEAMID>)" \
  build/dist/LazyAstroPhotoSuite.app
ditto -c -k --keepParent build/dist/LazyAstroPhotoSuite.app LAPS.zip
xcrun notarytool submit LAPS.zip --apple-id <id> --team-id <TEAMID> --wait
xcrun stapler staple build/dist/LazyAstroPhotoSuite.app
```

For local testing without signing, strip the quarantine flag after copying:
```bash
xattr -dr com.apple.quarantine build/dist/LazyAstroPhotoSuite.app
```

## Notes / gotchas handled

- **multiprocessing** — LazyFlight's parallel frame render uses a process pool. The entry
  point (`build/laps_entry.py`) calls `multiprocessing.freeze_support()` so a spawned worker
  runs as a worker, not a second app. (`parallel_frames` also falls back to sequential.)
- **Data files** — loaded via `Path(__file__)`; `--add-data` places them at the matching
  relative paths inside the bundle.
- **Icon** — `build/make_icon.py` renders `build/appicon.icns` (uses the star-spikes engine).
