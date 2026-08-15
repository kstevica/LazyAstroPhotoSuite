# Building LazyAstroPhotoSuite as a standalone macOS app

`build/build_macos.sh` compiles the whole suite into a self-contained
`LazyAstroPhotoSuite.app` with [Nuitka](https://nuitka.net) — no Python install
required on the target machine.

## One-time setup

Nuitka's macOS **standalone** mode refuses Apple's system Python (`/usr/bin/python3` — it's
tied to OS releases). Build from a **non-Apple CPython** (Homebrew or python.org). Make a
dedicated build venv so the everyday `.venv` is untouched:

```bash
/opt/homebrew/bin/python3.11 -m venv .venv-build
.venv-build/bin/python -m pip install --upgrade pip
.venv-build/bin/python -m pip install -e ".[gui,video,bg,tools,build]"
```

That pulls Nuitka, PySide6, imageio-ffmpeg (bundled ffmpeg), photutils and imagecodecs.
`build/build_macos.sh` auto-detects `.venv-build/` (override with `PYTHON=…`). Python 3.11 is
the safe target — every dependency has arm64 3.11 wheels.

## Build

```bash
./build/build_macos.sh
```

- Output: `build/dist/LazyAstroPhotoSuite.app` (gitignored).
- First compile is ~10–30 min; later builds are faster (Nuitka caches).
- The `.app` is **arm64**, matching this machine's Python. Intel/universal2 would need a
  universal Python + universal wheels for every dependency (not set up here).
- **Minimum macOS**: the bundle inherits its deployment target from the build Python.
  Homebrew's 3.11 targets the *current* SDK, so this build requires **macOS 26+** (check with
  `otool -l …/MacOS/LazyAstroPhotoSuite | grep -A2 LC_BUILD_VERSION`). To support older macOS,
  build against a **python.org** CPython (its universal2 installer targets macOS 11+) and set
  `MACOSX_DEPLOYMENT_TARGET=11.0` before running the script.

## What ends up inside

Bundled automatically: the Python runtime, numpy/scipy/astropy/photutils/astroalign/rawpy,
PySide6 + the Qt **cocoa** platform plugin and **JPEG imageformats** plugin (the launcher
card art is `.jpg`), and — via `--include-data-dir` — the launcher assets, `lazystretch_data.json`,
and the DSO catalog CSVs. The macOS app menu reads **LazyAstroPhotoSuite** from the bundle's
`CFBundleName` (no runtime hack needed in the `.app`).

**ffmpeg** (LazyFlight video export) is bundled from `imageio-ffmpeg`. At runtime the app
prefers a system `ffmpeg` on `PATH`, then falls back to the bundled binary — so export works
even on a machine with no ffmpeg installed.

**Not bundled, by design** — the external "AI wall" tools are feature-detected and stay
external; the app degrades gracefully without them. Users install these separately if wanted:

- RC-Astro CLI (`rc-astro`) — BlurX / StarX / NoiseX
- StarNet, GraXpert, DeepSNR — open-source substitutes
- ASTAP — plate solving

## Optional app icon

Drop an `build/appicon.icns` and the script picks it up (`--macos-app-icon`). Without one,
Nuitka uses a generic icon.

## Distributing to other Macs (signing + notarization)

An unsigned `.app` runs locally but other users hit Gatekeeper. To distribute:

1. **Sign** (Developer ID Application cert; sign nested binaries too — the bundled ffmpeg
   and Qt/dylibs):
   ```bash
   codesign --deep --force --options runtime --timestamp \
     --sign "Developer ID Application: <you> (<TEAMID>)" \
     build/dist/LazyAstroPhotoSuite.app
   ```
2. **Notarize** (Apple Developer account required):
   ```bash
   ditto -c -k --keepParent build/dist/LazyAstroPhotoSuite.app LAPS.zip
   xcrun notarytool submit LAPS.zip --apple-id <id> --team-id <TEAMID> --wait
   xcrun stapler staple build/dist/LazyAstroPhotoSuite.app
   ```

For local testing without signing, strip the quarantine flag after copying the app:
```bash
xattr -dr com.apple.quarantine build/dist/LazyAstroPhotoSuite.app
```

## Notes / gotchas handled

- **multiprocessing** — LazyFlight's parallel frame render uses a process pool. `main()`
  calls `multiprocessing.freeze_support()` so a spawned worker runs as a worker, not a second
  app. (`parallel_frames` also falls back to sequential if the pool can't start.)
- **Data files** — loaded via `Path(__file__)`; the two data subdirs have no `__init__.py`,
  so they're included with explicit `--include-data-dir` rather than package-data.
