# Changelog

All notable changes to **LazyStretchPy** (the standalone Python port of the LazyStretch
suite) are recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/);
the project is pre-1.0 (`0.1.0`), so everything below is under `[Unreleased]` until tagged.

## [Unreleased]

### Added
- **Star diffraction spikes (LazyDevelop).** A new "Star spikes" Studio tool auto-detects
  the brightest stars and marks them; click to select, click empty to add, drag to move,
  right-click to remove, with a live schematic overlay. Spike count (3–32), rotation,
  thickness and intensity are global; spike length is per-star (the Length slider sets the
  selected star). Spikes are tinted by each star's own colour and composited on Apply.
  Two opt-in looks (dials, 0 = off): **Chromatic fringe** blends a warm→cool spectral
  gradient into the outer arms (refractor-style diffraction colour), and **Arm-length
  jitter** varies each arm's length deterministically. The live overlay reflects the jitter.
- **BX / NX / SX tools in LazyDevelop.** A new "XTerminator" tool group adds
  BlurXTerminator, NoiseXTerminator and StarXTerminator (Reduce or Remove-to-starless).
  Each prefers the real RC-Astro product when the `rc-astro` CLI is installed and falls back
  to the open-source method otherwise (Richardson-Lucy, multiscale-median, StarNet or the
  morphological star reducer), so the same tool works on any machine.
- **Save / export logs.** Every run log can be written to a text file: a "Save log…" button
  in the Stretch, LazyMoonSun, and LazyDevelop windows, plus a right-click "Copy log" /
  "Save log…" menu on any log view (shared `LogExportMixin`). The CLI gains `--log-file
  <path>` to write the full run log alongside the output.
- **RC-Astro standalone CLI support** — when the `rc-astro` CLI (v1.1.x) is installed and
  licensed, the pipeline uses the *real* BlurXTerminator / StarXTerminator /
  NoiseXTerminator instead of the open-source substitutes. One feature-detected binary
  (`lazystretch/external/rcastro.py`), discovered via explicit path → `$LAZYSTRETCH_RCASTRO`
  → `PATH` → standard install dirs; new `--rcastro-path` CLI flag; the CLI/GUI tool-status
  lines list the three products when present. Preference per slot: deconvolution
  BlurX → classical Richardson-Lucy → skip; stars StarX → StarNet; each falls back in-step
  if a run fails (e.g. an inactive licence). The CLI argv was verified against the real
  `rc-astro --help`, not guessed.
- **LazyFlight — save/load fly-through recipes** (`.lfrecipe`). A recipe captures every v2
  setting (duration, zoom, Z/X/Y rotation, bloom, star count and min/max size, streaks and
  strength, fps, long edge, orientation, aspect, quality, bitrate) plus the full pan-point
  keyframe list and background-pan offset; loading rebuilds the engine and preview. Files
  whose kind isn't `lazyflight-v2` are refused.

### Changed
- **LazyDevelop crop is now draggable to refine.** Once the crop rectangle is drawn you can
  drag its edges/corners to resize, drag inside to reposition (clamped to the frame), or
  drag on empty area to draw a fresh one — with matching hover cursors. The BlurX tooltip
  now notes it sharpens in the nonlinear domain here (its ideal slot is the pipeline's
  linear deconvolution step).
- **NoiseXTerminator now runs in the linear NR slot** (pre-stretch, its native domain,
  PixInsight `js:3369`) instead of the post-stretch slot. The stretched-domain denoisers
  (DeepSNR → GraXpert → MMT) stay post-stretch and are skipped when NoiseX already ran;
  NoiseX is no longer in that chain. `inputStretched` inputs, which have no linear slot,
  fall to the post-stretch chain as before.
- **LazyFlight v2 — the star field now flies with the dolly.** Zooming *in* streams the
  stars toward the camera (as before); zooming *out* makes them **recede** (the opposite
  direction). The per-frame zoom velocity is turned into a smooth signed pace (`tanh`, so a
  dolly turn-around eases through zero), and the streak trails flip to match.

### Notes
- The test suite is hermetic: `tests/conftest.py` neutralises external-tool auto-detection
  (including `rc-astro`) so runs are deterministic even on a machine that has the real tools
  installed. Real-tool validation is done in one-off scripts.
