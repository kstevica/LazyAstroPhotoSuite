# Changelog

All notable changes to **LazyAstroPhotoSuite** are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/).

## [1.1.0] — 2026-08-17

First **source-available** release: the code is public on GitHub under the PolyForm
Noncommercial 1.0.0 licence (free for any non-commercial use; commercial use requires a
licence from the author — kstevica@gmail.com). Native macOS (Apple Silicon) and Windows
builds are published as GitHub Release assets.

### Added
- **LazyNightscape — its own tool.** The foreground-locked Milky Way workflow (sky-only
  registration + sharp-foreground companion) is now a dedicated launcher card and window,
  split out of LazyStack. The window shows only what a fixed-tripod sky stack needs
  (register-on-sky, normalization, crop, optional calibration) plus the foreground/sky
  segmentation and paint tools; nightscape mode is implied (no on/off checkbox). The plain
  LazyStack window is now deep-sky-only.
- **Develop a history image in one click (LazyStretch).** Each history run is already saved
  as a 16-bit TIFF; a new "Develop ▸" button beside "Continue from image ▸" opens that TIFF
  straight into LazyDevelop (the shell switches panels and loads it).
- **Keyboard preview in the Develop mask list.** Arrow-key navigation now previews the
  highlighted mask exactly like clicking it; a click on the already-shown mask still toggles
  back to the image (robust against the click-that-also-moves-selection double-event).

### Changed
- **History captures every dial and option.** An audit of all exposed controls found that
  `significance` plus `useClassicalDeconv`, `developForeground`, `inputStretched` and
  `debugBackground` were not persisted; history now round-trips **all** of them. Portable
  `.lsrecipe` files still exclude input-state / debug bools (`inputStretched`,
  `debugBackground`) so a recipe stays applicable on any input — only the on-disk history
  keeps them (`recipe_from_params(..., include_state=True)`).

### Added
- **Significance stretch (LazyStretch).** A new "Significance (needs stack)" dial
  (`--significance` in the CLI): the displayed brightness of every region is held to what
  its measured SNR statistically supports. The per-pixel significance
  S = (matched-filtered luminance − sky) / σ is computed in the linear domain from
  LazyStack's measured noise map (conservative: the PSF-scale detection kernel genuinely
  lowers σ, but that credit is never claimed), folded with frame-support coverage, and
  ramped 0→1 between 1.5σ and 4σ — both thresholds live in the PROC ledger and are
  pinnable. After the stretch, sub-proof pixels ease toward the stretched sky floor
  (hue preserved) while proven structure keeps the full stretch: faint real nebulosity
  stays up because the stack *proved* it, and noise stops masquerading as signal. Pairs
  naturally with the amplified stack's calibrated noise map; off by default, no-op
  without a noise companion.
- **Experimental — Amplified signal stacking (LazyStack).** One checkbox that changes what
  "stacking" means: instead of culling imperfect frames, it keeps their photons and uses
  physics to decide where each frame is trustworthy. Five measured mechanisms, all logged
  with evidence and receipted in `lazystack/amplified_meta.json`:
  1. *Soft frames kept* — bloated / bright-sky frames are no longer culled (only trailed
     frames still hard-reject); a frequency split feeds their full low band to the faint
     structure while their high band is down-weighted by their measured MTF.
  2. *Inverse-variance integration* — every frame weighted 1/σ² from its measured noise
     floor (the maximum-likelihood mean), replacing the flat SNR weight.
  3. *Photon-transfer noise model* — var = a·signal + b regressed from the registered
     frames themselves (no calibration data needed) as a lower-envelope fit over the
     sky-and-faint regime (χ²-debiased 25th-percentile bins, contaminated bins dropped),
     so star-halo jitter can't inflate it; used as a physical floor for the clip scale
     where the sample MAD collapses — capped below bright structure so trails over
     nebulosity still reject — and reported as an effective full-scale e⁻ figure.
  4. *Dither-validated static-pattern removal* — sensor-fixed structure that the moving
     sky can't explain is subtracted before registration, band-limited to spatial scales
     the dither can PROVE are static (highpass σ = dither/3), so extended nebulosity is
     untouchable by construction; star-safe via a temporal-MAD guard, soft-thresholded,
     never run in Nightscape mode (the land is static by design), written to separate
     staging so the reusable calibrated cache stays pristine, and hash-keyed into the
     registration cache. Skipped, with the reason logged, when the dither can't prove
     separation.
  5. *Fine-grid registration (drizzle-lite)* — undersampled dithered sets (FWHM < 2.5 px,
     ≥ 8 frames) are warped once, straight onto a 2× grid (`LZSGRID=2`), recovering the
     sub-pixel detail the dither actually sampled; never a second interpolation. Guarded:
     staged mode only, not with local normalization, and only with measured disk headroom
     for the 4× staging (the estimate and the decision are logged).
  The master is stamped `LZSAMP=1`; the noise/SNR companion reflects the new weighting, so
  the stretch's SNR-protect mask benefits immediately. No pixels are invented anywhere —
  every mechanism only re-weights, cleans, or re-grids photons that were captured.

## [1.0.0] — 2026-08-15

First public release of the suite: LazyStack, LazyStretch, LazyDevelop, LazyFlight and
LazyMoonSun, as native macOS and Windows apps.

### Added
- **Full-screen button in LazyDevelop** — shows the current canvas image full-screen
  (Esc to close), matching LazyStretch.

### Docs & branding
- **Product website** (`web/`, for kstevica.com/laps) — a one-page site with an animated
  starfield (meteors included), real in-app screenshots shot from the shipping build
  (`web/tools/shoot.py`), before/after comparison sliders, three LazyFlight demo clips,
  a masks deep-dive, and an option-by-option reference of every window (232 options,
  extracted from the source and audit-corrected; `web/tools/build_site.py`). Mobile
  friendly, no external dependencies.
- **README rewritten** as a standalone-product doc (LazyAstroPhotoSuite) and a new **USAGE.md**
  tutorial covering every tool. The website (**kstevica.com/laps**) and contact
  (**kstevica@gmail.com**) appear in the README, the tutorial, and the launcher footer.
- **LazyFlight launcher card** now uses an M45 (Pleiades) image.

### Packaging
- **Standalone macOS app build** (`build/build_macos.sh`, PyInstaller) — compiles the whole
  suite into a self-contained `LazyAstroPhotoSuite.app` (arm64) that needs no Python. The
  bundle's `CFBundleName` shows **LazyAstroPhotoSuite** in the app menu natively, carries a
  themed icon (`build/make_icon.py`), and includes the launcher art, data JSON, DSO catalogs,
  and an ffmpeg binary (via imageio-ffmpeg) so LazyFlight export works offline. External AI
  tools (RC-Astro, StarNet, GraXpert, ASTAP) stay unbundled and feature-detected. Freeze
  hardening: `multiprocessing.freeze_support()` in the entry point. PyInstaller (not Nuitka):
  Nuitka's C-compilation breaks astropy's runtime PLY unit-grammar build ("Unable to build
  parser"); PyInstaller keeps modules as bytecode + ships an astropy hook. A `build` extra
  pins the tooling; `build/README.md` documents setup, signing/notarization, and the
  deployment-target caveat.

### Added
- **Star diffraction spikes (LazyDevelop).** A new "Star spikes" Studio tool auto-detects
  the brightest stars and marks them; click to select, click empty to add, drag to move,
  right-click to remove, with a live schematic overlay. Spike count (3–32), rotation,
  thickness and intensity are global; spike length is per-star (the Length slider sets the
  selected star). Spikes are tinted by each star's own colour and composited on Apply.
  Three opt-in looks (dials, 0 = off): **Chromatic fringe** blends a warm→cool spectral
  gradient into the outer arms (refractor-style diffraction colour), **Arm-length jitter**
  varies each arm's length deterministically (the live overlay reflects it), and **Per-arm
  colour** gives each arm a slightly different hue (with a saturation floor so it shows even
  on near-white stars).
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
