# LazyAstroPhotoSuite — Usage & Tutorial

A friendly walkthrough of the whole workflow, tool by tool. If you just want to get a
finished image quickly: **LazyStack → LazyStretch → LazyDevelop**. Everything runs offline.

- Website: [kstevica.com/laps](https://kstevica.com/laps) · Contact: **kstevica@gmail.com**

---

## Getting started

Open **LazyAstroPhotoSuite**. You land on the **launcher**, which lays out the workflow as a
pipeline:

```
Build the master           Process the image
  LazyStack        →   LazyStretch → LazyDevelop → LazyFlight
  LazyMoonSun  (Sun & Moon, alongside)
```

Click any card to open that tool. The **Tools ▸ Home** menu (or ⇧⌘H) always brings you back
to the launcher.

---

## 1 · LazyStack — build a clean master

Turn a folder of light subs into a single integrated master.

1. Open **LazyStack**.
2. **Pick your subs** (a folder of calibrated or raw lights). Add flats/darks/bias if you have
   them.
3. LazyStack **calibrates, registers and integrates** them, rejecting outliers (satellite
   trails, cosmic rays). It also:
   - **edge-crops** the stack border where frames don't all overlap,
   - builds a **coverage map** (how many frames support each pixel),
   - measures a **per-pixel noise / SNR map** (used later to protect faint signal),
   - detects **meteors** as rejected transients (you can composite them back in later).
4. **Save the master** (FITS/TIFF). That file is your input to LazyStretch.

Tip: the noise/SNR and coverage companions travel with the master and make the stretch smarter
about where to push detail vs. protect the background.

**Experimental — Amplified signal.** Tick this option to stop *culling* imperfect frames and
instead keep their photons: soft frames feed the faint structure at low spatial frequencies
while only sharp frames draw the detail, integration is inverse-variance weighted, a noise
model is measured from your own frames, sensor-fixed patterns the dither can prove are static
are removed, and undersampled dithered sets are drizzled onto a 2× grid. It writes its receipts
to `lazystack/amplified_meta.json`. Best on mixed-quality nights; on a uniform set it does no
harm. Needs "Stage to disk" on for the fine grid.

---

## 2 · LazyStretch — automatic, object-aware stretch

The heart of the suite: a statistics-driven stretch that adapts to *what* you're imaging.

1. Open **LazyStretch** → **Open master…** and pick your integrated master.
2. **Identify the target** (optional): click **Identify Target** to plate-solve + name the
   object, or just choose an **Object class** (galaxy, emission, reflection, cluster, Milky Way,
   …). The class sets sensible defaults.
3. Click **Preview** to see the auto-stretch on a fast, deterministic pass.
4. Fine-tune on the **Setup / Adjust / Process** tabs — saturation, deepen, HDR core, black
   point, dehaze, star handling, and more. Every value is live; the **Process** tab shows the
   full computed pipeline and lets you pin overrides.
5. Click **Execute** for the full-resolution result, then **Save Result…**.
6. **Save log…** writes the whole run to a text file (or right-click the log). You can also
   save a **recipe** and re-apply the same settings to another master.

**Significance stretch** (needs a LazyStack master with a noise map). This dial holds each
region's displayed brightness to what its **measured SNR** statistically supports: sub-proof
pixels ease toward the sky floor while structure the stack *proved* keeps the full stretch, so
faint real nebulosity stays up and noise stops masquerading as signal. It pairs naturally with
an Amplified-signal master. The ledger shows (and lets you pin) the σ thresholds.

**History → Develop.** Every run is saved as a 16-bit TIFF in the history list. Select one and
click **Develop ▸** to open it straight in LazyDevelop, or **Continue from image ▸** to keep
processing on top of it. Every dial and option is saved with the run, so reloading a history
entry restores its exact settings.

---

## 3 · LazyDevelop — the darkroom

Hand-finish a stretched master, Lightroom-style, with full history and masks.

1. Open **LazyDevelop** → open a stretched image.
2. Pick a tool from the **palette** on the left, grouped by category:
   - **Geometry** — Crop, Edge crop, Rotate/flip. *(Crop: draw the box, then drag its
     edges/corners to resize or drag inside to reposition.)*
   - **Tone** — Curves, Levels, S-curve, Local contrast, HDR core, Highlight roll-off, Deepen.
   - **Color** — Saturation, White balance, SCNR, Chroma NR, Background neutralize, Reduce cast,
     Enhance emission.
   - **Detail** — Mid-scale structure, Noise reduction, Reduce stars, Deconvolution, Halo Tamer.
   - **XTerminator** — **BlurX / NoiseX / StarX**: uses the real RC-Astro tools if installed,
     otherwise open-source equivalents.
   - **Background** — Background extraction, Gradient cleanup, Dehaze, De-veil.
   - **Studio** — Selective Color, Wavelet clarity, and **Star spikes**.
3. Adjust the tool's controls (they preview live), optionally gate it through a **mask**
   (luminance / highlights / range / painted / one-click semantic auto-masks), then click
   **Apply** to add it as a history step. In the mask list you can **arrow-key** through masks
   to preview each one (click a shown mask to return to the image).
4. Use **Undo/Redo**, click any history step to edit it in place, and **Save recipe** to reuse
   the whole edit stack.

### Star spikes (Studio)

Add diffraction spikes to stars:

1. Open **Star spikes** — it **auto-detects** the brightest stars and marks them.
2. **Click a star** to select it, **click empty canvas** to add one, **drag** to move,
   **right-click** to remove.
3. Set the number of spikes (3–32), rotation, thickness, intensity. The **Length** slider sets
   the *selected* star's spike length (per star).
4. Optional looks: **Chromatic fringe** (refractor-style colour toward the tips), **Arm-length
   jitter**, and **Per-arm colour**. Click **Apply** to composite.

---

## 4 · LazyFlight — 3-D fly-through video

Turn a finished still into a gentle fly-through.

1. Open **LazyFlight** → open your finished image.
2. Set the **pan points** (up to 5) the camera eases through, each with its own **zoom** and
   **rotation** (Z / X / Y). Drag the numbered points on the image; the background moves within
   the viewport as stars fly toward the camera.
3. Choose **spikes/streaks**, star sizes, orientation (landscape/portrait), aspect and duration
   (up to 120 s).
4. Pick a **quality preset** or a manual **bitrate** (up to 150 Mbps) and **Render video**.
5. **Save recipe** to reuse a camera move on another image.

*Note: zoom **in** streams the stars toward you; zoom **out** makes them recede.*

---

## 5 · LazyMoonSun — Sun & Moon

Lucky-imaging burst stacking + finishing for the Sun and Moon.

1. Open **LazyMoonSun** → pick a **burst folder** (a run of frames).
2. Choose a **Sun**, **Moon**, or **Neutral** base preset (it can also auto-classify).
3. **Stack** — global + multi-point alignment picks and combines the sharpest detail, then
   finishes it. Save the result.

---

## 6 · LazyNightscape — foreground-locked Milky Way

Stack a fixed-tripod Milky Way while keeping a sharp foreground.

1. Open **LazyNightscape** → pick your folder of sky frames.
2. Click **Preview segmentation** to see where LazyNightscape splits **sky** from
   **foreground** (the horizon). Nudge the **Sky ↔ foreground bias**, or **paint** Sky/Earth
   strokes to refine it — the brush snaps to the horizon.
3. Optionally pick your own **Foreground image…** (a separate sharp exposure); otherwise the
   sharpest frame is used.
4. **Stack** — it registers on the **sky stars only** (the foreground is masked so it doesn't
   confuse alignment), integrates the sky, and saves the sharp foreground + sky mask as
   companions.
5. Open the resulting master in **LazyStretch** — the foreground is composited over the
   stretched deep sky automatically (a "Nightscape foreground" dial controls it).

---

## Optional: connect your AI tools

LAPS auto-detects these if present and prefers them; otherwise it uses built-in fallbacks:

- **RC-Astro CLI** (`rc-astro`) → BlurX / StarX / NoiseX. Put `rc-astro` on your `PATH`, or set
  `LAZYSTRETCH_RCASTRO=/path/to/rc-astro`.
- **StarNet++ / StarNet2**, **GraXpert**, **ASTAP** — see the README table for env vars / flags.

Nothing is required — the suite is complete on its own, and bundles its own ffmpeg for video.

---

## Handy extras

- **Save log…** on every tool writes the run/edit log to a text file (or right-click the log).
- **Recipes** save a tool's settings so you can re-apply them to the next image.
- The macOS app menu and window titles show **LazyAstroPhotoSuite**; each tool keeps its own
  name so you always know where you are.

---

## Questions & feedback

Found a bug, or have an idea? → **kstevica@gmail.com** · **[kstevica.com/laps](https://kstevica.com/laps)**
· **[GitHub](https://github.com/kstevica/LazyAstroPhotoSuite)**

LAPS is **source-available** and free for non-commercial use (PolyForm Noncommercial 1.0.0).
For commercial use, contact **kstevica@gmail.com**.
