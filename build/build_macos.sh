#!/usr/bin/env bash
#
# Build LazyAstroPhotoSuite as a standalone macOS .app with PyInstaller.
#
#   ./build/build_macos.sh
#
# Output:  build/dist/LazyAstroPhotoSuite.app  (self-contained; no Python needed)
#
# PyInstaller is used (not Nuitka) because it keeps modules as bytecode and ships mature
# hooks for the scientific stack. Astropy in particular builds its unit grammar at runtime
# with PLY (needs real docstrings + a writable table dir) — Nuitka's C-compilation breaks
# that ("Unable to build parser"); PyInstaller's astropy hook handles it. See build/README.md.
#
# Prereqs (once):  see build/README.md — a NON-system CPython venv (.venv-build) with
#   pip install -e ".[gui,video,bg,tools,build]" + rawpy + astroalign.
#
set -eo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-}"
if [[ -z "$PY" ]]; then
  if   [[ -x .venv-build/bin/python ]]; then PY=".venv-build/bin/python"
  else PY=".venv/bin/python"; fi
fi
APP="LazyAstroPhotoSuite"

ICON_ARG=()
[[ -f build/appicon.icns ]] && ICON_ARG=(--icon "${PWD}/build/appicon.icns")

echo ">> PyInstaller $("$PY" -m PyInstaller --version 2>/dev/null) building ${APP}"
echo ">> $(date)"

"$PY" -m PyInstaller \
  --name "${APP}" \
  --windowed \
  --noconfirm --clean \
  "${ICON_ARG[@]+"${ICON_ARG[@]}"}" \
  --osx-bundle-identifier "com.stevicakuharski.lazyastrophotosuite" \
  --distpath build/dist \
  --workpath build/pyi-work \
  --specpath build/pyi-spec \
  --add-data "${PWD}/lazystretch/gui/assets:lazystretch/gui/assets" \
  --add-data "${PWD}/lazystretch/data:lazystretch/data" \
  --collect-submodules lazystretch \
  --collect-all imageio_ffmpeg \
  --collect-data astropy \
  --collect-data photutils \
  --collect-data skimage \
  "${PWD}/build/laps_entry.py"

APP_PATH="build/dist/${APP}.app"
if [[ -d "$APP_PATH" ]]; then
  find "$APP_PATH" -type f -name 'ffmpeg-macos*' -exec chmod +x {} \; 2>/dev/null || true
  echo ">> Built: ${APP_PATH}"
  du -sh "$APP_PATH" 2>/dev/null || true
else
  echo "!! Expected bundle not found at ${APP_PATH} — check PyInstaller output above." >&2
  exit 1
fi
