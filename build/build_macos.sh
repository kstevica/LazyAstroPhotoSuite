#!/usr/bin/env bash
#
# Build LazyAstroPhotoSuite as a standalone macOS .app with Nuitka.
#
#   ./build/build_macos.sh
#
# Output:  build/dist/LazyAstroPhotoSuite.app  (a self-contained bundle; no Python needed)
#
# Prereqs (once):  .venv/bin/python -m pip install -e ".[gui,video,bg,tools,build]"
# First compile takes ~10-30 min. The .app is arm64 (matches this machine's Python).
#
# What the flags do (see build/README.md):
#   --enable-plugin=pyside6            Qt cocoa + JPEG imageformats plugins (the .jpg cards)
#   --include-data-dir=...             launcher assets + data JSON + DSO catalogs (loaded by
#                                      Path(__file__); their subdirs have no __init__.py so
#                                      package-data alone can miss them → include explicitly)
#   --include-package=imageio_ffmpeg   bundles the ffmpeg binary (dll-files plugin) for export
#   --nofollow-import-to=...           skip optional deps that aren't installed (SPCC/XISF)
#
set -eo pipefail   # not -u: macOS bash 3.2 errors on empty-array expansion
cd "$(dirname "$0")/.."

# Nuitka standalone needs a NON-Apple CPython (the system /usr/bin/python3 is rejected).
# Prefer a dedicated build venv made from Homebrew/python.org CPython; see build/README.md.
PY="${PYTHON:-}"
if [[ -z "$PY" ]]; then
  if   [[ -x .venv-build/bin/python ]]; then PY=".venv-build/bin/python"
  else PY=".venv/bin/python"; fi
fi
APP="LazyAstroPhotoSuite"
VERSION="0.1.0"

echo ">> Nuitka $("$PY" -m nuitka --version 2>/dev/null | head -1) building ${APP} ${VERSION}"
echo ">> $(date)"

ICON_ARG=()
if [[ -f build/appicon.icns ]]; then
  ICON_ARG=(--macos-app-icon=build/appicon.icns)
fi

"$PY" -m nuitka \
  --standalone \
  --macos-create-app-bundle \
  --macos-app-name="${APP}" \
  --macos-app-version="${VERSION}" \
  ${ICON_ARG[@]+"${ICON_ARG[@]}"} \
  --company-name="Stevica Kuharski" \
  --product-name="${APP}" \
  --product-version="${VERSION}" \
  --enable-plugin=pyside6 \
  --include-package=lazystretch \
  --include-package=astropy \
  --include-package=imageio_ffmpeg \
  --include-data-dir=lazystretch/data=lazystretch/data \
  --include-data-dir=lazystretch/gui/assets=lazystretch/gui/assets \
  --nofollow-import-to=astroquery \
  --nofollow-import-to=gaiaxpy \
  --nofollow-import-to=xisf \
  --nofollow-import-to=pytest \
  --assume-yes-for-downloads \
  --output-dir=build/dist \
  --output-filename="${APP}" \
  lazystretch/gui/app.py

# Nuitka names the bundle after the entry module (app.app); rename to the product name.
APP_PATH="build/dist/${APP}.app"
if [[ -d build/dist/app.app ]]; then
  rm -rf "$APP_PATH"
  mv build/dist/app.app "$APP_PATH"
fi
if [[ -d "$APP_PATH" ]]; then
  find "$APP_PATH" -type f -name 'ffmpeg-macos*' -exec chmod +x {} \; 2>/dev/null || true
  echo ">> Built: ${APP_PATH}"
  du -sh "$APP_PATH" 2>/dev/null || true
else
  echo "!! Expected bundle not found (looked for build/dist/app.app) — check Nuitka output." >&2
  exit 1
fi
