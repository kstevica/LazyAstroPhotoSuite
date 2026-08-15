# Build LazyAstroPhotoSuite.exe as a standalone Windows app with PyInstaller.
#
#   powershell -ExecutionPolicy Bypass -File build\build_windows.ps1
#
# Output: build\dist\LazyAstroPhotoSuite\LazyAstroPhotoSuite.exe  (self-contained folder).
#
# This MUST run ON Windows — PyInstaller cannot cross-compile from macOS (the interpreter,
# the C-extension wheels and the bootloader are all platform-specific). Use the GitHub
# Actions workflow (.github/workflows/build-windows.yml) or a Windows VM. See build/README.md.
#
# Prereqs (once, on Windows):  py -3.11 -m venv .venv-build ; .\.venv-build\Scripts\pip install -e ".[gui,video,bg,tools,build]"

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)     # repo root
$app = "LazyAstroPhotoSuite"

$py = $env:PYTHON
if (-not $py) {
  if (Test-Path ".venv-build\Scripts\python.exe") { $py = ".venv-build\Scripts\python.exe" }
  else { $py = "python" }
}

$iconArgs = @()
if (Test-Path "build\appicon.ico") { $iconArgs = @("--icon", "$PWD\build\appicon.ico") }

Write-Host ">> PyInstaller building $app on Windows (python: $py)"

# NOTE: on Windows --add-data uses ';' as the src;dest separator (':' on macOS/Linux).
& $py -m PyInstaller `
  --name $app `
  --windowed `
  --noconfirm --clean `
  @iconArgs `
  --distpath build\dist `
  --workpath build\pyi-work `
  --specpath build\pyi-spec `
  --add-data "$PWD\lazystretch\gui\assets;lazystretch/gui/assets" `
  --add-data "$PWD\lazystretch\data;lazystretch/data" `
  --collect-submodules lazystretch `
  --collect-all imageio_ffmpeg `
  --collect-data astropy `
  --collect-data photutils `
  --collect-data skimage `
  "$PWD\build\laps_entry.py"

if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed (exit $LASTEXITCODE)" }

$exe = "build\dist\$app\$app.exe"
if (Test-Path $exe) {
  Write-Host ">> Built: $exe"
} else {
  throw "Expected $exe not found - check PyInstaller output above."
}
