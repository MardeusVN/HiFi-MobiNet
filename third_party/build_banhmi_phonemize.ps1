# Builds banhmi_phonemize from source (this directory) instead of installing
# a prebuilt wheel. Requires:
#   - Visual Studio Build Tools with the "Desktop development with C++"
#     workload (provides cl.exe / MSVC)
#   - Network access: CMake downloads espeak-ng source (a pinned commit of
#     https://github.com/rhasspy/espeak-ng)
#
# Usage: powershell -File build_banhmi_phonemize.ps1

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\banhmi-phonemize

# PowerShell does not stop on a failing native command (like python.exe) by
# itself -- check $LASTEXITCODE after each step so a build failure doesn't
# silently fall through to the import test below (which would then pick up
# the raw, uninstalled source folder instead of a real installed package,
# since Python always adds the current directory to its module search path).
python -m pip install "scikit-build-core" "pybind11>=3.0.1" cmake ninja
if ($LASTEXITCODE -ne 0) { throw "pip install (build deps) failed" }

python -m pip install .
if ($LASTEXITCODE -ne 0) { throw "pip install . (banhmi_phonemize build) failed" }

Push-Location $env:TEMP
try {
    python -c "from banhmi_phonemize import phonemize_espeak; print(phonemize_espeak('Hello world', 'en-us'))"
    if ($LASTEXITCODE -ne 0) { throw "import test failed" }
} finally {
    Pop-Location
}
