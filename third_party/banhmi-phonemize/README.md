# banhmi_phonemize

espeak-ng-based text-to-phoneme library for BanhmiTTS. Independent
implementation (own `src/` C++ + `banhmi_phonemize/__init__.py` Python) --
see `NOTICE.md` for what it's compatible with and why.

## Usage

```python
from banhmi_phonemize import phonemize_espeak, phoneme_ids_espeak

sentences = phonemize_espeak("This is a test.", "en-us")
# [['ð', 'ɪ', 's', ' ', 'ɪ', 'z', ' ', 'ɐ', ' ', 't', 'ˈ', 'ɛ', 's', 't', '.']]

phoneme_ids = phoneme_ids_espeak([p for sentence in sentences for p in sentence])
```

## Building

```sh
python -m pip install .
```

Requires a C++ toolchain (MSVC on Windows) and network access -- CMake
downloads and compiles espeak-ng from source. See `../README.md` and
`../build_banhmi_phonemize.ps1` for the Windows build path this project
actually uses.
