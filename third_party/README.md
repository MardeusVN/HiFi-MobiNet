# third_party

## banhmi-phonemize

`banhmi_phonemize` is BanhmiTTS's own espeak-ng phonemization library. Its
C++/Python source (`banhmi-phonemize/src/`, `banhmi-phonemize/banhmi_phonemize/__init__.py`)
is an **independent implementation written for this project** -- it does not
compile or import any of `piper_phonemize`'s source code. The build/packaging
setup (CMakeLists.txt, pyproject.toml layout) was originally scaffolded from
the community fork [piper-phonemize-fix](https://github.com/easyaspi314/piper-phonemize-fix)
(Windows build fixes for `piper_phonemize`, since the official PyPI package
only ships Linux/macOS wheels); that vendored copy has since been removed
now that this package no longer depends on it. This is the version
`banhmi_train/preprocess.py` actually imports.

Its symbol/id table and its punctuation/sentence-boundary reconstruction
approach were arrived at by studying `piper_phonemize`'s public behavior and
source, so that phoneme output and phoneme ids stay numerically compatible
with it and with existing Piper-trained models -- see
`banhmi-phonemize/NOTICE.md` for the full attribution.

Compared to upstream `piper_phonemize`:

- **No Arabic tashkeel / onnxruntime dependency.** Only used for Arabic
  diacritization, which BanhmiTTS doesn't need.
- **Only English + Vietnamese espeak-ng dictionaries are compiled**, not all
  ~120 languages upstream ships. Patched into espeak-ng's own build
  (`cmake/data.cmake`) via `patch_espeak_langs.cmake`, so the (slow)
  per-language dictionary-compile step only runs for the two languages this
  project needs. Change `ESPEAK_NG_LANGS` in `CMakeLists.txt` to add more.
- **No `uni_algo` dependency.** Unicode NFD normalization and codepoint
  splitting are done in Python (`unicodedata`, stdlib) instead of a vendored
  2MB/72k-line C++ header library -- the C++ layer only calls espeak-ng and
  returns raw phoneme strings.
- **No standalone CLI executable, no `phoneme_ids.cpp`.** Only the espeak-ng
  call itself is C++; phoneme/id mapping is plain Python.
- **Caches the last-selected espeak-ng voice** and skips re-selecting it on
  every call, since every real caller here phonemizes a whole dataset with
  one fixed voice.

`espeak-ng` (a pinned commit of
[rhasspy/espeak-ng](https://github.com/rhasspy/espeak-ng), the fork Piper
depends on for its punctuation/sentence-boundary-preserving
`espeak_TextToPhonemesWithTerminator` function) is **compiled from source**
at build time, not installed as a prebuilt binary.

Verified against the original `piper_phonemize` (run on Linux/WSL) on the
full 13,100-utterance LJSpeech corpus: **100% identical phoneme output**.
Also faster and lighter than upstream `piper_phonemize`: ~2.9s vs. piper's
~8.2s to phonemize the whole corpus, ~1.7MB installed vs. piper's ~38MB.

### Building

Requires Visual Studio Build Tools ("Desktop development with C++"
workload) and network access.

```powershell
powershell -File build_banhmi_phonemize.ps1
```
