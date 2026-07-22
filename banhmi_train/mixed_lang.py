"""Detects Vietnamese proper nouns embedded in otherwise-English text (e.g.
"Ho Chi Minh", "Da Nang", "Tran Van Huong") and phonemizes them with the
Vietnamese espeak-ng voice instead of the English one, splicing the result
back into the sentence's phoneme stream.

This is text-preprocessing/orchestration, not a phonemization primitive, so
it lives here (banhmi_train) rather than in banhmi_phonemize: it just calls
banhmi_phonemize.phonemize_espeak() once per detected-language segment and
stitches the results together. banhmi_phonemize itself stays a plain
(text, voice) -> phonemes wrapper around espeak-ng.

Detection: a small dictionary of common Vietnamese place names/phrases
(VI_DICT) catches multi-word matches first; any remaining English-tagged
text is then re-checked word-by-word with `lingua` (a statistical language
detector) to catch Vietnamese proper nouns not in that list (e.g. personal
names like "Tran", "Huong").

Vietnamese phonemes that never occur in an English-only training corpus
(e.g. ɲ, ɗ -- see VI_TO_TRAINED_MAP) are remapped to the closest phoneme
that *is* in the trained set, since feeding a model an untrained phoneme
embedding produces undefined/garbled output rather than "a slightly wrong
accent" -- correctness here means staying inside the phoneme vocabulary the
model actually learned, not phonetic precision for its own sake.

This is inference-only: the training corpus (LJSpeech) is pure English, so
preprocess.py never needs Vietnamese detection -- only say.py calls this.
Regardless of when it's used, the VI_TO_TRAINED_MAP remapping above still
applies, since the model must never see a phoneme it wasn't trained on.
"""
import re
from typing import List, Optional

from banhmi_phonemize import phonemize_espeak
from lingua import Language, LanguageDetectorBuilder

# phonemize_espeak() already marks real word boundaries with a literal " "
# element in its output list. _process_vi/_process_en below join every
# phoneme *character* with an extra " " too (so the multi-character
# VI_TO_TRAINED_MAP substitutions have something to match), which makes a
# real boundary and an ordinary join separator both look like plain spaces
# -- indistinguishable once any later step collapses whitespace. Marking
# real boundaries with this placeholder first means they survive every
# later regex/whitespace step intact, and can be turned back into a single
# " " only at the very end, after all the artificial join spaces are gone.
_WORD_BOUNDARY = "\x00"


def _join_phonemes(raw_list: List[List[str]]) -> str:
    marked = [
        [_WORD_BOUNDARY if p == " " else p for p in sentence]
        for sentence in raw_list
    ]
    return " ".join(" ".join(sentence) for sentence in marked)


# --- language detector & segmentation --------------------------------------
_detector = (
    LanguageDetectorBuilder.from_languages(Language.ENGLISH, Language.VIETNAMESE)
    .with_minimum_relative_distance(0.0)
    .build()
)


def _is_vietnamese_word(word: str) -> bool:
    clean_word = re.sub(r"[^\w]", "", word)
    if len(clean_word) < 3:
        return False
    return _detector.detect_language_of(clean_word.lower()) == Language.VIETNAMESE


VI_DICT = [
    "ho chi minh", "ha noi", "vietnam", "da nang", "ba ria vung tau",
    "pho", "banh mi", "dalat", "da lat", "saigon", "hanoi", "hcmc",
]
_VI_DICT_SORTED = sorted(VI_DICT, key=len, reverse=True)
_VI_PATTERN = re.compile(r"\b(" + "|".join(_VI_DICT_SORTED) + r")\b", re.IGNORECASE)


def segment_text(text: str) -> List[tuple]:
    """Splits `text` into [(chunk, "en"|"vi"), ...] segments."""
    # Tang 1: bat cac cum tu co trong VI_DICT
    first_pass = []
    last_end = 0
    for match in _VI_PATTERN.finditer(text):
        start, end = match.span()
        if start > last_end:
            first_pass.append((text[last_end:start].strip(), "en"))
        first_pass.append((text[start:end].strip(), "vi"))
        last_end = end
    if last_end < len(text):
        first_pass.append((text[last_end:].strip(), "en"))

    # Tang 2: fallback cho cac doan 'en' con lai (doan tung tu bang lingua)
    final_segments = []
    for seg_text, lang in first_pass:
        if not seg_text:
            continue
        if lang == "vi":
            final_segments.append((seg_text, "vi"))
            continue

        # \S+ / \s+ so whitespace between tokens is kept as its own piece
        # instead of being silently dropped by re.findall.
        words = re.findall(r"\S+|\s+", seg_text)
        current_en: List[str] = []
        for word in words:
            if word.isspace():
                current_en.append(word)
                continue
            clean_word = re.sub(r"[^\w]", "", word)
            if not clean_word:
                current_en.append(word)
                continue
            if _is_vietnamese_word(clean_word):
                if current_en:
                    final_segments.append(("".join(current_en).strip(), "en"))
                    current_en = []
                final_segments.append((clean_word, "vi"))
            else:
                current_en.append(word)
        if current_en:
            final_segments.append(("".join(current_en).strip(), "en"))

    final_segments = [(t, l) for t, l in final_segments if t]

    # Punctuation phonemizes correctly only when espeak sees it attached to
    # the clause it's actually ending -- not when it's alone, and not when
    # it's sitting in front of the *next* clause's words either (e.g. ", and
    # Hue." from "...Da Nang, and Hue." -- phonemizing ", and Hue." alone
    # silently drops that leading comma, since as far as that isolated call
    # is concerned there's no preceding clause for it to terminate). So:
    # 1. Peel any leading punctuation off a segment and append it to the
    #    end of the previous segment instead.
    # 2. If a segment is left with nothing but punctuation (or started that
    #    way), merge the whole thing onto the previous segment too.
    merged_segments = []
    for t, lang in final_segments:
        m = re.match(r"^([^\w]+)(.*)$", t, re.DOTALL)
        if m and m.group(1) and merged_segments:
            prev_t, prev_lang = merged_segments[-1]
            merged_segments[-1] = (prev_t + m.group(1), prev_lang)
            t = m.group(2)
            if not t:
                continue

        if merged_segments and not re.search(r"\w", t):
            prev_t, prev_lang = merged_segments[-1]
            merged_segments[-1] = (prev_t + t, prev_lang)
        else:
            merged_segments.append((t, lang))

    return merged_segments


# --- phoneme processing -----------------------------------------------------
# Maps Vietnamese phonemes that never occur in an English-only training
# corpus to the closest phoneme that does (see module docstring).
VI_TO_TRAINED_MAP = {
    "ɓ": "b", "ɗ": "d", "ɲ": "n", "ʈ": "tʃ", "ʂ": "ʃ",
    "ɨ": "ʊ", "ɤ": "ɜ", "t̪": "t",
    "z iə": "ɹ i ə", "t̪ a w": "t a ʊ", "ɔ j": "ɔ ɪ",
}


def _process_vi(text: str, vi_voice: str) -> str:
    raw_list = phonemize_espeak(text, vi_voice)
    raw = _join_phonemes(raw_list)

    # 1. Xoa thanh dieu tieng Viet va tag ngon ngu nhu (en), (vi)
    raw = re.sub(r"[˥˦˧˨˩]+", "", raw)
    raw = re.sub(r"\([^)]+\)", "", raw)

    # 2. Map am vi dac biet sang am vi da duoc train
    for vi, en in VI_TO_TRAINED_MAP.items():
        raw = raw.replace(vi, en)

    # 3. Fix "Da Nang": eSpeak VI doc "d" thanh "z" (giong Bac) --
    # chi thay z o DAU tu, giu nguyen z o giua/cuoi tu
    raw = re.sub(r"\bz", "d", raw)

    # 4. Fix dau "-" (ranh gioi am tiet) khong can thiet
    raw = raw.replace("-", "")

    # 5. Rule "Minh" -> "Min"
    raw = re.sub(r"i n\b", "ɪ n", raw)

    # 6. Chuan hoa khoang trang
    raw = re.sub(r"\s+", " ", raw).strip()
    return raw


def _process_en(text: str, en_voice: str) -> str:
    raw_list = phonemize_espeak(text, en_voice)
    raw = _join_phonemes(raw_list)
    raw = re.sub(r"\s+", " ", raw).strip()
    return raw


def phonemize_mixed(
    text: str,
    en_voice: str = "en-us",
    vi_voice: str = "vi",
    casing=None,
) -> List[List[str]]:
    """Phonemizes `text`, using `vi_voice` for spans detected as Vietnamese
    proper nouns and `en_voice` for everything else. Same call signature
    shape as banhmi_phonemize.phonemize_espeak (returns List[List[str]]) so
    it's a drop-in replacement in the preprocess/inference pipeline.

    `casing`, if given, is applied to each segment right before it's handed
    to espeak-ng -- segmentation itself always runs on the original,
    unmodified casing, since Vietnamese-name detection depends on it.
    """
    if not en_voice.startswith("en"):
        return phonemize_espeak(casing(text) if casing else text, en_voice)

    segs = segment_text(text)
    parts = []
    for t, lang in segs:
        if not t:
            continue
        seg_text = casing(t) if casing else t
        parts.append(
            _process_vi(seg_text, vi_voice)
            if lang == "vi"
            else _process_en(seg_text, en_voice)
        )

    # A part can already END in its own _WORD_BOUNDARY marker -- e.g.
    # phonemizing "Hi," alone comes back with a trailing real-boundary
    # space, because phonemize_espeak itself appends a pause after a
    # comma's clause terminator. Strip any leading/trailing marker each
    # part brought with it first, so joining below adds exactly one
    # boundary between segments instead of doubling up on ones already
    # there.
    parts = [p.strip(_WORD_BOUNDARY + " ") for p in parts]

    # Join segments with the same real-boundary marker _join_phonemes uses
    # internally (a plain " " here would be indistinguishable from the
    # artificial per-character join spaces once they're both just spaces).
    final_str = _WORD_BOUNDARY.join(parts)

    # Every remaining literal " " at this point is one of _process_en/
    # _process_vi's artificial per-character join separators -- real word
    # boundaries are still safely marked as _WORD_BOUNDARY, so it's safe to
    # drop all of them, then turn the markers into single real spaces.
    final_str = final_str.replace(" ", "")
    final_str = final_str.replace(_WORD_BOUNDARY, " ").strip()

    return [list(final_str)]
