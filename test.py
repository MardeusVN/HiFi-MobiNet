import re
from banhmi_phonemize import phonemize_espeak
from lingua import Language, LanguageDetectorBuilder

# phonemize_espeak() already marks real word boundaries with a literal " "
# element in its output list. process_en/process_vi below join every
# phoneme *character* with an extra " " too (so the multi-character
# VI_TO_TRAINED_MAP substitutions have something to match), which makes a
# real boundary and an ordinary join separator both look like plain spaces
# -- indistinguishable once any later step collapses whitespace. Marking
# real boundaries with this placeholder first means they survive every
# later regex/whitespace step intact, and can be turned back into a single
# " " only at the very end, after all the artificial join spaces are gone.
_WORD_BOUNDARY = "\x00"


def _join_phonemes(raw_list):
    marked = [
        [_WORD_BOUNDARY if p == " " else p for p in sentence]
        for sentence in raw_list
    ]
    return " ".join(" ".join(sentence) for sentence in marked)

# ==========================================
# LANGUAGE DETECTOR & PHÂN ĐOẠN
# ==========================================
detector = LanguageDetectorBuilder.from_languages(
    Language.ENGLISH, Language.VIETNAMESE
).with_minimum_relative_distance(0.0).build()

def is_vietnamese_word(word):
    clean_word = re.sub(r'[^\w]', '', word)
    if len(clean_word) < 3:
        return False
    return detector.detect_language_of(clean_word.lower()) == Language.VIETNAMESE

VI_DICT = [
    "ho chi minh", "ha noi", "vietnam", "da nang", "ba ria vung tau",
    "pho", "banh mi", "dalat", "da lat", "saigon", "hanoi", "hcmc",
]
VI_DICT.sort(key=len, reverse=True)
VI_PATTERN = re.compile(r'\b(' + '|'.join(VI_DICT) + r')\b', re.IGNORECASE)

def segment_text(text):
    # Tầng 1: bắt các cụm từ có trong VI_DICT
    first_pass = []
    last_end = 0
    for match in VI_PATTERN.finditer(text):
        start, end = match.span()
        if start > last_end:
            first_pass.append((text[last_end:start].strip(), 'en'))
        first_pass.append((text[start:end].strip(), 'vi'))
        last_end = end
    if last_end < len(text):
        first_pass.append((text[last_end:].strip(), 'en'))

    # Tầng 2: fallback cho các đoạn 'en' còn lại (đoán từng từ bằng lingua)
    final_segments = []
    for seg_text, lang in first_pass:
        if not seg_text:
            continue
        if lang == 'vi':
            final_segments.append((seg_text, 'vi'))
        else:
            # \S+ / \s+ so whitespace between tokens is kept as its own
            # piece instead of being silently dropped by re.findall.
            words = re.findall(r'\S+|\s+', seg_text)
            current_en = []
            for word in words:
                if word.isspace():
                    current_en.append(word)
                    continue
                clean_word = re.sub(r'[^\w]', '', word)
                if not clean_word:
                    current_en.append(word)
                    continue
                if is_vietnamese_word(clean_word):
                    if current_en:
                        final_segments.append(("".join(current_en).strip(), 'en'))
                        current_en = []
                    final_segments.append((clean_word, 'vi'))
                else:
                    current_en.append(word)
            if current_en:
                final_segments.append(("".join(current_en).strip(), 'en'))

    final_segments = [(t, l) for t, l in final_segments if t]

    # Punctuation phonemizes correctly only when espeak sees it attached to
    # the clause it's actually ending -- not when it's alone, and not when
    # it's sitting in front of the *next* clause's words either (e.g. ", and
    # Hue." from "...Da Nang, and Hue." -- phonemize_espeak(", and Hue.")
    # silently drops that leading comma, since as far as that isolated call
    # is concerned there's no preceding clause for it to terminate). So:
    # 1. Peel any leading punctuation off a segment and append it to the
    #    end of the previous segment instead.
    # 2. If a segment is left with nothing but punctuation (or started that
    #    way), merge the whole thing onto the previous segment too.
    merged_segments = []
    for t, lang in final_segments:
        m = re.match(r'^([^\w]+)(.*)$', t, re.DOTALL)
        if m and m.group(1) and merged_segments:
            prev_t, prev_lang = merged_segments[-1]
            merged_segments[-1] = (prev_t + m.group(1), prev_lang)
            t = m.group(2)
            if not t:
                continue

        if merged_segments and not re.search(r'\w', t):
            prev_t, prev_lang = merged_segments[-1]
            merged_segments[-1] = (prev_t + t, prev_lang)
        else:
            merged_segments.append((t, lang))

    return merged_segments

# ==========================================
# HÀM XỬ LÝ ÂM VỊ
# ==========================================
# Ánh xạ các âm vị tiếng Việt sang âm vị gần nhất mà dữ liệu train tiếng Anh
# (LJSpeech) thực sự có -- tránh dùng phải embedding chưa từng được train.
# (ɲ, ɗ... không xuất hiện trong LJSpeech nên embedding của chúng chưa từng
# nhận gradient -- xem lại phần thảo luận về ký hiệu chưa-train.)
VI_TO_TRAINED_MAP = {
    'ɓ': 'b', 'ɗ': 'd', 'ɲ': 'n', 'ʈ': 'tʃ', 'ʂ': 'ʃ',
    'ɨ': 'ʊ', 'ɤ': 'ɜ', 't̪': 't',
    'z iə': 'ɹ i ə', 't̪ a w': 't a ʊ', 'ɔ j': 'ɔ ɪ',
}

def process_vi(text):
    raw_list = phonemize_espeak(text, "vi")
    raw = _join_phonemes(raw_list)

    # 1. Xóa thanh điệu tiếng Việt và tag ngôn ngữ như (en), (vi)
    raw = re.sub(r'[˥˦˧˨˩]+', '', raw)
    raw = re.sub(r'\([^)]+\)', '', raw)

    # 2. Map âm vị đặc biệt sang âm vị đã được train
    for vi, en in VI_TO_TRAINED_MAP.items():
        raw = raw.replace(vi, en)

    # 3. Fix "Da Nang": eSpeak VI đọc "d" thành "z" (giọng Bắc) --
    # chỉ thay z ở ĐẦU từ, giữ nguyên z ở giữa/cuối từ
    raw = re.sub(r'\bz', 'd', raw)

    # 4. Fix dấu "-" (ranh giới âm tiết) không cần thiết
    raw = raw.replace('-', '')

    # 5. Rule "Minh" -> "Min"
    raw = re.sub(r'i n\b', 'ɪ n', raw)

    # 6. Chuẩn hóa khoảng trắng
    raw = re.sub(r'\s+', ' ', raw).strip()
    return raw

def process_en(text):
    raw_list = phonemize_espeak(text, "en-us")
    raw = _join_phonemes(raw_list)
    raw = re.sub(r'\s+', ' ', raw).strip()
    return raw

# ==========================================
# PIPELINE TỔNG HỢP
# ==========================================
def phonemize_text(text):
    segs = segment_text(text)
    parts = []
    for t, lang in segs:
        if not t:
            continue
        parts.append(process_vi(t) if lang == 'vi' else process_en(t))

    # A part can already END in its own _WORD_BOUNDARY marker -- e.g.
    # process_en("Hi,") comes back with a trailing real-boundary space,
    # because phonemize_espeak itself appends a pause after a comma's
    # clause terminator. Strip any leading/trailing marker each part
    # brought with it first, so joining below adds exactly one boundary
    # between segments instead of doubling up on ones that are already
    # there.
    parts = [p.strip(_WORD_BOUNDARY + " ") for p in parts]

    # Join segments with the same real-boundary marker _join_phonemes uses
    # internally (a plain " " here would be indistinguishable from the
    # artificial per-character join spaces once they're both just spaces).
    final_str = _WORD_BOUNDARY.join(parts)

    # Every remaining literal " " at this point is one of process_en/
    # process_vi's artificial per-character join separators -- real word
    # boundaries are still safely marked as _WORD_BOUNDARY, so it's safe to
    # drop all of them, then turn the markers into single real spaces.
    final_str = final_str.replace(' ', '')
    final_str = final_str.replace(_WORD_BOUNDARY, ' ').strip()

    # Convert string thành list of chars, bọc trong list ngoài
    char_list = list(final_str)
    return [char_list]  # Trả về List[List[str]]

print("Loading language detector...")

text = "Tran Van Huong is my dad."
pn = phonemize_espeak(text, "en-us")
print("Banhmi:", pn)

custome = phonemize_text(text)
print("Custom:", custome)