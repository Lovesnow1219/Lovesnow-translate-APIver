# -*- coding: utf-8 -*-
"""Judge and clean a dubbed line before it is saved."""
import json
import os
import re

from tools.asr_source import _source_chars, split_text_into_sentences
from tools.target_language import (
    is_asr_junk,
    is_chinese_target,
    spoken_char_pace,
    spoken_word_pace,
    translation_language,
    uses_char_budget,
    uses_word_budget,
)


def _spoken_word_budget(duration, target_language='English'):
    """Space-separated tokens that can be spoken in the original slot."""
    pace = spoken_word_pace(target_language)
    duration = max(0.25, float(duration or 0))
    floor = 4 if translation_language(target_language) == 'Vietnamese' else 2
    return max(floor, int(round(duration * pace)))


def _word_slack(target_language):
    lang = translation_language(target_language)
    if lang == 'English':
        return 14
    if lang == 'Vietnamese':
        return 8
    return 2

_STUMP_PHRASES = {
    'congratulations',
    'good heavens',
    'you see',
    'see',
    'look',
    "that's it",
    'looks like it',
}
def _sentence_units(text, target_language='English'):
    if uses_char_budget(target_language) or translation_language(target_language) in {
        'Japanese', 'Thai', 'Korean', 'Cantonese',
    }:
        return _spoken_chars(text)
    return len([word for word in (text or '').split() if word not in _WORD_PUNCT])


def _is_stump_sentence(text, target_language='English'):
    units = _sentence_units(text, target_language)
    if uses_word_budget(target_language) or translation_language(target_language) == 'English':
        stripped = (text or '').rstrip()
        if units <= 2:
            return True
        if units <= 3 and stripped.endswith(('?', '!')):
            return True
        return units <= 3
    return units <= 5


def merge_split_sentences(sentences, target_language='English'):
    """Keep 'Congratulations.' attached to the real sentence instead of its own card."""
    if not sentences:
        return []
    if len(sentences) == 1:
        return list(sentences)
    out = []
    carry = ''
    for i, sentence in enumerate(sentences):
        piece = f'{carry} {sentence}'.strip() if carry else sentence
        last = i == len(sentences) - 1
        if not last and _is_stump_sentence(piece, target_language):
            carry = piece
            continue
        if last and out and _is_stump_sentence(piece, target_language):
            out[-1] = f'{out[-1]} {piece}'.strip()
            carry = ''
            continue
        out.append(piece)
        carry = ''
    if carry:
        if out:
            out[-1] = f'{out[-1]} {carry}'.strip()
        else:
            out.append(carry)
    return out


def _apportion_source(source, parts):
    source = source or ''
    if len(parts) <= 1:
        return [source]
    src_sents = split_text_into_sentences(source)
    if len(src_sents) == len(parts):
        return src_sents
    weights = [max(1, len(part or '')) for part in parts]
    total = sum(weights) or 1
    n = len(source)
    slices = []
    cursor = 0
    for i, weight in enumerate(weights):
        if i == len(parts) - 1:
            slices.append((source[cursor:] or source).strip())
            break
        take = max(1, int(round(n * weight / total)))
        end = min(n, cursor + take)
        snapped = None
        for j in range(end, min(n, end + 6)):
            if source[j - 1:j] in '。！？，、,.!? ':
                snapped = j
                break
        if snapped is None:
            for j in range(end, max(cursor + 1, end - 6), -1):
                if source[j - 1:j] in '。！？，、,.!? ':
                    snapped = j
                    break
        end = snapped or end
        if end <= cursor:
            end = min(n, cursor + take)
        piece = source[cursor:end].strip()
        slices.append(piece or source[cursor:end])
        cursor = end
    return slices


def translation_postprocess(result, target_language='简体中文'):
    result = (result or '').strip()
    if not is_chinese_target(target_language):
        result = re.sub(r'^(Translated text:\s*|Translation:\s*|译文[：:]\s*|翻译[：:]\s*)', '', result, flags=re.I)
        result = result.strip().strip('"“”\'')
        result = re.sub(r'\[[^\]]+\]', '', result)
        result = re.sub(r'\s*\n+\s*', ' ', result)
        return result.strip()
    result = re.sub(r'\（[^)]*\）', '', result)
    result = result.replace('...', '，')
    result = re.sub(r'(?<=\d),(?=\d)', '', result)
    result = result.replace('²', '的平方').replace(
        '————', '：').replace('——', '：').replace('°', '度')
    result = result.replace("AI", '人工智能')
    result = result.replace('变压器', "Transformer")
    return result


def _spoken_char_budget(duration, target_language='Thai'):
    pace = spoken_char_pace(target_language)
    duration = max(0.25, float(duration or 0))
    floor = 6 if translation_language(target_language) == 'Japanese' else 4
    return max(floor, int(round(duration * pace)))


def _char_slack(target_language):
    return 20 if translation_language(target_language) == 'Japanese' else 4


def _slot_seconds(line):
    orig_start = line.get('orig_start')
    orig_end = line.get('orig_end')
    if orig_start is not None and orig_end is not None:
        span = float(orig_end) - float(orig_start)
        if span > 0.05:
            return max(0.25, span)
    return max(0.25, float(line.get('end') or 0) - float(line.get('start') or 0))


_INSTRUCTION_MARKERS = (
    'only translate the following',
    'the translation is too long',
    "don't include",
    'only output the',
    'give me the result',
    'give me the final translation',
)


def _unwrap_translation(translation):
    translation = (translation or '').strip()
    if translation.startswith('```') and translation.endswith('```'):
        return translation[3:-3].strip()
    if (translation.startswith('“') and translation.endswith('”')) or (
        translation.startswith('"') and translation.endswith('"')
    ):
        return translation[1:-1].strip()
    labeled = '翻译' in translation or '译文' in translation or 'Translation' in translation
    if labeled:
        for left, right in (('：“', '”'), ('："', '"'), (':"', '"'), (': "', '"')):
            if left in translation and right in translation:
                return translation.split(left)[-1].split(right)[0].strip()
    return translation


def _source_chars(text):
    return len(re.sub(r'\s+', '', text or ''))


def _spoken_chars(text):
    return len(re.sub(r'[\s\.,!?;:。、！？…・「」『』（）()\[\]【】…]+', '', text or ''))

KANA_RE = re.compile(r'[\u3040-\u30ff]')
HAN_RE = re.compile(r'[\u4e00-\u9fff]')
CN_LEFT_RE = re.compile(r'[的们这什么没还过给吗吧]')
# 的 is also Japanese てき. Only flag simplified-only particles.
JA_CN_LEFT_RE = re.compile(r'[们这什么没还过给吗吧]')
STAT_RE = re.compile(r'[0-9０-９]+[/／][0-9０-９]+|[%％]|[+＋]\s*[0-9０-９]')
END_RE = re.compile(r'[。！？.!?]$')
_WORD_PUNCT = {',', '.', '!', '?', ';', ':', '…', '...', '—', '-'}


def _stat_fix_message(lang):
    if lang == 'English':
        return (
            'TTS may skip 0/10, %, or +. '
            'Write spoken English such as zero of ten or plus fifteen percent. '
            'Output only the line.'
        )
    if lang == 'Vietnamese':
        return (
            'TTS may skip 0/10, %, or +. '
            'Write spoken Vietnamese such as không trên mười or cộng mười lăm phần trăm. '
            'Output only the line.'
        )
    return (
        'TTS cannot read 0/10, %, or +. '
        'Write spoken Japanese such as ゼロの十 or プラス十五パーセント. '
        'Output only the Japanese line.'
    )


def _source_unfinished(text):
    return (text or '').rstrip().endswith(('…', '...', '，', ',', '、'))


def leftover_chinese_reason(cleaned, target_language):
    if is_chinese_target(target_language):
        return None
    lang = translation_language(target_language)
    if lang == 'Japanese':
        return None
    if HAN_RE.search(cleaned or ''):
        return (
            f'Do not leave Chinese. Write spoken {lang}. '
            'Output only the line.'
        )
    return None


def _forbids_chinese_script(target_language):
    return leftover_chinese_reason('字', target_language) is not None


def usable_target_text(text, target_language):
    cleaned = translation_postprocess(text, target_language)
    if not cleaned or leftover_chinese_reason(cleaned, target_language):
        return ''
    return cleaned


def leftover_chinese_indices(transcript, target_language):
    if not _forbids_chinese_script(target_language):
        return []
    return [
        i for i, line in enumerate(transcript or [])
        if leftover_chinese_reason(line.get('translation') or '', target_language)
    ]


def _folder_has_leftover_chinese(folder, target_language):
    path = os.path.join(folder, 'translation.json')
    if not os.path.isfile(path) or not _forbids_chinese_script(target_language):
        return False
    with open(path, 'r', encoding='utf-8') as handle:
        data = json.load(handle)
    return bool(leftover_chinese_indices(data, target_language))



def _incomplete_reason(text, cleaned, target_language, duration):
    leftover = leftover_chinese_reason(cleaned, target_language)
    if leftover:
        return leftover
    lang = translation_language(target_language)
    compact = _source_chars(text)
    if lang in {'English', 'Vietnamese', 'Japanese'}:
        source_cut = _source_unfinished(text) or not END_RE.search((text or '').rstrip())
        if compact >= 8 and cleaned.rstrip().endswith(('…', '...', '……')) and not source_cut:
            return (
                'The translation is incomplete. Finish the whole sentence. '
                f'Output only the {lang} line.'
            )
        if lang != 'English' and STAT_RE.search(cleaned):
            return _stat_fix_message(lang)
    if lang in {'English', 'Vietnamese'}:
        words = [w for w in cleaned.split() if w not in _WORD_PUNCT]
        if lang == 'English':
            stump = re.sub(r"[^a-zA-Z' ]+", ' ', cleaned).strip().lower()
            stump = re.sub(r'\s+', ' ', stump)
            if compact >= 8 and stump in _STUMP_PHRASES:
                return (
                    'Do not output only an interjection. Translate the whole sentence. '
                    'Output only the line.'
                )
        if compact < 10:
            return None
        expected = max(3, min(
            int(_spoken_word_budget(duration, target_language) * 0.45),
            int(compact * (0.25 if lang == 'Vietnamese' else 0.10)),
        ))
        if len(words) < expected * 0.5:
            return (
                f'The translation dropped clauses. Translate the whole sentence into spoken {lang}. '
                'Output only the line.'
            )
        if lang == 'English':
            if compact >= 10 and len(words) <= 2:
                return (
                    'The translation dropped clauses. Translate the whole sentence into spoken English. '
                    'Output only the line.'
                )
            if compact >= 16 and len(words) <= 3:
                return (
                    'The translation dropped clauses. Translate the whole sentence into spoken English. '
                    'Output only the line.'
                )
            if compact >= 16:
                bits = [bit.strip() for bit in re.split(r'[.!?]+', cleaned) if bit.strip()]
                if len(bits) >= 3 and all(len(bit.split()) <= 2 for bit in bits):
                    return (
                        'Do not telegraph. Write a spoken English sentence. '
                        'Output only the line.'
                    )
        return None
    if lang == 'Japanese':
        if compact >= 6 and HAN_RE.search(cleaned) and not KANA_RE.search(cleaned):
            return (
                'Kanji-only lines are read as Chinese. Add hiragana or katakana. '
                'Output only the Japanese line.'
            )
        if JA_CN_LEFT_RE.search(cleaned) or re.search(r'[威薇维]拉|本小姐', cleaned):
            return (
                'Do not leave Chinese names or particles. Willa is ウィラ. '
                'Output only the Japanese line.'
            )
        han = len(HAN_RE.findall(cleaned))
        kana = len(KANA_RE.findall(cleaned))
        spoken = _spoken_chars(cleaned)
        if spoken >= 18 and kana and not HAN_RE.search(cleaned):
            return (
                'Do not spell Chinese readings in hiragana. Write normal mixed Japanese. '
                'Output only the line.'
            )
        if spoken >= 6 and han >= 5 and kana < 2:
            return (
                'Too little hiragana or katakana. TTS will skip those words. '
                'Write spoken Japanese. Output only the line.'
            )
        if compact < 10:
            return None
        expected = max(6, min(
            int(_spoken_char_budget(duration, target_language) * 0.5),
            int(compact * 0.3),
        ))
        if spoken < expected * 0.65:
            return (
                'The translation dropped clauses. Translate the whole sentence into spoken Japanese. '
                'Output only the line.'
            )
        return None
    return None


_SYSTEM_UI_RE = re.compile(
    r'(系統|系统|任務|任务|獲得|获得|提示|等級|等级|經驗|经验|背包|商店|屬性|属性|'
    r'【|】|\[系統\]|\[系统\]|\bHP\b|\bMP\b|\bEXP\b)',
    re.I,
)
_NARRATOR_SPEAKERS = {
    'narrator', 'narration', 'system', 'ui', '旁白', '系统', '系統',
}


def _extra_budget_scale(line):
    speaker = str((line or {}).get('speaker') or '').strip().lower()
    text = (line or {}).get('text') or ''
    if speaker in _NARRATOR_SPEAKERS or speaker.startswith('narrat'):
        return 1.4
    if _SYSTEM_UI_RE.search(text):
        return 1.35
    if '心想' in text or text.startswith('（') or text.startswith('('):
        return 1.25
    return 1.0


def _translate_user_content(text, duration, target_language, budget_scale=1.0, extra_note=''):
    lang = translation_language(target_language)
    note = extra_note or ''
    if uses_word_budget(target_language):
        budget = max(2, int(round(_spoken_word_budget(duration, target_language) * budget_scale)))
        if lang == 'Vietnamese':
            return (
                f'{note}Translate the whole line into spoken Vietnamese in {duration:.1f}s, '
                f'up to {budget} syllables (one space per syllable). '
                f'Keep every clause; speak numbers (không trên mười, not 0/10). '
                f'Keep 嗯/呵/哼/嘿嘿 as Ừ/Hề/Hừ/He he. 本小姐 is the speaker (tôi). '
                f'Do not end with … unless the source is unfinished:"{text}"'
            )
        if lang == 'English':
            return (
                f'{note}Translate the whole line into spoken English, about {duration:.1f}s '
                f'({budget} words is a guide, a bit more is OK). '
                f'Use the outline names. Keep meaning, emotion, and 啊吧呢呀嗯呵哼. '
                f'Do not add names that are not in this line. Do not chop into fragments:"{text}"'
            )
        return f'{note}Translate in {duration:.1f}s, max {budget} {lang} words:"{text}"'
    if uses_char_budget(target_language):
        budget = max(4, int(round(_spoken_char_budget(duration, target_language) * budget_scale)))
        if lang == 'Japanese':
            return (
                f'{note}Translate the whole line into spoken Japanese in {duration:.1f}s, '
                f'up to {budget} characters. Mixed Japanese, not Chinese calques. '
                f'ウィラ not 薇拉. リアルワールド/マネーカード/ウォーゲーム/金庫, not 現実世界貨幣変換. '
                f'Keep レベル99 and 30日 as digits. Never 0/10, %, or +. '
                f'Keep 嗯/呵/哼/嘿嘿 as ん/ふふ/ふん/へへ. 本小姐 is この私. Keep every clause:"{text}"'
            )
        return f'{note}Translate in {duration:.1f}s, max {budget} {lang} characters:"{text}"'
    return f'{note}Translate into spoken {lang}:"{text}"'
def valid_translation(text, translation, target_language='简体中文', duration=None, budget_scale=1.0):
    translation = (translation or '').strip()
    if not translation:
        return False, 'Only output the translation.'
    if any(marker in translation.lower() for marker in _INSTRUCTION_MARKERS):
        return False, 'Only output the translation.'

    translation = _unwrap_translation(translation)
    scale = max(0.5, float(budget_scale or 1.0))

    if uses_word_budget(target_language) or uses_char_budget(target_language):
        lang = translation_language(target_language)
        cleaned = translation_postprocess(translation, target_language)
        if not cleaned:
            return False, f'Only output the {lang} translation.'
        if len(text) > 0 and len(cleaned) > max(120, len(text) * 8):
            return False, 'Only translate the following sentence and give me the result.'
        incomplete = _incomplete_reason(text, cleaned, target_language, duration)
        if incomplete:
            return False, incomplete
        if duration is not None and uses_word_budget(target_language):
            budget = max(2, int(round(_spoken_word_budget(duration, target_language) * scale)))
            words = len(cleaned.split())
            if words > budget + _word_slack(target_language):
                unit = 'syllables' if lang == 'Vietnamese' else 'spoken words'
                return False, (
                    f'Too long for a {float(duration):.1f}s dubbing line. '
                    f'Rewrite in at most {budget} {lang} {unit}. '
                    'Keep names and meaning. Output only the translation.'
                )
        if duration is not None and uses_char_budget(target_language):
            budget = max(4, int(round(_spoken_char_budget(duration, target_language) * scale)))
            if len(cleaned) > budget + _char_slack(target_language):
                return False, (
                    f'Too long for a {float(duration):.1f}s dubbing line. '
                    f'Rewrite in at most {budget} {lang} characters. '
                    'Keep names and meaning. Output only the translation.'
                )
        return True, cleaned

    if len(text) <= 10:
        if len(translation) > 15:
            return False, f'Only translate the following sentence and give me the result.'
    elif len(translation) > len(text)*0.75:
        return False, f'The translation is too long. Only translate the following sentence and give me the result.'

    forbidden = ['翻译', '译文', '这句', '\n', '简体中文', '中文', 'translate', 'Translate', 'translation', 'Translation']
    translation = translation.strip()
    for word in forbidden:
        if word in translation:
            return False, f"Don't include `{word}` in the translation. Only translate the following sentence and give me the result."

    return True, translation_postprocess(translation, target_language)
def split_sentences(translation, use_char_based_end=True, target_language='English'):
    output_data = []
    for item in translation:
        start = item['start']
        text = item['text']
        speaker = item['speaker']
        translation_text = item['translation']
        extra = {
            key: value for key, value in item.items()
            if key not in {'start', 'end', 'text', 'speaker', 'translation', 'orig_start', 'orig_end'}
        }

        if not translation_text or not str(translation_text).strip():
            output_data.append({
                **extra,
                "start": round(start, 3),
                "end": round(item['end'], 3),
                "text": text,
                "speaker": speaker,
                "translation": '' if is_asr_junk(text) or not is_chinese_target(target_language) else (translation_text or "未翻译"),
            })
            continue

        sentences = merge_split_sentences(
            split_text_into_sentences(translation_text), target_language,
        )
        if not sentences:
            sentences = [translation_text]

        if use_char_based_end:
            duration_per_char = (item['end'] - item['start']) / max(1, len(translation_text))
        else:
            duration_per_char = 0

        sources = _apportion_source(text, sentences)
        for i, sentence in enumerate(sentences):
            if use_char_based_end:
                sentence_end = start + duration_per_char * len(sentence)
            else:
                sentence_end = item['end']
            if i == len(sentences) - 1:
                sentence_end = item['end']
            source = sources[i] if i < len(sources) and sources[i] else text
            child = {
                **extra,
                "start": round(start, 3),
                "end": round(sentence_end, 3),
                "orig_start": round(start, 3),
                "orig_end": round(sentence_end, 3),
                "text": source,
                "speaker": speaker,
                "translation": sentence,
            }
            if source != text:
                child['parent_text'] = text
            output_data.append(child)
            if use_char_based_end:
                start = sentence_end

    return output_data

