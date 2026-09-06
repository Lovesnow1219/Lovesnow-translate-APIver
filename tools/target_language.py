# -*- coding: utf-8 -*-
"""One UI language → translation label + TTS label."""
import os
import re

TARGET_LANGUAGES = [
    'English',
    'Japanese',
    '越南文',
    '簡體中文',
    '繁體中文',
    '粵語',
    'Korean（部分支援）',
    '西班牙文（部分支援）',
    'French（部分支援）',
    '泰文（部分支援）',
    '印尼文（部分支援）',
    '馬來文（部分支援）',
    '菲律賓文（部分支援）',
]
_PARTIAL_MARK = '（部分支援）'

# UI / alias → (translation, tts)
_MAP = {
    'English': ('English', 'English'),
    'en': ('English', 'English'),
    '簡體中文': ('简体中文', '中文'),
    '简体中文': ('简体中文', '中文'),
    '中文': ('简体中文', '中文'),
    '繁體中文': ('繁体中文', '中文'),
    '繁体中文': ('繁体中文', '中文'),
    '粵語': ('Cantonese', '粤语'),
    '粤语': ('Cantonese', '粤语'),
    'Cantonese': ('Cantonese', '粤语'),
    'Japanese': ('Japanese', 'Japanese'),
    'Korean': ('Korean', 'Korean'),
    '西班牙文': ('Spanish', 'Spanish'),
    '西班牙語': ('Spanish', 'Spanish'),
    '西班牙语': ('Spanish', 'Spanish'),
    'Spanish': ('Spanish', 'Spanish'),
    'es': ('Spanish', 'Spanish'),
    'French': ('French', 'French'),
    '越南文': ('Vietnamese', 'Vietnamese'),
    '越南語': ('Vietnamese', 'Vietnamese'),
    '越南语': ('Vietnamese', 'Vietnamese'),
    'Vietnamese': ('Vietnamese', 'Vietnamese'),
    'vi': ('Vietnamese', 'Vietnamese'),
    '泰文': ('Thai', 'Thai'),
    '泰語': ('Thai', 'Thai'),
    '泰语': ('Thai', 'Thai'),
    'Thai': ('Thai', 'Thai'),
    'th': ('Thai', 'Thai'),
    '印尼文': ('Indonesian', 'Indonesian'),
    '印尼語': ('Indonesian', 'Indonesian'),
    '印尼语': ('Indonesian', 'Indonesian'),
    'Indonesian': ('Indonesian', 'Indonesian'),
    'id': ('Indonesian', 'Indonesian'),
    '馬來文': ('Malay', 'Malay'),
    '馬來語': ('Malay', 'Malay'),
    '马来文': ('Malay', 'Malay'),
    '马来语': ('Malay', 'Malay'),
    'Malay': ('Malay', 'Malay'),
    'ms': ('Malay', 'Malay'),
    '菲律賓文': ('Filipino', 'Filipino'),
    '菲律宾语': ('Filipino', 'Filipino'),
    'Filipino': ('Filipino', 'Filipino'),
    'Tagalog': ('Filipino', 'Filipino'),
    'tl': ('Filipino', 'Filipino'),
    'fil': ('Filipino', 'Filipino'),
}

CHINESE_TRANSLATION_LANGUAGES = {'简体中文', '繁体中文', '中文'}
WORD_BUDGET_LANGUAGES = {
    'English', 'Vietnamese', 'Indonesian', 'Malay', 'Filipino',
    'Spanish', 'French', 'Polish',
}
CHAR_BUDGET_LANGUAGES = {'Thai', 'Japanese'}
# Spoken tokens per second. English 2.2 matches Fish; 3.0 never triggered tighten.
# Vietnamese spaces every syllable, so a low English-like pace would force half-sentences.
WORD_PACE = {
    'English': 2.2,
    'Vietnamese': 5.2,
    'Indonesian': 2.6,
    'Malay': 2.6,
    'Filipino': 2.8,
    'Spanish': 2.8,
    'French': 2.8,
    'Polish': 2.4,
}
# Characters per second for scripts that do not space words.
CHAR_PACE = {
    'Thai': 6.0,
    # Mora-ish characters. 8 forced telegram kanji that Fish cannot read.
    'Japanese': 10.0,
}
TRANSLATOR_CODES = {
    'English': 'en',
    '简体中文': 'zh-CN',
    '繁体中文': 'zh-TW',
    'Cantonese': 'zh-TW',
    'Japanese': 'ja',
    'Korean': 'ko',
    'Spanish': 'es',
    'French': 'fr',
    'Polish': 'pl',
    'Vietnamese': 'vi',
    'Thai': 'th',
    'Indonesian': 'id',
    'Malay': 'ms',
    'Filipino': 'tl',
}


def strip_language_mark(label):
    key = str(label or 'English').strip()
    if key.endswith(_PARTIAL_MARK):
        key = key[: -len(_PARTIAL_MARK)].strip()
    return key


def split_target_language(label):
    key = strip_language_mark(label)
    return _MAP.get(key, (key, key))


def translation_language(label):
    return split_target_language(label)[0]


def ui_language_label(language):
    lang = translation_language(language)
    for label in TARGET_LANGUAGES:
        if translation_language(label) == lang:
            return label
    return language or 'English'


def tts_language(label):
    return split_target_language(label)[1]


def is_chinese_target(label):
    return translation_language(label) in CHINESE_TRANSLATION_LANGUAGES


def uses_word_budget(label):
    return translation_language(label) in WORD_BUDGET_LANGUAGES or tts_language(label) in WORD_BUDGET_LANGUAGES


def uses_char_budget(label):
    return translation_language(label) in CHAR_BUDGET_LANGUAGES or tts_language(label) in CHAR_BUDGET_LANGUAGES


def spoken_word_pace(label):
    lang = translation_language(label)
    env_key = 'DUBBING_WORDS_PER_SEC' if lang == 'English' else f'DUBBING_{lang.upper()}_WORDS_PER_SEC'
    raw = os.getenv(env_key)
    if raw:
        try:
            return float(raw)
        except (TypeError, ValueError):
            pass
    return float(WORD_PACE.get(lang, 2.4))


def spoken_char_pace(label):
    lang = translation_language(label)
    raw = os.getenv(f'DUBBING_{lang.upper()}_CHARS_PER_SEC')
    if raw:
        try:
            return float(raw)
        except (TypeError, ValueError):
            pass
    return float(CHAR_PACE.get(lang, 6.0))


def needs_translation_refresh(label):
    """Stale-cache path: rewrite from source instead of only shortening."""
    return translation_language(label) in {'English', 'Vietnamese', 'Japanese'}


def required_translation_version(label):
    lang = translation_language(label)
    return max(TRANSLATION_VERSION, LANGUAGE_TRANSLATION_VERSION.get(lang, TRANSLATION_VERSION))


def translator_code(label):
    trans = translation_language(label)
    if trans in TRANSLATOR_CODES:
        return TRANSLATOR_CODES[trans]
    if '中文' in trans:
        return 'zh-CN'
    if trans.lower() in {'english', 'en', 'eng'}:
        return 'en'
    return trans


def _same_lang(a, b, kind='translation'):
    if not a or not b:
        return False
    if kind == 'tts':
        return tts_language(a) == tts_language(b)
    return translation_language(a) == translation_language(b)


def load_dub_meta(folder):
    import json
    import os
    path = os.path.join(folder, 'dub_meta.json')
    if os.path.isfile(path):
        try:
            with open(path, 'r', encoding='utf-8') as handle:
                data = json.load(handle)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    summary_path = os.path.join(folder, 'summary.json')
    if os.path.isfile(summary_path):
        try:
            with open(summary_path, 'r', encoding='utf-8') as handle:
                summary = json.load(handle)
            lang = (summary or {}).get('language')
            if lang:
                return {
                    'translation': translation_language(lang),
                    'tts': tts_language(lang),
                }
        except Exception:
            pass
    return {}


AUDIO_MIX_VERSION = 7
AUDIO_LAYOUT_VERSION = 5
TRANSLATION_VERSION = 7
LANGUAGE_TRANSLATION_VERSION = {
    'English': 11,
    'Vietnamese': 8,
    'Japanese': 11,
}


_LAUGH_CHARS = set('哈呵嘻嘿啊嗯哦呜哇')


def is_asr_junk(text):
    """Repeated-character ASR noise that should not be translated or dubbed."""
    compact = re.sub(r'\s+', '', text or '')
    if len(compact) < 8:
        return False
    if all(ch in _LAUGH_CHARS for ch in compact):
        return False
    collapsed = re.sub(r'(.)\1{2,}', r'\1', compact)
    if len(collapsed) <= 2:
        return True
    best = max((compact.count(ch) for ch in set(compact)), default=0)
    return best >= 8 and best / len(compact) >= 0.8


def drop_asr_junk_lines(transcript):
    kept = []
    dropped = 0
    for item in transcript or []:
        if is_asr_junk((item or {}).get('text')):
            dropped += 1
            continue
        kept.append(item)
    return kept, dropped


def purge_asr_junk_folder(folder):
    """Remove hallucinated repeat-token lines from saved ASR / translation files."""
    import json

    dropped = 0
    for name, indent in (('transcript.json', 4), ('translation.json', 2)):
        path = os.path.join(folder, name)
        if not os.path.isfile(path):
            continue
        try:
            with open(path, 'r', encoding='utf-8') as handle:
                data = json.load(handle)
        except Exception:
            continue
        if not isinstance(data, list):
            continue
        kept, n = drop_asr_junk_lines(data)
        if not n:
            continue
        with open(path, 'w', encoding='utf-8') as handle:
            json.dump(kept, handle, indent=indent, ensure_ascii=False)
        dropped += n
    if dropped:
        try:
            from tools.translation_versions import snapshot_active
            snapshot_active(folder)
        except Exception:
            pass
    return dropped


SOURCE_LANGUAGES = [
    '中文',
    '粵語',
    'English',
    '日本語',
    '越南文',
    'Korean',
    '西班牙文',
    'French',
    '泰文',
    '印尼文',
    '馬來文',
    '菲律賓文',
]
SOURCE_LANGUAGE_CODES = {
    '中文': 'zh',
    '簡體中文': 'zh',
    '简体中文': 'zh',
    '繁體中文': 'zh',
    '繁体中文': 'zh',
    '粵語': 'zh',
    '粤语': 'zh',
    'Cantonese': 'zh',
    'English': 'en',
    'en': 'en',
    '日本語': 'ja',
    'Japanese': 'ja',
    'ja': 'ja',
    '越南文': 'vi',
    'Vietnamese': 'vi',
    'vi': 'vi',
    'Korean': 'ko',
    'ko': 'ko',
    '西班牙文': 'es',
    'Spanish': 'es',
    'es': 'es',
    'French': 'fr',
    'fr': 'fr',
    '泰文': 'th',
    'Thai': 'th',
    'th': 'th',
    '印尼文': 'id',
    'Indonesian': 'id',
    'id': 'id',
    '馬來文': 'ms',
    'Malay': 'ms',
    'ms': 'ms',
    '菲律賓文': 'tl',
    'Filipino': 'tl',
    'tl': 'tl',
}


def asr_language_code(label=None):
    text = (label or '').strip()
    if not text or text.lower() in {'auto', 'none'}:
        return 'zh'
    if text in SOURCE_LANGUAGE_CODES:
        return SOURCE_LANGUAGE_CODES[text]
    stripped = text.replace(_PARTIAL_MARK, '').strip()
    if stripped in SOURCE_LANGUAGE_CODES:
        return SOURCE_LANGUAGE_CODES[stripped]
    if len(text) == 2 and text.isalpha():
        return text.lower()
    return 'zh'


def recorded_demucs_shifts(folder):
    raw = load_dub_meta(folder).get('demucs_shifts')
    if raw is None:
        return 1
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 1


def clear_asr_downstream(folder, keep_bible=True):
    """Wipe transcript and anything that depends on it. Keep vocal stems unless caller deletes them."""
    _remove_path(os.path.join(folder, 'transcript.json'))
    _remove_path(os.path.join(folder, 'asr_review.json'))
    _remove_path(os.path.join(folder, 'asr_repair.json'))
    _remove_path(os.path.join(folder, 'source_bible.json'))
    _remove_path(os.path.join(folder, 'translations'))
    clear_translation_cache(folder, keep_bible=keep_bible)


def save_dub_meta(folder, translation=None, tts=None, mix_version=None, layout_version=None, speakers_locked=None, translation_version=None, demucs_shifts=None, asr_language=None):
    import json
    import os
    meta = load_dub_meta(folder)
    if translation:
        meta['translation'] = translation_language(translation)
    if tts:
        meta['tts'] = tts_language(tts)
    if mix_version is not None:
        meta['mix_version'] = mix_version
    if layout_version is not None:
        meta['layout_version'] = layout_version
    if speakers_locked is not None:
        meta['speakers_locked'] = bool(speakers_locked)
    if translation_version is not None:
        meta['translation_version'] = int(translation_version)
        lang = meta.get('translation')
        if lang:
            versions = dict(meta.get('versions') or {})
            versions[lang] = int(translation_version)
            meta['versions'] = versions
    if demucs_shifts is not None:
        meta['demucs_shifts'] = int(demucs_shifts)
    if asr_language is not None:
        meta['asr_language'] = asr_language_code(asr_language)
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, 'dub_meta.json'), 'w', encoding='utf-8') as handle:
        json.dump(meta, handle, indent=2, ensure_ascii=False)
    return meta


def _remove_path(path):
    import os
    import shutil
    if os.path.isfile(path):
        os.remove(path)
    elif os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)


def titled_video_path(folder, language=None):
    """Explorer deliverable: {folder_basename}_{Language}.mp4."""
    lang = language or load_dub_meta(folder).get('translation') or 'English'
    lang = translation_language(lang)
    base = os.path.basename(os.path.normpath(folder))
    return os.path.join(folder, f'{base}_{lang}.mp4')


def drop_working_video_alias(folder, keep_path=None):
    """Remove leftover video.mp4 so the folder only shows the titled deliverable."""
    working = os.path.join(folder, 'video.mp4')
    keep = os.path.abspath(keep_path or titled_video_path(folder))
    if not os.path.isfile(working):
        return
    if os.path.normcase(os.path.abspath(working)) == os.path.normcase(keep):
        return
    try:
        os.remove(working)
    except OSError:
        pass


def published_video_paths(folder):
    """Finished dub files: leftover video.mp4 plus {foldername}_{Language}.mp4."""
    import os
    paths = [os.path.join(folder, 'video.mp4')]
    try:
        paths.append(titled_video_path(folder))
    except Exception:
        pass
    base = os.path.basename(os.path.normpath(folder))
    try:
        for name in os.listdir(folder):
            if name.startswith(base + '_') and name.lower().endswith('.mp4'):
                paths.append(os.path.join(folder, name))
    except OSError:
        pass
    seen = set()
    unique = []
    for path in paths:
        key = os.path.normcase(os.path.abspath(path))
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def remove_published_videos(folder):
    for path in published_video_paths(folder):
        _remove_path(path)


def clear_tts_cache(folder, keep_video=False):
    import os
    for name in ('audio_combined.wav', 'audio_tts.wav'):
        _remove_path(os.path.join(folder, name))
    if not keep_video:
        for path in published_video_paths(folder):
            _remove_path(path)
    _remove_path(os.path.join(folder, 'wavs'))


def clear_translation_cache(folder, keep_bible=False, language=None):
    import json
    import os
    from tools.translation_versions import lang_key, review_version_path, snapshot_active, version_path

    snapshot_active(folder)
    lang = lang_key(language) if language else None
    if lang:
        _remove_path(version_path(folder, lang))
        _remove_path(review_version_path(folder, lang))
        meta = load_dub_meta(folder)
        if _same_lang(meta.get('translation'), lang, 'translation'):
            _remove_path(os.path.join(folder, 'translation.json'))
            _remove_path(os.path.join(folder, 'translation_review.json'))
        versions = dict(meta.get('versions') or {})
        versions.pop(lang, None)
        meta['versions'] = versions
        if _same_lang(meta.get('translation'), lang, 'translation'):
            meta['translation_version'] = 0
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, 'dub_meta.json'), 'w', encoding='utf-8') as handle:
            json.dump(meta, handle, indent=2, ensure_ascii=False)
    else:
        _remove_path(os.path.join(folder, 'translation.json'))
        _remove_path(os.path.join(folder, 'translation_review.json'))
        save_dub_meta(folder, speakers_locked=False, translation_version=0)
    if not keep_bible:
        _remove_path(os.path.join(folder, 'summary.json'))
    clear_tts_cache(folder)


def translation_cache_ok(folder, target_language):
    from tools.translation_versions import has_version, lang_key

    if not has_version(folder, target_language):
        return False
    meta = load_dub_meta(folder)
    lang = lang_key(target_language)
    versions = meta.get('versions') or {}
    ver = versions.get(lang)
    if ver is None and _same_lang(meta.get('translation'), target_language, 'translation'):
        ver = meta.get('translation_version')
    try:
        return int(ver or 0) >= required_translation_version(target_language)
    except (TypeError, ValueError):
        return False


def mix_cache_ok(folder):
    import os
    if not os.path.isfile(os.path.join(folder, 'audio_combined.wav')):
        return False
    try:
        return int(load_dub_meta(folder).get('mix_version') or 0) >= AUDIO_MIX_VERSION
    except (TypeError, ValueError):
        return False


def layout_cache_ok(folder):
    try:
        return int(load_dub_meta(folder).get('layout_version') or 0) >= AUDIO_LAYOUT_VERSION
    except (TypeError, ValueError):
        return False


def speakers_are_locked(folder):
    return bool(load_dub_meta(folder).get('speakers_locked'))


def tts_cache_ok(folder, target_language):
    import os
    if not os.path.isfile(os.path.join(folder, 'audio_combined.wav')):
        return False
    if not translation_cache_ok(folder, target_language):
        return False
    cached = load_dub_meta(folder).get('tts')
    return _same_lang(cached, target_language, 'tts')
