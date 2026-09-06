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
    if lang == 'Vietnamese':
        return 3
    return 2


_NAME_STOP = {
    'the', 'this', 'that', 'you', 'your', 'and', 'but', 'for', 'with', 'from',
    'have', 'will', 'what', 'when', 'where', 'how', 'why', 'not', 'she', 'he',
    'they', 'his', 'her', 'our', 'its', 'can', 'don', 'are', 'was', 'were',
    'been', 'being', 'had', 'has', 'did', 'does', 'all', 'any', 'even', 'still',
    'just', 'then', 'now', 'yes', 'let', 'get', 'got', 'come', 'came',
}


def _spoken_tokens(text):
    cleaned = re.sub(r'\[[^\]]*\]', ' ', text or '')
    cleaned = cleaned.replace('—', ' ').replace('–', ' ').replace('-', ' ')
    return [part for part in re.findall(r"[A-Za-z0-9']+", cleaned) if part]


def _name_tokens(text):
    names = []
    for word in _spoken_tokens(text):
        if len(word) < 3 or word.lower() in _NAME_STOP:
            continue
        if word[0].isupper():
            names.append(word.lower())
    return names


def _source_asks(text):
    src = (text or '').strip()
    if re.search(r'[？?]', src):
        return True
    return bool(re.search(r'(怎么|怎麼|如何|吗|嗎|么|麼|呢)\s*$', src) or ('怎么' in src) or ('怎麼' in src) or ('如何' in src))


def _cjk_clause_count(text):
    parts = [
        part for part in re.split(r'[，、。；;！!？?]', text or '')
        if re.findall(r'[\u4e00-\u9fff]', part)
    ]
    return len(parts)


def _looks_like_chant(text):
    """Short parallel beats (锤玉成云，碎波成霜，以意易容形), not a long prose sentence."""
    parts = [
        part.strip()
        for part in re.split(r'[，、。；;！!？?]', text or '')
        if re.findall(r'[\u4e00-\u9fff]', part)
    ]
    if len(parts) < 3:
        return False
    sizes = [len(re.findall(r'[\u4e00-\u9fff]', part)) for part in parts]
    if max(sizes) > 6 or min(sizes) < 3:
        return False
    return not re.search(r'[的了着過过]', text or '')


def _is_technique_source(text):
    src = text or ''
    return bool(
        re.search(r'[·・]', src)
        or re.search(r'[剑劍诀訣阵陣]', src)
        or _looks_like_chant(src)
    )


def _latin_source(text):
    src = (text or '').strip()
    if not src or re.search(r'[\u4e00-\u9fff]', src):
        return False
    return len(re.findall(r'[A-Za-z]', src)) >= 4


def _foreign_asr_source(text):
    src = (text or '').strip()
    if not src:
        return False
    if _latin_source(src):
        return True
    if re.search(r'[\u4e00-\u9fff]', src):
        return False
    return bool(re.search(r'[\u3040-\u30ff]', src))


def _mostly_cjk_episode(transcript):
    lines = [line for line in (transcript or []) if (line.get('text') or '').strip()]
    if not lines:
        return False
    cjk = sum(1 for line in lines if re.search(r'[\u4e00-\u9fff]', line.get('text') or ''))
    need = 3 if len(lines) < 8 else int(0.6 * len(lines))
    return cjk >= max(need, 1)


def _glued_two_mouths(text):
    return bool(re.search(r'\s/\s', text or ''))


def _needs_empty_vocal(line):
    src = ((line or {}).get('text') if isinstance(line, dict) else line) or ''
    src = src.strip()
    dub = ((line or {}).get('translation') if isinstance(line, dict) else '') or ''
    if dub.strip() or not src or is_asr_junk(src):
        return False
    from tools.vocal_particles import is_particle_card
    if is_particle_card(src):
        return False
    compact = re.sub(r'[\s,，。！？!?、…]+', '', src)
    return bool(re.fullmatch(r'[\u4e00-\u9fff]{1,2}', compact))


def _technique_crushed(src, dub):
    if not _is_technique_source(src):
        return False
    words = _spoken_tokens(dub)
    chars = len(re.findall(r'[\u4e00-\u9fff]', src or ''))
    clauses = _cjk_clause_count(src)
    spoken = dub or ''
    if clauses >= 3 and ('—' in spoken or '–' in spoken) and 0 < len(words) <= clauses * 2:
        return True
    if clauses >= 3 and len(words) < clauses + 1:
        return True
    return bool(re.search(r'[·・]', src or '')) and chars >= 10 and 0 < len(words) <= 4


def _invents_other_language(src, sug):
    if not _latin_source(src):
        return False
    src_toks = {word.lower() for word in _spoken_tokens(src)}
    sug_toks = {word.lower() for word in _spoken_tokens(sug)}
    if len(sug_toks) < 2:
        return False
    return not (src_toks & sug_toks)


_ELLIPSIS_END_RE = re.compile(r'(…+|\.{3,}|。。。)\s*$')
_STAMMER_FLIP_RE = re.compile(r'(…+|\.{3,}|。。。)\s*[不没沒][!！]?$')
_GUESS_WORD_RE = re.compile(
    r'\b(somehow|anyway|i guess|i suppose|or something)\b',
    re.I,
)
_VOCATIVE_CMD_RE = re.compile(r'[\u4e00-\u9fff]{2,}[，,]\s*[上来去滚杀]')
_FIRST_MOUTH_RE = re.compile(r'[我咱俺]|好啊|告诉|那我')
_TOLD_RE = re.compile(r"\b(i('ll| will)? tell|i('ll| will)? confess|fine,? i)\b", re.I)
_ORDER_RE = re.compile(r'\b(hall|move!?|guards?|seize|take (him|her)|bring (him|her))\b', re.I)
_STAMMER_KEEP_RE = re.compile(r'\b(y-?yes|n-?no|no)\b', re.I)
_CONFESS_RE = re.compile(
    r"\b(i('ll| will)? tell|i('ll| will)? confess|fine,? i|admit|alright|okay)\b",
    re.I,
)


def _source_ellipsis(text):
    return bool(_ELLIPSIS_END_RE.search((text or '').rstrip()))


def _source_stammer_flip(text):
    """是、是、是……不！ — agree, trail off, then reverse. Not a finished thought."""
    return bool(_STAMMER_FLIP_RE.search((text or '').strip()))


def _rewrites_stammer_flip(src, dub):
    if not _source_stammer_flip(src):
        return False
    spoken = (dub or '').strip()
    if not spoken:
        return False
    if _CONFESS_RE.search(spoken):
        return True
    if _STAMMER_KEEP_RE.search(spoken) and re.search(r'(\.\.\.|…|—)', spoken):
        return False
    if _ELLIPSIS_END_RE.search(spoken) or spoken.endswith(('—', '--')):
        return False
    return bool(re.search(r'[.!?]"?$', spoken))


def _norm_latin(text):
    return re.sub(r'[^a-z0-9]+', ' ', (text or '').lower()).strip()


def _rewrites_foreign_asr(src, dub):
    if not _foreign_asr_source(src):
        return False
    left, right = _norm_latin(src), _norm_latin(dub)
    if not right or left == right:
        return False
    return True


def _guesses_unfinished(src, dub):
    if _rewrites_stammer_flip(src, dub):
        return True
    if not _source_ellipsis(src):
        return False
    spoken = (dub or '').strip()
    if not spoken:
        return False
    if _GUESS_WORD_RE.search(spoken):
        return True
    if re.search(r'(\.\.\.|…)\s+\S', spoken):
        return True
    if _ELLIPSIS_END_RE.search(spoken) or spoken.endswith(('—', '--')):
        return False
    return bool(re.search(r'[.!?]"?$', spoken))


def should_not_tighten(text):
    """Paper/wav shorten must not touch trail-offs, stammed flips, chants, or foreign ASR."""
    src = (text or '').strip()
    if not src:
        return False
    if _foreign_asr_source(src):
        return True
    if _source_ellipsis(src) or _source_stammer_flip(src):
        return True
    if _looks_like_chant(src) or re.search(r'[·・]', src):
        return True
    return False


def restores_voice(source, current, suggest, target_language='English'):
    """True when suggest undoes a guessed, crushed, or glued shorten."""
    if is_chinese_target(target_language):
        return False
    src = (source or '').strip()
    old = (current or '').strip()
    sug = (suggest or '').strip()
    if not sug or sug == old:
        return False
    if _guesses_unfinished(src, old) and not _guesses_unfinished(src, sug):
        return True
    if _technique_crushed(src, old) and not _technique_crushed(src, sug):
        return True
    if _rewrites_foreign_asr(src, old) and not _rewrites_foreign_asr(src, sug):
        return True
    if _speaks_both_mouths(src, old) and not _speaks_both_mouths(src, sug) and not _glued_two_mouths(sug):
        return True
    if _sense_broken(src, old) and not _sense_broken(src, sug):
        return True
    return False


def _two_mouth_source(src):
    parts = [
        part.strip()
        for part in re.split(r'[。！？!?]', src or '')
        if re.findall(r'[\u4e00-\u9fff]', part)
    ]
    if len(parts) < 2:
        return False
    return bool(_VOCATIVE_CMD_RE.search(parts[-1]) and _FIRST_MOUTH_RE.search(''.join(parts[:-1])))


def _speaks_both_mouths(src, dub):
    if not _two_mouth_source(src):
        return False
    spoken = dub or ''
    sentences = [part for part in re.split(r'[.!?]+', spoken) if part.strip()]
    if len(sentences) >= 2:
        return True
    return bool(_TOLD_RE.search(spoken) and _ORDER_RE.search(spoken))


_EXPERIENCE_VERB_RE = re.compile(r'去过|去過|看过|看過|听过|聽過|到过|到過|见过|見過')
_QUESTION_VERB_EN = re.compile(
    r'\b(seen|been|gone|go|heard|hear|visited|visit|ever|know|met|meet|reached|reach|entered|enter)\b',
    re.I,
)
_AIR_NEG_SRC = re.compile(r'空气中没|空氣中沒|空气中沒|空氣中没|[中里裡]没有|[中里裡]沒有')
_PLACE_EN = re.compile(r'\b(in the air|in the wind|around here|here)\b', re.I)
_SPEAR_CHAR = re.compile(r'[枪槍]')
_MODERN_GUN_SRC = re.compile(
    r'手枪|手槍|开枪|開槍|子弹|子彈|枪械|槍械|步枪|步槍|'
    r'火枪|火槍|鸟枪|鳥槍|猎枪|獵槍|机枪|機槍|烟枪|煙槍|'
    r'枪决|槍決|枪毙|槍斃|枪击|槍擊|水枪|水槍'
)
_GUN_EN = re.compile(r'\bguns?\b', re.I)
_REALM_RIVER_SRC = re.compile(
    r'(?:寻常|尋常|平常|普通)?河道\s*[巔巅]?峰|'
    r'河道\s*(?:强者|期|境|大成)|'
    r'(?:寻常|尋常|平常|普通)河道'
)
_CHANNEL_EN = re.compile(r'\b(channels?|waterways?|rivers?)\b', re.I)
_TITLED_HERO_SRC = re.compile(r'[称稱][\u4e00-\u9fff]{2,8}俊[傑杰]')
_GENERIC_HERO_NAMES = {'hero', 'heroes', 'heaven', 'earth', 'unashamed'}


def _drops_question_verb(src, dub):
    if not _source_asks(src) or not _EXPERIENCE_VERB_RE.search(src or ''):
        return False
    return not bool(_QUESTION_VERB_EN.search(dub or ''))


def _drops_location_negation(src, dub):
    if not _AIR_NEG_SRC.search(src or ''):
        return False
    spoken = dub or ''
    if _PLACE_EN.search(spoken):
        return False
    first = re.split(r'[.!?]', spoken.strip(), 1)[0].strip()
    if re.match(r'^no\b', first, re.I) and not re.search(r'\b(in|here|around)\b', first, re.I):
        return bool(re.search(r'空气|空氣|这里|這裡', src or ''))
    return False


def _spear_as_gun(src, dub):
    if _MODERN_GUN_SRC.search(src or '') or not _GUN_EN.search(dub or ''):
        return False
    return bool(_SPEAR_CHAR.search(src or ''))


def _realm_as_waterway(src, dub):
    if not _REALM_RIVER_SRC.search(src or ''):
        return False
    return bool(_CHANNEL_EN.search(dub or ''))


def _drops_heaven_earth(src, dub):
    if '天地' not in (src or ''):
        return False
    spoken = dub or ''
    if not re.search(r'\bheavens?\b', spoken, re.I):
        return False
    return not bool(re.search(r'\bearth\b', spoken, re.I))


def _crushes_titled_hero(src, dub):
    if not _TITLED_HERO_SRC.search(src or ''):
        return False
    spoken = (dub or '').strip()
    if not re.match(r'^\s*(?:an? |the )?hero(?:es)?\b', spoken, re.I):
        return False
    if re.search(r'\bhero(?:es)?\s+(of|from|in)\b', spoken, re.I):
        return False
    leftover = set(_name_tokens(spoken)) - _GENERIC_HERO_NAMES
    return not leftover


def sense_issue(src, dub):
    """Chinese issue string when meaning was rewritten into the wrong thing."""
    if _drops_question_verb(src, dub):
        return '問句被收成電報，動詞沒了'
    if _drops_location_negation(src, dub):
        return '否定句丟了處所'
    if _spear_as_gun(src, dub):
        return '冷兵器的槍被譯成 gun'
    if _realm_as_waterway(src, dub):
        return '境界近音被譯成河道'
    if _drops_heaven_earth(src, dub):
        return '天地被收成只有 Heaven'
    if _crushes_titled_hero(src, dub):
        return '稱〇俊傑被收成 Hero'
    return ''


def _sense_broken(src, dub):
    return bool(sense_issue(src, dub))


def glossary_forced_on_junk(src, dub, glossary_pairs):
    """True when a short ASR hash was replaced by a glossary name not in the source."""
    han = re.findall(r'[\u4e00-\u9fff]', src or '')
    if not (3 <= len(han) <= 8):
        return None
    spoken = re.sub(r'[.!?…]+$', '', (dub or '').strip())
    if not spoken:
        return None
    for cn, dst in glossary_pairs or []:
        if not dst or len(cn) < 2:
            continue
        if cn in (src or ''):
            continue
        if spoken.lower() == dst.strip().lower():
            return cn, dst
    return None


def suggest_wrecks_voice(source, current, suggest, target_language='English'):
    """True when a rewrite strips speech-act, names, or turns into a bark."""
    if is_chinese_target(target_language):
        return False
    src = (source or '').strip()
    old = (current or '').strip()
    sug = (suggest or '').strip()
    if not sug or sug == old:
        return False
    from tools.vocal_particles import is_particle_card, particle_translation

    if is_particle_card(src):
        compact = ''.join(ch for ch in src if ch.strip() and ch not in '，,。！？!?、…')
        spoken = ''.join(_spoken_tokens(sug)).lower()
        if compact in {'嘿', '嘿嘿', '哈哈', '嘻嘻'} and spoken in {'ha', 'hah'}:
            return True
        mapped = particle_translation(src, target_language)
        if mapped and spoken and spoken != ''.join(_spoken_tokens(mapped)).lower() and spoken in {'ha', 'hah'}:
            return True
        return False

    if restores_voice(src, old, sug, target_language):
        return False

    src_chars = len(re.findall(r'[\u4e00-\u9fff]', src))
    sug_words = _spoken_tokens(sug)
    old_words = _spoken_tokens(old)
    if _source_asks(src) and '?' not in sug and '？' not in sug:
        return True
    if src_chars >= 5 and len(sug_words) <= 1:
        return True
    if old and len(old_words) >= 4 and len(sug_words) <= 2:
        return True
    old_names = set(_name_tokens(old))
    sug_names = set(_name_tokens(sug))
    if old_names and not (old_names & sug_names) and src_chars >= 4:
        return True
    if len(old_names) >= 2 and len(old_names & sug_names) < 2 and src_chars >= 6:
        return True
    if not _source_stammer_flip(src):
        clauses = _cjk_clause_count(src)
        if clauses >= 3 and len(sug_words) < clauses + 1:
            if not (_looks_like_chant(src) and len(sug_words) >= max(len(old_words), 1)):
                return True
    if _is_technique_source(src):
        dropped = old_names - sug_names - {'swords', 'sword', 'art', 'arts', 'form', 'system'}
        if dropped:
            return True
        if src_chars >= 10 and old and len(sug_words) <= 4 and len(sug_words) < len(old_words):
            return True
    if _glued_two_mouths(sug) and not _glued_two_mouths(old):
        return True
    if _invents_other_language(src, sug):
        return True
    if _rewrites_foreign_asr(src, sug):
        return True
    if _guesses_unfinished(src, sug):
        return True
    if _speaks_both_mouths(src, sug):
        return True
    if _sense_broken(src, sug):
        return True
    return False

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
        mid_cjk = (
            end < n
            and '\u4e00' <= source[end - 1] <= '\u9fff'
            and '\u4e00' <= source[end] <= '\u9fff'
        )
        window = 12 if mid_cjk else 6
        for j in range(end, min(n, end + window)):
            if source[j - 1:j] in '。！？，、,.!? ':
                snapped = j
                break
        if snapped is None:
            for j in range(end, max(cursor + 1, end - window), -1):
                if source[j - 1:j] in '。！？，、,.!? ':
                    snapped = j
                    break
        if snapped is None and mid_cjk:
            snapped = n if i >= len(parts) - 2 else end
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
    if _guesses_unfinished(text, cleaned):
        return (
            'The Chinese trails off or flips after ……. Keep the stammer or the named thing '
            'and trail off. Do not confess or finish the thought. Output only the line.'
        )
    if _technique_crushed(text, cleaned):
        return (
            'Keep every chant/technique clause. Cut filler only. '
            'Do not crush it into an em-dash compound. Output only the line.'
        )
    if _rewrites_foreign_asr(text, cleaned):
        return (
            'Copy the source as-is. Do not rewrite recognized English or Latin. '
            'Output only the line.'
        )
    if _speaks_both_mouths(text, cleaned):
        return (
            'This card has two speakers. Translate only the first speaker. '
            "Do not speak the other mouth's command. Output only the line."
        )
    if _sense_broken(text, cleaned):
        return (
            'Keep the verb, the place, and the real weapon or realm word. '
            'Do not telegraph. A spear is not a gun. A cultivation realm is not a river. '
            'Heaven and Earth stay together. A titled hero keeps the place name. '
            'Output only the line.'
        )
    if _source_ellipsis(text) or _source_stammer_flip(text):
        return None
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
        return None
    if lang == 'Japanese':
        if compact >= 6 and HAN_RE.search(cleaned) and not KANA_RE.search(cleaned):
            return (
                'Kanji-only lines are read as Chinese. Add hiragana or katakana. '
                'Output only the Japanese line.'
            )
        if JA_CN_LEFT_RE.search(cleaned):
            return (
                'Do not leave Chinese particles. Use the outline glossary. '
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
        return 1.1
    if _SYSTEM_UI_RE.search(text):
        return 1.08
    if '心想' in text or text.startswith('（') or text.startswith('('):
        return 1.05
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
                f'Keep 嗯/呵/哼/嘿嘿 as Ừ/Hề/Hừ/He he. '
                f'Do not end with … unless the source is unfinished:"{text}"'
            )
        if lang == 'English':
            return (
                f'{note}Translate the whole line into spoken English in {duration:.1f}s, '
                f'max {budget} words. Fish TTS is slow; do not overflow the slot. '
                f'Use the outline names. Keep meaning, emotion, and 啊吧呢呀嗯呵哼. '
                f'If the Chinese is angry, scared, or mocking, do not sound polite. '
                f'If the Chinese trails off with ……, keep the named thing and trail off; '
                f'do not add somehow or finish the thought. '
                f'If this card glued two speakers, translate only the first mouth. '
                f'If the line is already Latin letters or kana, copy it; do not clean it. '
                f'Do not add names that are not in this line. Do not chop into fragments:"{text}"'
            )
        return f'{note}Translate in {duration:.1f}s, max {budget} {lang} words:"{text}"'
    if uses_char_budget(target_language):
        budget = max(4, int(round(_spoken_char_budget(duration, target_language) * budget_scale)))
        if lang == 'Japanese':
            return (
                f'{note}Translate the whole line into spoken Japanese in {duration:.1f}s, '
                f'up to {budget} characters. Mixed Japanese, not Chinese calques. '
                f'Use the outline glossary for names (katakana). Do not calque Chinese compounds. '
                f'Never write 0/10, %, or +. '
                f'Keep 嗯/呵/哼/嘿嘿 as ん/ふふ/ふん/へへ. Keep every clause:"{text}"'
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
    """Keep one source card as one dubbed card.

    Target-language periods must not re-slice the Chinese. Long source
    lines are split before translation (ASR polish / split_long_segments).
    """
    del use_char_based_end, target_language
    output_data = []
    for item in translation or []:
        line = dict(item)
        if line.get('orig_start') is None:
            line['orig_start'] = line.get('start')
        if line.get('orig_end') is None:
            line['orig_end'] = line.get('end')
        output_data.append(line)
    return output_data

