# -*- coding: utf-8 -*-
"""After ASR: write a Chinese outline, then let the model repair the cards.

No show-specific word lists. The outline is the only episode knowledge.
"""
import json
import os
import re
from datetime import datetime, timezone
from difflib import SequenceMatcher

from loguru import logger

from tools.asr_source import _glue_source, _is_particle_only
from tools.target_language import is_asr_junk

REPAIR_NAME = 'asr_repair.json'
SOURCE_BIBLE_NAME = 'source_bible.json'
REPAIR_VERSION = 4
_BATCH = 36
_OVERLAP = 4
_AUDIT_BATCH = 200
_AUDIT_OVERLAP = 12
_MAX_AUDIT_FIXES = 120
_MAX_ABSORB = 6
_MAX_NEW_SPEAKERS = 4
_HAN_RE = re.compile(r'[\u4e00-\u9fff]')
_HAN_KEEP = re.compile(r'[^\w\u4e00-\u9fff]+', flags=re.UNICODE)
_SPEAKER_ID_RE = re.compile(r'^SPEAKER_[A-Z][A-Z0-9]{0,3}$')
_REALM_RIVER_FIXES = (
    (re.compile(r'(寻常|尋常|平常|普通)河道'), r'\1合道'),
    (re.compile(r'河道(?=\s*[巔巅]?峰)'), '合道'),
    (re.compile(r'河道(?=\s*强者)'), '合道'),
    (re.compile(r'河道(?=\s*期)'), '合道'),
    (re.compile(r'河道(?=\s*境)'), '合道'),
    (re.compile(r'河道(?=\s*大成)'), '合道'),
)


def repair_path(folder):
    return os.path.join(folder, REPAIR_NAME)


def source_bible_path(folder):
    return os.path.join(folder, SOURCE_BIBLE_NAME)


def already_repaired(folder):
    path = repair_path(folder)
    if not os.path.isfile(path):
        return False
    try:
        with open(path, 'r', encoding='utf-8') as handle:
            data = json.load(handle)
        return bool(
            isinstance(data, dict)
            and data.get('applied')
            and int(data.get('version') or 0) >= REPAIR_VERSION
        )
    except Exception:
        return False


def ensure_repaired_transcript(folder, transcript=None, method='OpenAI', progress_callback=None, force=False):
    """Load or run outline + repair. Writes transcript.json when it changes cards."""
    path = os.path.join(folder, 'transcript.json')
    if transcript is None:
        if not os.path.isfile(path):
            return transcript
        with open(path, 'r', encoding='utf-8') as handle:
            transcript = json.load(handle)
    if not isinstance(transcript, list) or not transcript:
        return transcript
    if not force and already_repaired(folder):
        from tools.line_roles import reassign_addressed_you_lines
        moved = reassign_addressed_you_lines(transcript)
        homophones = apply_realm_homophone_fixes(transcript)
        if moved or homophones:
            os.makedirs(folder, exist_ok=True)
            with open(path, 'w', encoding='utf-8') as handle:
                json.dump(transcript, handle, indent=4, ensure_ascii=False)
            if moved:
                logger.info(f'語意講者規則改了 {moved} 句：{folder}')
            if homophones:
                logger.info(f'境界近音河道改回合道 {homophones} 句：{folder}')
        else:
            logger.info(f'辨識後修稿已做過，略過：{folder}')
        return transcript
    return repair_asr_script(
        folder, transcript, method=method, progress_callback=progress_callback,
    )


def repair_asr_script(folder, transcript, method='OpenAI', progress_callback=None):
    """Build a Chinese outline, then repair ASR cards against it."""
    from tools.job_control import JobStopped

    method = _resolve_method(method)
    lines = [dict(item) for item in (transcript or [])]
    if not lines:
        return lines
    from tools.line_roles import promote_bgm_speech
    rescued = promote_bgm_speech(lines)
    if rescued:
        logger.info(f'BGM 裡的人聲改回對白 {rescued} 句')
    _progress(progress_callback, '辨識後先寫中文大綱…')
    bible = ensure_source_bible(folder, lines, method)
    before = len(lines)
    changed = skip_n = merge_n = moved_n = 0
    new_ids = []
    ai_error = None
    if not method:
        logger.info('沒有語言模型，辨識後只保留原句')
    else:
        try:
            changed, skip_n, merge_n, moved_n, new_ids, lines = _ai_repair(
                folder, lines, bible, method, progress_callback=progress_callback,
            )
        except Exception as exc:
            if isinstance(exc, JobStopped):
                raise
            ai_error = str(exc)
            logger.warning(f'辨識後修稿失敗，沿用原句：{exc}')

    homophones = apply_realm_homophone_fixes(lines)
    if homophones:
        logger.info(f'境界近音河道改回合道 {homophones} 句')
        changed += homophones

    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, 'transcript.json'), 'w', encoding='utf-8') as handle:
        json.dump(lines, handle, indent=4, ensure_ascii=False)
    if moved_n or new_ids:
        try:
            from tools.asr import generate_speaker_audio
            generate_speaker_audio(folder, lines)
        except Exception as exc:
            logger.warning(f'修稿後重切講者音檔略過：{exc}')
    report = {
        'applied': ai_error is None or changed or skip_n or merge_n or moved_n or rescued,
        'version': REPAIR_VERSION,
        'applied_at': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'before_count': before,
        'after_count': len(lines),
        'rewritten': changed,
        'skipped': skip_n,
        'merged': merge_n,
        'speakers': moved_n,
        'rescued_narration': rescued,
        'new_speakers': new_ids,
        'ai_error': ai_error,
    }
    with open(repair_path(folder), 'w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    _progress(
        progress_callback,
        f'辨識後修稿完成：改 {changed} 句、講者 {moved_n} 句、新講者 {", ".join(new_ids) or "無"}、'
        f'略過 {skip_n} 句、併 {merge_n} 句，{before} → {len(lines)} 張卡片',
    )
    return lines


def ensure_source_bible(folder, transcript, method='OpenAI'):
    path = source_bible_path(folder)
    if os.path.isfile(path):
        try:
            with open(path, 'r', encoding='utf-8') as handle:
                data = json.load(handle)
            if isinstance(data, dict) and (data.get('outline') or data.get('summary')):
                return data
        except Exception:
            pass
    if not method:
        return {}
    from tools.translation_bible import _load_folder_info, build_dubbing_bible

    info = _load_folder_info(folder)
    bible = build_dubbing_bible(info, transcript, '简体中文', method)
    if not bible:
        return {}
    os.makedirs(folder, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(bible, handle, indent=2, ensure_ascii=False)
    return bible


def _resolve_method(method):
    method = (method or 'OpenAI').strip() or 'OpenAI'
    if method in {'OpenAI', 'LLM', '阿里云-通义千问', 'Ernie', 'Ollama'}:
        if method == 'OpenAI' and not (os.getenv('OPENAI_API_KEY') or '').strip():
            return None
        return method
    if (os.getenv('OPENAI_API_KEY') or '').strip():
        return 'OpenAI'
    return None


def _progress(progress_callback, message):
    logger.info(message)
    if progress_callback:
        progress_callback(message)


def fix_realm_homophones(text):
    """合道 heard as 河道 next to a realm word is a realm, not a river."""
    out = text or ''
    for pattern, repl in _REALM_RIVER_FIXES:
        out = pattern.sub(repl, out)
    return out


def apply_realm_homophone_fixes(transcript):
    changed = 0
    for line in transcript or []:
        src = line.get('text') or ''
        fixed = fix_realm_homophones(src)
        if fixed != src:
            line['text'] = fixed
            changed += 1
    return changed


def _han_compact(text):
    return _HAN_KEEP.sub('', text or '')


def _han_n(text):
    return len(_HAN_RE.findall(text or ''))


def _duration(line):
    return max(0.0, float(line.get('end') or 0) - float(line.get('start') or 0))


def _known_speakers(transcript):
    return sorted({
        str(line.get('speaker') or '')
        for line in transcript or []
        if str(line.get('speaker') or '')
    })


def _next_free_speaker_ids(known, count=3):
    used = {str(item) for item in (known or []) if item}
    out = []
    for code in range(ord('A'), ord('Z') + 1):
        name = f'SPEAKER_{chr(code)}'
        if name in used:
            continue
        out.append(name)
        if len(out) >= count:
            break
    return out


def _is_allowed_speaker_id(name, known):
    from tools.line_roles import SPEAKER_BGM, SPEAKER_NARR, SPEAKER_SYS
    name = str(name or '').strip()
    if not name or name == SPEAKER_BGM:
        return False
    if name in {SPEAKER_SYS, SPEAKER_NARR, 'NARRATOR'} or name in known:
        return True
    return bool(_SPEAKER_ID_RE.match(name))


def _context_blob(bible, transcript):
    from tools.translation_bible import bible_context

    parts = [bible_context(bible) or '']
    parts.append(' '.join((line.get('text') or '') for line in transcript[:8]))
    parts.append(' '.join((line.get('text') or '') for line in transcript[-8:]))
    return '\n'.join(part for part in parts if part)


def _rewrite_ok(old, new, context, duration):
    src = (old or '').strip()
    dst = (new or '').strip()
    if not dst or dst == src:
        return False
    if _han_n(src) <= 3 and duration < 0.55:
        return False
    old_h = _han_compact(src)
    new_h = _han_compact(dst)
    if not new_h:
        return False
    if len(new_h) > max(int(len(old_h) * 2.2), len(old_h) + 10):
        return False
    extra = sum(1 for ch in new_h if '\u4e00' <= ch <= '\u9fff' and ch not in old_h and ch not in context)
    if extra > 4:
        return False
    if extra <= 2:
        return True
    return SequenceMatcher(None, old_h, new_h).ratio() >= 0.5


def _join_legal(transcript, host, absorbed):
    from tools.translation_review import _card_t0, _card_t1

    if host < 0 or host >= len(transcript) or not absorbed:
        return False
    if len(absorbed) > _MAX_ABSORB:
        return False
    if absorbed != list(range(host + 1, host + 1 + len(absorbed))):
        return False
    speaker = str(transcript[host].get('speaker') or '')
    prev = host
    for index in absorbed:
        if index < 0 or index >= len(transcript):
            return False
        if str(transcript[index].get('speaker') or '') != speaker:
            return False
        left = (transcript[prev].get('text') or '').strip()
        right = (transcript[index].get('text') or '').strip()
        if not left or not right:
            return False
        if _is_particle_only(left) or _is_particle_only(right):
            return False
        if is_asr_junk(left) or is_asr_junk(right):
            return False
        if _card_t0(transcript[index]) - _card_t1(transcript[prev]) > 1.2:
            return False
        prev = index
    return True


def _parse_fixes(raw, allowed):
    from tools.translation_bible import _extract_json_object
    from tools.translation_review import _merge_span, _parse_ai_findings

    try:
        data = _extract_json_object(raw)
    except Exception:
        return _parse_ai_findings(raw, allowed)
    items = data.get('fixes') or data.get('findings') or []
    out = []
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            index = int(item.get('index'))
        except (TypeError, ValueError):
            continue
        if index not in allowed:
            continue
        row = dict(item)
        row['index'] = index
        span = _merge_span(row)
        if span is None and row.get('merge') not in (None, [], ''):
            continue
        out.append(row)
    return out


def _format_card(index, line):
    from tools.translation_review import _card_t0, _card_t1

    return (
        f'#{index} {line.get("speaker") or ""} '
        f'{_card_t0(line):.2f}-{_card_t1(line):.2f} | '
        f'{(line.get("text") or "").strip()}'
    )


def _speaker_authority_rules(known, extras):
    extra_note = ', '.join(extras) if extras else '(none left)'
    known_note = ', '.join(known) or '(none)'
    return (
        'Diarization only clustered similar voices. It is often wrong. '
        'Reassign speakers from meaning: who is addressed, who would say this, '
        'and the roles in the outline / voices. '
        f'Existing ids: {known_note}. '
        'When wording + nearby cards + outline show this cannot be the labeled speaker, '
        'you MUST set speaker to the correct existing id, or a new walk-on id from this '
        f'free list, in order: {extra_note}. Reuse the same new id for the same walk-on. '
        'Must-fix cases include: the addressee is the labeled speaker (they would not say '
        '"you" to themselves); a companion/buyer/tease line glued onto a salesperson or clerk; '
        'someone pointing at a lead with "you this X" / an insult / a challenge, so the speaker '
        'cannot be that lead. '
        'You MUST move a mis-attributed line OFF a lead onto another lead or a walk-on '
        'when the line is clearly not that lead talking. '
        'Do not swap two leads\' entire IDs just because they sound similar — that is different '
        'from correcting individual mislabeled cards. '
        'Do not invent a role the outline does not mention. '
        'Use SPEAKER_SYS only for on-screen system or UI notices. '
        'Use SPEAKER_NARR for narrator / 旁白 / TV or livestream speech that is not a lead. '
    )


def _parse_speaker_fixes(raw, allowed):
    out = []
    for item in _parse_fixes(raw, allowed):
        if not str(item.get('speaker') or '').strip():
            continue
        out.append(item)
    return out


def _apply_speaker_item(transcript, item, allowed_speakers, created):
    from tools.line_roles import SPEAKER_NARR
    from tools.translation_review import _merge_span

    try:
        index = int(item.get('index'))
    except (TypeError, ValueError):
        return 0
    if index < 0 or index >= len(transcript):
        return 0
    new_spk = str(item.get('speaker') or '').strip()
    if new_spk == 'NARRATOR':
        new_spk = SPEAKER_NARR
    if not new_spk or not _is_allowed_speaker_id(new_spk, allowed_speakers):
        return 0
    if new_spk not in allowed_speakers:
        if len(created) >= _MAX_NEW_SPEAKERS:
            return 0
        allowed_speakers.add(new_spk)
        created.append(new_spk)
    moved = 0
    targets = [index]
    span_ids = _merge_span(item)
    if span_ids:
        targets.extend(span_ids[1])
    for target in targets:
        if target < 0 or target >= len(transcript):
            continue
        if new_spk != str(transcript[target].get('speaker') or ''):
            transcript[target]['speaker'] = new_spk
            moved += 1
    return moved


def _speaker_audit(transcript, bible, method, allowed_speakers, created, progress_callback=None):
    from tools.job_control import check_stop
    from tools.translation_backends import llm_translate
    from tools.translation_bible import bible_context

    if not transcript:
        return 0, created, transcript
    outline = bible_context(bible) or '(no outline yet)'
    known = _known_speakers(transcript)
    extras = _next_free_speaker_ids(known, 3)
    system = (
        'You audit speaker labels on Chinese ASR cards for a dubbed episode. '
        + _speaker_authority_rules(known, extras)
        + 'Do not invent people, places, numbers, or plot. '
        'Do not change text, merge cards, or skip. Speaker only. '
        'Only list cards whose speaker must change. speaker is required. '
        'Output JSON only: {"fixes":[{"index":0,"speaker":"SPEAKER_A"}]}'
    )
    if len(transcript) <= _AUDIT_BATCH:
        starts = [0]
    else:
        starts = list(range(0, len(transcript), _AUDIT_BATCH - _AUDIT_OVERLAP)) or [0]
    batches = len(starts)
    fixes = []
    for batch_i, start in enumerate(starts, 1):
        end = min(len(transcript), start + _AUDIT_BATCH)
        if start >= end:
            continue
        allowed = set(range(start, end))
        if batches == 1:
            _progress(progress_callback, '辨識後覆核講者（整集語意）…')
        else:
            _progress(
                progress_callback,
                f'辨識後覆核講者 {batch_i}/{batches}（卡片 {start}–{end - 1}）…',
            )
        payload = [
            'Speaker-only audit. Read the wording, nearby cards, and outline. '
            'Reassign every card that cannot be the labeled speaker. Do not invent roles.',
            outline,
            '',
        ]
        payload.extend(_format_card(index, transcript[index]) for index in range(start, end))
        check_stop()
        raw = llm_translate(
            method,
            [
                {'role': 'system', 'content': system},
                {'role': 'user', 'content': '\n'.join(payload)},
            ],
        )
        fixes.extend(_parse_speaker_fixes(raw, allowed))

    by_index = {}
    for item in fixes:
        by_index[int(item['index'])] = item
    items = list(by_index.values())[:_MAX_AUDIT_FIXES]
    moved = 0
    for item in items:
        moved += _apply_speaker_item(transcript, item, allowed_speakers, created)
    if moved:
        logger.info(f'講者語意覆核改了 {moved} 句')
    return moved, created, transcript


def _ai_repair(folder, transcript, bible, method, progress_callback=None):
    from tools.job_control import check_stop
    from tools.translation_backends import llm_translate
    from tools.translation_bible import bible_context
    from tools.translation_review import _join_lines, _merge_span

    context = _context_blob(bible, transcript)
    outline = bible_context(bible) or '(no outline yet)'
    known = _known_speakers(transcript)
    extras = _next_free_speaker_ids(known, 3)
    system = (
        'You repair Chinese ASR subtitle cards before they are translated. '
        'You have an episode outline written from these cards. '
        'Fix misheard words when the outline and nearby cards make the intended word clear. '
        'Cultivation realm homophones (合道 heard as 河道) go back to the realm word, not a river. '
        'Join consecutive SAME-speaker cards that are one spoken sentence. '
        'Set skip=true only for sung lyrics or unintelligible noise. Leave those cards empty. '
        'Keep speech that sits in the music bed: opening TV, livestream ads, recap, narrator. '
        'Those must be translated. Do not skip them as radio, BGM, or background. '
        + _speaker_authority_rules(known, extras)
        + 'Do not invent people, places, numbers, or plot. '
        'Do not turn a 1-3 character stump into a name. Skip it instead. '
        'index = card to keep. merge = next consecutive indices to absorb. '
        'text = repaired Chinese. speaker = corrected id when diarization is wrong. '
        'skip = true to not dub. '
        'Only list cards that change. Output JSON only: '
        '{"fixes":[{"index":0,"merge":[1],"text":"…","speaker":"","skip":false}]}'
    )
    fixes = []
    starts = list(range(0, len(transcript), _BATCH - _OVERLAP)) or [0]
    batches = len(starts)
    for batch_i, start in enumerate(starts, 1):
        end = min(len(transcript), start + _BATCH)
        if start >= end:
            continue
        allowed = set(range(start, end))
        _progress(progress_callback, f'辨識後修稿 {batch_i}/{batches}（卡片 {start}–{end - 1}）…')
        payload = [
            'Chinese ASR cards. Repair against the outline. Do not invent plot.',
            outline,
            '',
        ]
        for index in range(start, end):
            payload.append(_format_card(index, transcript[index]))
        check_stop()
        raw = llm_translate(
            method,
            [
                {'role': 'system', 'content': system},
                {'role': 'user', 'content': '\n'.join(payload)},
            ],
        )
        fixes.extend(_parse_fixes(raw, allowed))

    from tools.line_roles import SPEAKER_NARR, SPEAKER_SYS
    allowed_speakers = set(_known_speakers(transcript))
    allowed_speakers.update({SPEAKER_SYS, SPEAKER_NARR})
    created = []
    changed = skip_n = merge_n = moved_n = 0
    merges = []
    for item in fixes:
        try:
            index = int(item.get('index'))
        except (TypeError, ValueError):
            continue
        if index < 0 or index >= len(transcript):
            continue
        line = transcript[index]
        src = line.get('text') or ''
        new_spk = str(item.get('speaker') or '').strip()
        if new_spk == 'NARRATOR':
            new_spk = SPEAKER_NARR
        want_skip = item.get('skip') is True or str(item.get('skip')).lower() == 'true'
        if new_spk in {SPEAKER_NARR, SPEAKER_SYS} or item.get('skip') is False:
            line.pop('skip_tts', None)
        if want_skip and new_spk not in {SPEAKER_NARR, SPEAKER_SYS}:
            line['skip_tts'] = True
            skip_n += 1
            continue
        if new_spk:
            moved_n += _apply_speaker_item(transcript, item, allowed_speakers, created)
        new_text = str(item.get('text') or item.get('suggest_source') or '').strip()
        if new_text and _rewrite_ok(src, new_text, context, _duration(line)):
            line['text'] = new_text
            changed += 1
        span = _merge_span(item)
        if span:
            merges.append((span[0], span[1], new_text))

    blocked = set()
    deleted = []
    for host, absorbed, suggest_source in sorted(merges, key=lambda item: -item[0]):
        if host in blocked or any(index in blocked for index in absorbed):
            continue
        if not _join_legal(transcript, host, absorbed):
            continue
        glued = transcript[host].get('text') or ''
        for index in absorbed:
            glued = _glue_source(glued, transcript[index].get('text') or '')
        source = suggest_source if suggest_source and _rewrite_ok(glued, suggest_source, context, 99) else ''
        _join_lines(transcript, host, absorbed, '', source or None)
        if not source:
            transcript[host]['text'] = glued
        transcript[host].pop('translation', None)
        blocked.add(host)
        blocked.update(absorbed)
        deleted.extend(absorbed)
        merge_n += len(absorbed)
    for index in sorted(set(deleted), reverse=True):
        if 0 <= index < len(transcript):
            del transcript[index]
    try:
        audit_n, created, transcript = _speaker_audit(
            transcript, bible, method, allowed_speakers, created, progress_callback,
        )
        moved_n += audit_n
    except Exception as exc:
        if isinstance(exc, JobStopped):
            raise
        logger.warning(f'講者語意覆核失敗，沿用分批修稿：{exc}')
    from tools.line_roles import reassign_addressed_you_lines
    moved_n += reassign_addressed_you_lines(transcript)
    return changed, skip_n, merge_n, moved_n, created, transcript
