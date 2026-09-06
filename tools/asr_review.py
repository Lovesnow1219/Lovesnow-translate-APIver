# -*- coding: utf-8 -*-
"""Chinese-only sentence-cut polish after ASR, before translation.

Heuristic merge is cheap but blind. A short AI pass over the whole Chinese
script joins same-speaker cards that were one spoken line, so translation
sees complete sentences instead of fragments.
"""
import json
import os
import re
from datetime import datetime, timezone
from difflib import SequenceMatcher

from loguru import logger

from tools.asr_source import _glue_source, _is_particle_only, cleanup_asr_source, merge_utterance_cards
from tools.target_language import is_asr_junk

REVIEW_NAME = 'asr_review.json'
_BATCH = 40
_OVERLAP = 4
_MAX_ABSORB = 6
_MAX_GAP = 1.15
_FINISHED_GAP = 0.35
_SENTENCE_END = ('。', '！', '？', '.', '!', '?')
_HAN_KEEP = re.compile(r'[^\w\u4e00-\u9fff]+', flags=re.UNICODE)


def review_path(folder):
    return os.path.join(folder, REVIEW_NAME)


def polish_asr_segmentation(folder, transcript=None, method='OpenAI', progress_callback=None, force=False):
    """Compatibility wrapper. Outline + repair lives in asr_repair."""
    from tools.asr_repair import ensure_repaired_transcript

    return ensure_repaired_transcript(
        folder, transcript, method=method, progress_callback=progress_callback, force=force,
    )


def _already_polished(folder):
    path = review_path(folder)
    if not os.path.isfile(path):
        return False
    try:
        with open(path, 'r', encoding='utf-8') as handle:
            data = json.load(handle)
        return bool(isinstance(data, dict) and data.get('applied'))
    except Exception:
        return False


def _has_translation(folder):
    if os.path.isfile(os.path.join(folder, 'translation.json')):
        return True
    trans_dir = os.path.join(folder, 'translations')
    if not os.path.isdir(trans_dir):
        return False
    try:
        names = os.listdir(trans_dir)
    except OSError:
        return False
    return any(
        name.endswith('.json') and 'review' not in name.lower()
        for name in names
    )


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


def _open_cuts(transcript):
    from tools.translation_review import _card_t0, _card_t1, _join_candidates

    marked = set(_join_candidates(transcript))
    for i in range(len(transcript) - 1):
        cur = transcript[i]
        nxt = transcript[i + 1]
        if str(cur.get('speaker') or '') != str(nxt.get('speaker') or ''):
            continue
        left = (cur.get('text') or '').strip()
        right = (nxt.get('text') or '').strip()
        if not left or not right:
            continue
        if _is_particle_only(left) or _is_particle_only(right):
            continue
        if is_asr_junk(left) or is_asr_junk(right):
            continue
        gap = _card_t0(nxt) - _card_t1(cur)
        if gap > 0.45:
            continue
        if left.rstrip().endswith(_SENTENCE_END):
            continue
        marked.add(i)
    return sorted(marked)


def _han_compact(text):
    return _HAN_KEEP.sub('', text or '')


def _source_ok(glued, suggested):
    a = _han_compact(glued)
    b = _han_compact(suggested)
    if not b:
        return False
    if a == b:
        return True
    if b in a and len(b) >= max(2, int(len(a) * 0.85)):
        return True
    extra = sum(1 for ch in b if '\u4e00' <= ch <= '\u9fff' and ch not in a)
    if extra > 2:
        return False
    return SequenceMatcher(None, a, b).ratio() >= 0.9


def _glue_span(transcript, host, absorbed):
    glued = transcript[host].get('text') or ''
    for index in absorbed:
        glued = _glue_source(glued, transcript[index].get('text') or '')
    return glued


def _join_legal(transcript, host, absorbed):
    from tools.translation_review import _card_t0, _card_t1

    if host < 0 or host >= len(transcript):
        return False
    if not absorbed or len(absorbed) > _MAX_ABSORB:
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
        from tools.line_roles import line_role
        if line_role(left) != line_role(right):
            return False
        gap = _card_t0(transcript[index]) - _card_t1(transcript[prev])
        if gap > _MAX_GAP:
            return False
        if left.rstrip().endswith(_SENTENCE_END) and gap > _FINISHED_GAP:
            return False
        prev = index
    return True


def _ai_join_chinese(folder, transcript, method, progress_callback=None):
    from tools.job_control import check_stop
    from tools.translation_backends import llm_translate
    from tools.translation_bible import bible_context
    from tools.translation_review import (
        _card_t0,
        _card_t1,
        _load_summary,
        _merge_span,
        _parse_ai_findings,
        _review_model_kwargs,
    )

    if len(transcript) < 2:
        return []
    bible = bible_context(_load_summary(folder)) or ''
    cuts = set(_open_cuts(transcript))
    system = (
        'You are checking ASR subtitle cards in Chinese before they are translated for dubbing. '
        'Your only job is sentence segmentation: merge cards that are one spoken sentence wrongly cut by ASR. '
        'Join only consecutive cards with the SAME speaker. '
        'Do not join different speakers. '
        'Do not join standalone particles (嗯呵哼啊欸哦呃唉). '
        'Do not join two finished sentences that already end with 。！？ and have a real pause. '
        'Do not invent, drop, or rewrite plot words. You may only glue text and fix punctuation. '
        'index = first card to keep. merge = the next card index or indices to absorb (must be consecutive). '
        'suggest_source = the merged Chinese line (glue + punctuation only). '
        'issue = Traditional Chinese starting with 斷句： '
        'Skip cards that are already complete sentences. '
        'Cards marked <<CUT?>> are likely oversplits, but you may merge any legal consecutive pair. '
        'Output JSON only: '
        '{"findings":[{"index":0,"merge":[1],"severity":"mid","issue":"斷句：…","suggest_source":"…"}]}'
    )
    findings = []
    starts = list(range(0, len(transcript), _BATCH - _OVERLAP)) or [0]
    batches = len(starts)
    for batch_i, start in enumerate(starts, 1):
        end = min(len(transcript), start + _BATCH)
        if start >= end:
            continue
        allowed = set(range(start, end))
        _progress(
            progress_callback,
            f'辨識後斷句 {batch_i}/{batches}（卡片 {start}–{end - 1}）…',
        )
        payload = ['Chinese ASR cards. Merge only wrongly cut spoken sentences.', bible or '(no outline)', '']
        for index in range(start, end):
            line = transcript[index]
            gap = None
            if index + 1 < len(transcript):
                gap = _card_t0(transcript[index + 1]) - _card_t1(line)
            mark = ' <<CUT?>>' if index in cuts else ''
            gap_s = f' gap={gap:.2f}s' if gap is not None else ''
            payload.append(
                f'#{index} {line.get("speaker") or ""} '
                f'{_card_t0(line):.2f}-{_card_t1(line):.2f}{gap_s}{mark} | '
                f'{(line.get("text") or "").strip()}'
            )
        try:
            check_stop()
            raw = llm_translate(
                method,
                [
                    {'role': 'system', 'content': system},
                    {'role': 'user', 'content': '\n'.join(payload)},
                ],
                **_review_model_kwargs(),
            )
        except Exception as exc:
            from tools.job_control import JobStopped
            if isinstance(exc, JobStopped):
                raise
            logger.warning(f'辨識後斷句第 {batch_i} 批失敗：{exc}')
            continue
        for item in _parse_ai_findings(raw, allowed):
            span = _merge_span(item)
            if span is None and item.get('index') in allowed:
                nxt = int(item['index']) + 1
                if nxt in allowed:
                    item['merge'] = [nxt]
                    span = _merge_span(item)
            if span is None:
                continue
            host, absorbed = span
            if not _join_legal(transcript, host, absorbed):
                continue
            issue = (item.get('issue') or '').strip()
            if issue and not issue.startswith('斷句'):
                item['issue'] = f'斷句：{issue}'
            item['via'] = 'asr-join'
            item['suggest'] = ''
            findings.append(item)
    _progress(progress_callback, f'辨識後斷句 AI 標出 {len(findings)} 處')
    return findings


def _apply_joins(transcript, findings):
    from tools.translation_review import _coalesce_merge_groups, _join_lines, _merge_span

    groups = []
    for item in findings or []:
        span = _merge_span(item)
        if span is None:
            continue
        host, absorbed = span
        if not _join_legal(transcript, host, absorbed):
            continue
        groups.append((host, absorbed, '', item.get('suggest_source')))
    coalesced = _coalesce_merge_groups(groups)
    ready = []
    for host, absorbed, suggest, suggest_source in coalesced:
        if _join_legal(transcript, host, absorbed):
            ready.append((host, absorbed, suggest, suggest_source))
            continue
        for orig in groups:
            if orig[0] < host or orig[1][-1] > absorbed[-1]:
                continue
            if _join_legal(transcript, orig[0], orig[1]):
                ready.append(orig)
    deleted = []
    merged_n = 0
    used = set()
    for host, absorbed, suggest, suggest_source in sorted(ready, key=lambda item: -item[0]):
        if host in used or any(index in used for index in absorbed):
            continue
        if not _join_legal(transcript, host, absorbed):
            continue
        glued = _glue_span(transcript, host, absorbed)
        source = str(suggest_source or '').strip()
        if source and not _source_ok(glued, source):
            logger.info(f'斷句建議改寫太多，改用原文黏合：#{host} {source[:40]}')
            source = None
        _join_lines(transcript, host, absorbed, suggest or '', source)
        deleted.extend(absorbed)
        used.add(host)
        used.update(absorbed)
        merged_n += len(absorbed)
    for index in sorted(set(deleted), reverse=True):
        del transcript[index]
    return merged_n, transcript
