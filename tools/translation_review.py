# -*- coding: utf-8 -*-
"""Flag dubbed-line problems after translation. Apply is a separate, explicit step."""
import json
import os
import re
from datetime import datetime, timezone

from loguru import logger

from tools.target_language import is_asr_junk, is_chinese_target, load_dub_meta, translation_language
from tools.translation_prompts import review_system_preamble
from tools.translation_quality import (
    HAN_RE, JA_CN_LEFT_RE, KANA_RE, leftover_chinese_reason, restores_voice,
    sense_issue, suggest_wrecks_voice, glossary_forced_on_junk, _foreign_asr_source,
    _glued_two_mouths, _guesses_unfinished, _mostly_cjk_episode, _needs_empty_vocal,
    _rewrites_foreign_asr, _sense_broken, _speaks_both_mouths, _technique_crushed,
    _two_mouth_source,
)
from tools.vocal_particles import is_particle_card, particle_translation

REVIEW_NAME = 'translation_review.json'
_BATCH = 10
_CONT_BATCH = 90
_GLOSS_SKIP_RE = re.compile(
    r'世界|现代|現代|战争|戰爭|游戏|遊戲|银行|銀行|货币|貨幣|金币|金幣'
)
_SEVERITY_RANK = {'high': 0, 'mid': 1, 'low': 2}
_PRONOUN_ISSUE = re.compile(r'複數|自言自語|第一人稱|不應使用複數|不该使用复数')


def review_path(folder):
    return os.path.join(folder, REVIEW_NAME)


def load_review(folder, language=None):
    from tools.translation_versions import review_version_path

    paths = []
    if language:
        paths.append(review_version_path(folder, language))
    paths.append(review_path(folder))
    for path in paths:
        if not os.path.isfile(path):
            continue
        try:
            with open(path, 'r', encoding='utf-8') as handle:
                data = json.load(handle)
            if isinstance(data, dict):
                if language and data.get('language') and translation_language(data.get('language')) != translation_language(language):
                    continue
                return data
        except Exception:
            continue
    return {}


def _findings(folder, language=None):
    report = load_review(folder, language)
    lang = translation_language(language or report.get('language') or 'English')
    return [_enrich_finding(item, lang) for item in (report.get('findings') or [])]


def review_table_rows(folder, language=None):
    rows = []
    for item in _findings(folder, language):
        rows.append([
            item.get('index'),
            item.get('start'),
            item.get('suggest') or '',
            item.get('severity') or '',
            item.get('issue') or '',
        ])
    return rows


def review_pick_choices(folder, language=None):
    choices = []
    for item in _findings(folder, language):
        suggest = re.sub(r'\s+', ' ', str(item.get('suggest') or '')).strip()
        dest = str(item.get('suggest_speaker') or '').strip()
        if not suggest and dest:
            suggest = f'講者 → {dest}'
        if not suggest:
            continue
        try:
            index = int(item.get('index'))
        except (TypeError, ValueError):
            continue
        start = item.get('start')
        label = f'{index}｜{start}s｜{suggest}'
        choices.append((label, str(index)))
    return choices


def review_suggested_values(folder, language=None):
    return [value for _label, value in review_pick_choices(folder, language)]


_AUTO_SKIP_ISSUE = re.compile(r'推銷|推销|突然變成|突然变成|講者設定|讲者设定|變成推銷')
_SPEAKER_LABEL_ISSUE = re.compile(
    r'說話者|说话者|講者|讲者|誤由|误由|身分|誤置|误置|應由|应由|改回兩個|改回两个'
)
_SPEAKER_ID_RE = re.compile(r'^SPEAKER_[A-Z][A-Z0-9]{0,3}$')


def _suggest_drops_glued_dub(suggest, glued_dub):
    """True when a join rewrite dropped most of the already-dubbed clauses."""
    sug = (suggest or '').strip()
    old = (glued_dub or '').strip()
    if not sug or not old:
        return False

    def ntok(text):
        return len(re.findall(r"[A-Za-z0-9']+|[\u4e00-\u9fff]", text))

    old_n, new_n = ntok(old), ntok(sug)
    return old_n >= 12 and new_n < max(8, int(0.55 * old_n))


def _valid_suggest_speaker(name, current, known):
    dest = str(name or '').strip()
    if dest == 'NARRATOR':
        dest = 'SPEAKER_NARR'
    if dest not in {'SPEAKER_SYS', 'SPEAKER_NARR'} and not _SPEAKER_ID_RE.match(dest):
        return ''
    if dest == str(current or '').strip():
        return ''
    if known and dest not in known:
        return ''
    return dest


def _sync_source_speaker(folder, line, new_spk):
    path = os.path.join(folder, 'transcript.json')
    if not os.path.isfile(path):
        return
    try:
        with open(path, 'r', encoding='utf-8') as handle:
            rows = json.load(handle)
    except Exception:
        return
    if not isinstance(rows, list):
        return
    target_text = (line.get('text') or '').strip()
    try:
        target_start = float(line.get('orig_start') if line.get('orig_start') is not None else line.get('start') or -1)
    except (TypeError, ValueError):
        target_start = -1
    changed = False
    for src in rows:
        if (src.get('text') or '').strip() != target_text:
            continue
        try:
            src_start = float(src.get('orig_start') if src.get('orig_start') is not None else src.get('start') or -2)
        except (TypeError, ValueError):
            continue
        if abs(src_start - target_start) > 0.08:
            continue
        src['speaker'] = new_spk
        changed = True
        break
    if changed:
        with open(path, 'w', encoding='utf-8') as handle:
            json.dump(rows, handle, indent=4, ensure_ascii=False)


def auto_apply_review_picks(folder, language=None):
    """Suggestions safe to write before TTS. Skip style nits; keep speaker-id moves."""
    from tools.line_roles import is_asr_stump

    picks = []
    for item in _findings(folder, language):
        suggest = str(item.get('suggest') or '').strip()
        dest = str(item.get('suggest_speaker') or '').strip()
        if not suggest and not dest:
            continue
        if suggest and not dest and suggest == str(item.get('translation') or '').strip():
            continue
        issue = str(item.get('issue') or '')
        severity = str(item.get('severity') or 'mid').lower()
        if severity == 'low' and not dest and not _PRONOUN_ISSUE.search(issue):
            continue
        via = str(item.get('via') or '')
        if 'continuity' in via and _AUTO_SKIP_ISSUE.search(issue):
            continue
        if _SPEAKER_LABEL_ISSUE.search(issue) and not dest:
            continue
        if via == 'join' or str(issue).startswith('斷句'):
            continue
        if any(key in issue for key in ('辨識殘句', '原文被切斷', '辨識幻聽')):
            continue
        src = str(item.get('source') or '')
        from tools.line_roles import line_role
        if 'continuity' in via and line_role(src):
            continue
        empty_now = not str(item.get('translation') or '').strip()
        after_empty = via == 'after_dub' and empty_now and suggest
        if ('殘句' in issue or '残句' in issue) and not after_empty:
            continue
        if is_asr_stump(src) and not after_empty:
            continue
        if _glued_two_mouths(suggest) or _speaks_both_mouths(src, suggest):
            continue
        if _suggest_drops_glued_dub(suggest, item.get('translation')):
            continue
        if _guesses_unfinished(src, suggest) or _rewrites_foreign_asr(src, suggest):
            continue
        if '不要硬套' in issue:
            continue
        if suggest and suggest_wrecks_voice(
            src, item.get('translation'), suggest, language or 'English',
        ) and not restores_voice(
            src, item.get('translation'), suggest, language or 'English',
        ):
            continue
        try:
            picks.append(str(int(item.get('index'))))
        except (TypeError, ValueError):
            continue
    return picks


def applied_review_indices(folder, language=None):
    out = []
    for item in _findings(folder, language):
        if not item.get('applied'):
            continue
        try:
            out.append(int(item.get('index')))
        except (TypeError, ValueError):
            continue
    return out


def _remap_indices(indices, deleted):
    removed = set(deleted or [])
    out = []
    seen = set()
    for raw in indices or []:
        try:
            index = int(float(raw))
        except (TypeError, ValueError):
            continue
        if index in removed:
            continue
        new_index = index - sum(1 for gone in removed if gone < index)
        if new_index in seen:
            continue
        seen.add(new_index)
        out.append(new_index)
    return out


def _slot_budget_note(line, language):
    from tools.translation_quality import _slot_seconds, _spoken_char_budget, _spoken_word_budget
    from tools.target_language import uses_char_budget, uses_word_budget

    duration = _slot_seconds(line)
    if uses_word_budget(language):
        budget = _spoken_word_budget(duration, language)
        unit = 'syllables' if translation_language(language) == 'Vietnamese' else 'words'
        return f'{duration:.1f}s, max {budget} {unit}'
    if uses_char_budget(language):
        budget = _spoken_char_budget(duration, language)
        return f'{duration:.1f}s, max {budget} characters'
    return f'{duration:.1f}s'


def _line_qc_flags(line, language, transcript=None, glossary=None):
    from tools.line_roles import is_asr_stump, should_skip_dub

    src = (line.get('text') or '').strip()
    dub = (line.get('translation') or '').strip()
    flags = []
    if should_skip_dub(line):
        flags.append('SKIP')
    if is_asr_stump(line):
        flags.append('STUMP')
    if is_particle_card(src):
        flags.append('PARTICLE')
    elif not dub and (not should_skip_dub(line) or _needs_empty_vocal(line)):
        flags.append('EMPTY')
    if leftover_chinese_reason(dub, language):
        flags.append('HAN')
    if _short_slot_overflow(line, language):
        flags.append('RUSH')
    if _glued_two_mouths(dub) or _speaks_both_mouths(src, dub) or _two_mouth_source(src):
        flags.append('GLUE')
    if _technique_crushed(src, dub):
        flags.append('TECHNIQUE')
    if _guesses_unfinished(src, dub):
        flags.append('GUESS')
    if _sense_broken(src, dub):
        flags.append('SENSE')
    if glossary_forced_on_junk(src, dub, glossary):
        flags.append('JUNK')
    if transcript is not None and _mostly_cjk_episode(transcript) and _foreign_asr_source(src):
        flags.append('FOREIGN')
    return flags


def _ship_line_row(index, line, language, transcript=None, folder=None):
    from tools.translation_quality import _slot_seconds, _spoken_word_budget
    from tools.target_language import uses_word_budget

    duration = _slot_seconds(line)
    dub = (line.get('translation') or '').strip()
    words = len(dub.split()) if dub else 0
    budget = _spoken_word_budget(duration, language) if uses_word_budget(language) else 0
    flags = ','.join(_line_qc_flags(
        line, language, transcript=transcript, glossary=_folder_glossary_pairs(folder),
    )) or '-'
    wav_bit = ''
    if folder:
        _slot, wav, late = _wav_line_stats(folder, index, line)
        if wav is not None:
            wav_bit = f' wav={wav:.2f}s late={late:.2f}s'
    return (
        f'#{index} {line.get("speaker") or ""} {duration:.2f}s{wav_bit} '
        f'words={words} max={budget} [{flags}] | '
        f'SRC: {line.get("text") or ""} | DUB: {dub}'
    )


def _voice_table(folder):
    path = os.path.join(folder or '', 'speaker_voices.json')
    if not os.path.isfile(path):
        return ''
    try:
        data = json.load(open(path, encoding='utf-8'))
    except Exception:
        return ''
    if not isinstance(data, dict):
        return ''
    rows = ['Voice bank (stock match, not clone):']
    for speaker, info in data.items():
        if not isinstance(info, dict):
            continue
        pitch = info.get('pitch_hz')
        pitch_s = f'{float(pitch):.0f}Hz' if isinstance(pitch, (int, float)) else '?'
        rows.append(
            f'{speaker}: role={info.get("role") or "?"} pitch={pitch_s} '
            f'voice={info.get("voice_name") or "?"}'
        )
    return '\n'.join(rows) if len(rows) > 1 else ''


def review_status_line(folder, language=None):
    findings = _findings(folder, language)
    if not findings:
        report = load_review(folder, language)
        if report:
            return '審稿：沒有標出異常（譯文未改）'
        return '審稿：尚未跑過'
    counts = {'high': 0, 'mid': 0, 'low': 0}
    applied_n = 0
    suggest_n = 0
    for item in findings:
        key = item.get('severity') if item.get('severity') in counts else 'mid'
        counts[key] += 1
        if item.get('applied'):
            applied_n += 1
        if str(item.get('suggest') or '').strip():
            suggest_n += 1
    extra = f'已套用 {applied_n} 條。' if applied_n else '尚未套用。'
    cont_n = sum(
        1 for item in findings
        if str(item.get('via') or '') == 'continuity' or str(item.get('issue') or '').startswith('連貫')
    )
    join_n = sum(1 for item in findings if _merge_span(item))
    bits = []
    if cont_n:
        bits.append(f'連貫 {cont_n}')
    if join_n:
        bits.append(f'斷句 {join_n}')
    ship_n = sum(
        1 for item in findings
        if str(item.get('via') or '') == 'ship' or str(item.get('issue') or '').startswith('品管')
    )
    if ship_n:
        bits.append(f'品管 {ship_n}')
    extra_kinds = f'其中{"／".join(bits)} 條。' if bits else ''
    return (
        f'審稿標出 {len(findings)} 句（高 {counts["high"]}／中 {counts["mid"]}／低 {counts["low"]}），'
        f'其中 {suggest_n} 條有建議譯文可勾選套用。{extra_kinds}{extra}'
    )


def _suggests_from_table(table):
    if table is None:
        return {}
    if hasattr(table, 'values'):
        rows = [list(row) for row in table.values.tolist()]
    else:
        rows = [list(row) for row in (table or [])]
    out = {}
    for row in rows:
        if not row:
            continue
        try:
            index = int(float(row[0]))
        except (TypeError, ValueError, IndexError):
            continue
        suggest = ''
        if len(row) > 2 and row[2] is not None:
            suggest = str(row[2]).strip()
        if suggest:
            out[index] = suggest
    return out


def apply_selected_review(folder, language, selected, review_table=None):
    """Write checked suggestions into this language version. Does not run TTS."""
    from tools.speaker_edit import episode_folder
    from tools.translation import _clear_line_wavs, _clear_mix_outputs
    from tools.translation_versions import (
        load_lines_for_language,
        snapshot_active,
        write_review_report,
        write_translation,
    )

    folder = episode_folder(folder)
    language = translation_language(language or load_dub_meta(folder).get('translation') or 'English')
    snapshot_active(folder)
    transcript = load_lines_for_language(folder, language)
    if not transcript:
        raise FileNotFoundError(f'找不到 {language} 譯文：{folder}')
    selected_idx = set()
    for item in selected or []:
        try:
            selected_idx.add(int(float(item)))
        except (TypeError, ValueError):
            continue
    report = load_review(folder, language)
    lang = translation_language(language or report.get('language') or 'English')
    by_index = {}
    for finding in report.get('findings') or []:
        try:
            by_index[int(finding.get('index'))] = _enrich_finding(finding, lang)
        except (TypeError, ValueError):
            continue
    table_suggests = _suggests_from_table(review_table)
    groups = []
    merge_ids = set()
    for index in selected_idx:
        span = _merge_span(by_index.get(index) or {})
        if not span:
            continue
        host, absorbed = span
        suggest = str(
            table_suggests.get(host)
            or table_suggests.get(index)
            or (by_index.get(host) or {}).get('suggest')
            or (by_index.get(index) or {}).get('suggest')
            or ''
        ).strip()
        if not suggest:
            continue
        groups.append((host, absorbed, suggest, (by_index.get(host) or {}).get('suggest_source')))
        merge_ids.add(host)
        merge_ids.update(absorbed)
    groups = _coalesce_merge_groups(groups)
    applied = 0
    skipped = 0
    merged_n = 0
    applied_old = []
    deleted = []
    for index in sorted(selected_idx):
        if index in merge_ids:
            continue
        if index < 0 or index >= len(transcript):
            skipped += 1
            continue
        finding = by_index.get(index) or {}
        suggest = str(
            table_suggests.get(index)
            or finding.get('suggest')
            or ''
        ).strip()
        if suggest.startswith('講者 →'):
            suggest = ''
        new_spk = _valid_suggest_speaker(
            finding.get('suggest_speaker'),
            transcript[index].get('speaker'),
            {str(line.get('speaker') or '') for line in transcript},
        )
        if not suggest and not new_spk:
            skipped += 1
            continue
        if suggest and _glued_two_mouths(suggest):
            skipped += 1
            continue
        if suggest and suggest_wrecks_voice(
            transcript[index].get('text'),
            transcript[index].get('translation'),
            suggest,
            language,
        ) and not restores_voice(
            transcript[index].get('text'),
            transcript[index].get('translation'),
            suggest,
            language,
        ):
            skipped += 1
            continue
        if suggest:
            transcript[index]['translation'] = suggest
        if not suggest and not new_spk:
            skipped += 1
            continue
        if new_spk:
            transcript[index]['speaker'] = new_spk
            _sync_source_speaker(folder, transcript[index], new_spk)
        if index in by_index:
            by_index[index]['applied'] = True
            if suggest:
                by_index[index]['suggest'] = suggest
            if new_spk:
                by_index[index]['suggest_speaker'] = new_spk
        _clear_line_wavs(folder, index)
        applied_old.append(index)
        applied += 1
    if groups:
        from tools.target_language import clear_tts_cache
        for host, absorbed, suggest, suggest_source in sorted(groups, key=lambda item: -item[0]):
            if host < 0 or host >= len(transcript):
                skipped += 1
                continue
            if any(j < 0 or j >= len(transcript) for j in absorbed):
                skipped += 1
                continue
            host_spk = str(transcript[host].get('speaker') or '')
            if any(str(transcript[j].get('speaker') or '') != host_spk for j in absorbed):
                skipped += 1
                continue
            from tools.line_roles import line_role
            host_role = line_role(transcript[host].get('text'))
            if any(line_role(transcript[j].get('text')) != host_role for j in absorbed):
                skipped += 1
                continue
            from tools.asr_source import _glue_source
            glued_src = transcript[host].get('text') or ''
            glued_dub = transcript[host].get('translation') or ''
            for j in absorbed:
                glued_src = _glue_source(glued_src, transcript[j].get('text') or '')
                glued_dub = _glue_dub(glued_dub, transcript[j].get('translation'), language)
            if _suggest_drops_glued_dub(suggest, glued_dub):
                suggest = (glued_dub or suggest).strip()
            _join_lines(transcript, host, absorbed, suggest, suggest_source)
            if host in by_index:
                by_index[host]['applied'] = True
                by_index[host]['suggest'] = suggest
            deleted.extend(absorbed)
            applied_old.append(host)
            applied += 1
            merged_n += len(absorbed)
        for index in sorted(set(deleted), reverse=True):
            del transcript[index]
        by_index = _reindex_findings(by_index, deleted)
        clear_tts_cache(folder, keep_video=True)
    applied_new = _remap_indices(applied_old, deleted)
    kept = [str(index) for index in _remap_indices(selected_idx, deleted)]
    if applied:
        write_translation(folder, transcript, language)
        _clear_mix_outputs(folder, keep_video=True)
        if applied_new:
            from tools.translation import tighten_overlong_lines
            _summary, transcript = tighten_overlong_lines(
                folder, language, indices=applied_new, slack=2,
            )
            by_lookup = {int(item.get('index')): item for item in by_index.values() if item.get('index') is not None}
            for idx in applied_new:
                text = ''
                if 0 <= idx < len(transcript):
                    text = str(transcript[idx].get('translation') or '').strip()
                row = by_lookup.get(idx) or by_index.get(idx)
                if row and text:
                    row['suggest'] = text
        report['findings'] = sorted(
            by_index.values(),
            key=lambda row: (_SEVERITY_RANK.get(row.get('severity'), 9), row.get('index', 0)),
        )
        report['applied'] = True
        report['applied_count'] = applied
        write_review_report(folder, language, report)
    parts = [f'已套用 {applied} 條建議到 {language}']
    if merged_n:
        parts.append(f'併回 {merged_n} 張被切錯的卡片')
    if skipped:
        parts.append(f'略過 {skipped} 條（沒有建議譯文或句號不對）')
    if applied_new:
        parts.append('過長建議已收緊到原句秒數')
    parts.append('譯文已寫入，尚未重配。成片仍留在資料夾；要聽新聲音請按「只重配音」。')
    message = '。'.join(parts)
    logger.info(message)
    return folder, applied, skipped, message, kept


def _progress(progress_callback, message):
    logger.info(message)
    if progress_callback:
        progress_callback(message)


def review_folder(folder, target_language=None, method='OpenAI', progress_callback=None):
    """Write a review report. Does not change translation.json; caller may apply."""
    from tools.translation_versions import load_lines_for_language, snapshot_active, write_review_report

    language = translation_language(
        target_language or load_dub_meta(folder).get('translation') or 'English'
    )
    _progress(progress_callback, f'載入 {language} 譯文…')
    snapshot_active(folder)
    transcript = load_lines_for_language(folder, language)
    if not transcript:
        raise FileNotFoundError(f'找不到 {language} 譯文可審稿：{folder}')
    if not any((line or {}).get('translation') for line in transcript):
        raise FileNotFoundError(f'{language} 尚未翻譯，沒有譯文可審稿：{folder}')
    _progress(progress_callback, f'已載入 {len(transcript)} 句，先跑規則掃過…')
    summary = _load_summary(folder)
    rules = _rule_findings(transcript, language, summary)
    suspects = _suspect_indices(transcript, language, rules)
    _progress(progress_callback, f'規則標出 {len(rules)} 條，可疑句 {len(suspects)} 句')
    ai_findings = []
    if not is_chinese_target(language) and method in {'OpenAI', 'LLM', '阿里云-通义千问', 'Ernie', 'Ollama'}:
        ai_findings = _ai_review(
            transcript, language, summary, suspects, method, folder=folder,
            progress_callback=progress_callback,
        )
        ai_findings.extend(_continuity_review(
            transcript, language, summary, method, folder=folder,
            progress_callback=progress_callback,
        ))
        ai_findings.extend(_ship_qc_review(
            transcript, language, summary, method, folder=folder,
            progress_callback=progress_callback,
        ))
    else:
        _progress(progress_callback, '這個語言略過 AI，只保留規則結果')
    _progress(progress_callback, '彙整並寫入審稿報告…')
    findings = [_enrich_finding(item, language) for item in _merge_findings(transcript, rules, ai_findings)]
    report = {
        'language': language,
        'reviewed_at': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'line_count': len(transcript),
        'suspect_count': len(suspects),
        'applied': False,
        'findings': findings,
    }
    write_review_report(folder, language, report)
    _progress(progress_callback, f'審稿寫入 {len(findings)} 條、未改譯文')
    return report


def _load_summary(folder):
    path = os.path.join(folder, 'summary.json')
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, 'r', encoding='utf-8') as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _glossary_pairs(summary):
    pairs = []
    raw = (summary or {}).get('glossary') or ''
    for chunk in re.split(r'[;\n]+', raw):
        if '=' not in chunk:
            continue
        left, right = chunk.split('=', 1)
        src = left.strip()
        dst = right.strip()
        if src and dst and _is_name_gloss(src, dst):
            pairs.append((src, dst))
    return pairs


def _folder_glossary_pairs(folder):
    if not folder:
        return []
    path = os.path.join(folder, 'summary.json')
    if not os.path.isfile(path):
        return []
    try:
        with open(path, 'r', encoding='utf-8') as handle:
            return _glossary_pairs(json.load(handle))
    except Exception:
        return []


def _is_name_gloss(src, dst):
    if len(src) < 2 or _GLOSS_SKIP_RE.search(src):
        return False
    if re.fullmatch(r'[\u30a0-\u30ffー]+', dst or ''):
        return True
    return bool(re.fullmatch(
        r"[A-Z][A-Za-z0-9'’\-]+(?:\s+[A-Z0-9][A-Za-z0-9'’\-]+){0,5}"
        r"(?::\s+[A-Z][A-Za-z0-9'’\-]+(?:\s+[A-Z0-9][A-Za-z0-9'’\-]+){0,5})?",
        dst or '',
    ))


def _line_start(line):
    try:
        return round(float(line.get('orig_start', line.get('start') or 0)), 2)
    except (TypeError, ValueError):
        return 0.0


def _compact_src(text):
    return ''.join(ch for ch in (text or '') if ch.strip() and ch not in '，,。！？!?、…')


def _card_t0(line):
    if line.get('orig_start') is not None:
        return float(line.get('orig_start') or 0)
    return float(line.get('start') or 0)


def _card_t1(line):
    if line.get('orig_end') is not None:
        return float(line.get('orig_end') or 0)
    return float(line.get('end') or 0)


def _merge_ids(raw, host):
    ids = {int(host)}
    if raw in (None, '', [], ()):
        return []
    if not isinstance(raw, (list, tuple, set)):
        raw = re.split(r'[,，\s]+', str(raw))
    for item in raw:
        try:
            ids.add(int(float(item)))
        except (TypeError, ValueError):
            continue
    return sorted(ids)


def _merge_span(finding):
    if not finding:
        return None
    try:
        host = int(finding.get('index'))
    except (TypeError, ValueError):
        return None
    ordered = _merge_ids(finding.get('merge'), host)
    if len(ordered) < 2:
        return None
    if ordered != list(range(ordered[0], ordered[-1] + 1)):
        return None
    return ordered[0], ordered[1:]


def _coalesce_merge_groups(groups):
    spans = []
    for host, absorbed, suggest, source in groups:
        if not absorbed:
            continue
        spans.append([host, absorbed[-1], suggest, source])
    spans.sort()
    out = []
    for start, end, suggest, source in spans:
        if out and start <= out[-1][1] + 1:
            out[-1][1] = max(out[-1][1], end)
            if suggest:
                out[-1][2] = suggest
            if source:
                out[-1][3] = source
        else:
            out.append([start, end, suggest, source])
    return [(start, list(range(start + 1, end + 1)), suggest, source) for start, end, suggest, source in out]


def _glue_dub(left, right, language):
    left = (left or '').strip()
    right = (right or '').strip()
    if not left:
        return right
    if not right or right == left:
        return left
    lang = translation_language(language)
    if lang in {'Japanese', '简体中文', '繁体中文', '粤语'}:
        if left.endswith(('、', '，', ',', '…')):
            return left + right
        return left + right
    if left.endswith((',', ';', '…', '...')):
        return f'{left} {right}'.strip()
    return f'{left.rstrip(",")} {right}'.strip()


def _join_lines(transcript, host, absorbed, suggest, suggest_source=None):
    from tools.asr_source import _glue_source

    line = transcript[host]
    texts = [line.get('text') or '']
    for index in absorbed:
        texts.append(transcript[index].get('text') or '')
        line['end'] = transcript[index].get('end', line.get('end'))
        if transcript[index].get('orig_end') is not None or line.get('orig_start') is not None:
            line['orig_end'] = transcript[index].get('orig_end', transcript[index].get('end'))
    glued = texts[0]
    for piece in texts[1:]:
        glued = _glue_source(glued, piece)
    if str(suggest_source or '').strip():
        line['text'] = str(suggest_source).strip()
    else:
        line['text'] = glued
    line['translation'] = suggest
    line['start'] = line.get('start', _card_t0(line))
    if line.get('orig_start') is None:
        line['orig_start'] = _card_t0(line)
    if line.get('orig_end') is None:
        line['orig_end'] = _card_t1(line)


def _reindex_findings(by_index, deleted):
    removed = set(deleted or [])
    shifted = {}
    for index, item in by_index.items():
        if index in removed:
            continue
        new_index = index - sum(1 for gone in removed if gone < index)
        row = dict(item)
        row['index'] = new_index
        if row.get('merge'):
            row.pop('merge', None)
        shifted[new_index] = row
    return shifted


def _join_candidates(transcript):
    from tools.asr_source import _is_particle_only

    out = []
    for i in range(len(transcript or []) - 1):
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
        if gap > 0.85:
            continue
        left_cut = left.rstrip().endswith(('，', ',', '、', '…', '...'))
        left_open = not left.rstrip().endswith(('。', '！', '？', '.', '!', '?'))
        short_right = len(_compact_src(right)) <= 8
        short_left = len(_compact_src(left)) <= 6
        mid_cut = (
            gap <= 0.08
            and left_open
            and min(len(_compact_src(left)), len(_compact_src(right))) <= 10
        )
        if left_cut or short_right or short_left or mid_cut:
            out.append(i)
    return out


def _rule_join_findings(transcript, language):
    out = []
    seen = set()
    for i in _join_candidates(transcript):
        if i in seen:
            continue
        cur = transcript[i]
        nxt = transcript[i + 1]
        left = (cur.get('text') or '').strip()
        if not left.rstrip().endswith(('，', ',', '、')):
            continue
        seen.add(i)
        suggest = _glue_dub(cur.get('translation'), nxt.get('translation'), language)
        if not suggest:
            continue
        out.append({
            'index': i,
            'start': _line_start(cur),
            'speaker': cur.get('speaker') or '',
            'source': left,
            'translation': cur.get('translation') or '',
            'severity': 'mid',
            'issue': '斷句：原文被切斷，應與下一句併成一句',
            'suggest': suggest,
            'merge': [i + 1],
            'via': 'join',
        })
    return out


def _ai_join_review(transcript, language, summary, method, progress_callback=None, folder=None):
    hosts = _join_candidates(transcript)
    if not hosts:
        _progress(progress_callback, '斷句拼接：沒有切口，略過')
        return []
    from tools.translation_backends import llm_translate
    from tools.translation_bible import bible_context, review_bible_context

    bible = review_bible_context(folder, summary) or bible_context(summary) or ''
    system = (
        'You are a dubbing subtitle joiner, not a translator of the whole episode. '
        'Some consecutive cards are one spoken sentence cut wrongly by ASR or over-split. '
        'Join only those. '
        'Do not join different speakers, standalone particles (嗯呵哼啊欸), '
        'or two finished sentences with a real pause. '
        'index = first card to keep. merge = the next card index or two to absorb. '
        'suggest = ONE natural dubbed sentence in the target language. '
        'It must cover every clause in the merged Chinese. Do not drop a later clause. '
        'It must fit the combined cards\' spoken time. Do not add plot. '
        'Do not explain in suggest. issue = Traditional Chinese starting with 斷句： '
        'Output JSON only: {"findings":[{"index":0,"merge":[1],"severity":"mid","issue":"斷句：…","suggest":"…"}]}'
    )
    allowed = set()
    for host in hosts:
        allowed.add(host)
        allowed.add(host + 1)
        if host + 2 < len(transcript):
            allowed.add(host + 2)
    findings = []
    batch = 16
    batches = max(1, (len(hosts) + batch - 1) // batch)
    for batch_i, start in enumerate(range(0, len(hosts), batch), 1):
        chunk_hosts = hosts[start:start + batch]
        _progress(
            progress_callback,
            f'斷句拼接 {batch_i}/{batches}（{len(chunk_hosts)} 處切口）…',
        )
        payload = [f'Language: {language}', bible or '(no outline)', '', 'Candidate cuts:']
        shown = set()
        for host in chunk_hosts:
            for index in (host, host + 1):
                if index in shown or index >= len(transcript):
                    continue
                shown.add(index)
                line = transcript[index]
                payload.append(
                    f'#{index} {line.get("speaker") or ""} {_slot_budget_note(line, language)} | '
                    f'SRC: {line.get("text") or ""} | DUB: {line.get("translation") or ""}'
                )
        try:
            from tools.job_control import check_stop
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
            logger.warning(f'斷句拼接審核失敗：{exc}')
            continue
        for item in _parse_ai_findings(raw, allowed):
            span = _merge_span(item)
            if span is None and item.get('index') in hosts:
                item['merge'] = [int(item['index']) + 1]
                span = _merge_span(item)
            if span is None:
                continue
            host, absorbed = span
            if host >= len(transcript) or any(j >= len(transcript) for j in absorbed):
                continue
            if str(transcript[host].get('speaker') or '') != str(transcript[absorbed[0]].get('speaker') or ''):
                continue
            item['via'] = 'join'
            issue = (item.get('issue') or '').strip()
            if issue and not issue.startswith('斷句'):
                item['issue'] = f'斷句：{issue}'
            if not (item.get('suggest') or '').strip():
                item['suggest'] = _glue_dub(
                    transcript[host].get('translation'),
                    transcript[absorbed[0]].get('translation'),
                    language,
                )
            findings.append(item)
    _progress(progress_callback, f'斷句拼接完成，標出 {len(findings)} 條')
    return findings


def _rule_rewrite(src, dub, language):
    """Particles only. Wording stays with the translator."""
    from tools.vocal_particles import collapse_double_particle_lead

    src = src or ''
    text = dub or ''
    if not text:
        return particle_translation(src, language) or ''
    original = text
    expected = particle_translation(src, language)
    if expected and expected.rstrip('。. ') not in text:
        text = expected
    collapsed = collapse_double_particle_lead(text, language)
    if collapsed:
        text = collapsed
    return text if text != original else ''


def _enrich_finding(item, language):
    row = dict(item or {})
    if str(row.get('suggest') or '').strip():
        return row
    rewritten = _rule_rewrite(row.get('source') or '', row.get('translation') or '', language)
    if rewritten:
        row['suggest'] = rewritten
    return row


def _rule_findings(transcript, language, summary):
    from tools.line_roles import should_skip_dub
    from tools.vocal_particles import collapse_double_particle_lead

    glossary = _glossary_pairs(summary)
    out = []
    for i, line in enumerate(transcript or []):
        src = (line.get('text') or '').strip()
        dub = (line.get('translation') or '').strip()
        if should_skip_dub(line) and not _needs_empty_vocal(line):
            continue
        issues = []
        suggest = ''
        leftover = leftover_chinese_reason(dub, language)
        if leftover:
            issues.append('譯文殘留中文')
        if language == 'Japanese' and JA_CN_LEFT_RE.search(dub):
            issues.append('日文裡出現簡體語氣字')
        expected = particle_translation(src, language)
        if expected and dub and expected.rstrip('。. ') not in dub:
            issues.append(f'語氣詞應為 {expected.rstrip()}')
            suggest = expected
        if not suggest:
            suggest = _rule_rewrite(src, dub, language)
        if language == 'Japanese' and len(dub) >= 18 and KANA_RE.search(dub) and not HAN_RE.search(dub):
            issues.append('整句幾乎沒有漢字，可能是品質檢查逼成全假名')
        collapsed = collapse_double_particle_lead(dub, language)
        if collapsed and collapsed != dub:
            issues.append('語氣詞重複了')
            if not suggest:
                suggest = collapsed
        for cn, dst in glossary:
            if cn in src and dst and dst not in dub:
                if len(cn) >= 2:
                    issues.append(f'人名「{cn}」大綱是 {dst}，譯文沒對上')
                    break
        if not dub and src and not is_particle_card(src):
            issues.append('真實口白沒有配音')
        if _glued_two_mouths(dub) or _speaks_both_mouths(src, dub):
            issues.append('兩人口吻黏在同一句')
        elif _two_mouth_source(src):
            issues.append('兩人口吻黏在同一張卡，只譯講者那一句')
        if _technique_crushed(src, dub):
            issues.append('招式／口訣被收成電報')
        if _guesses_unfinished(src, dub):
            issues.append('未說完的句子被猜完')
        broken = sense_issue(src, dub)
        if broken:
            issues.append(broken)
        forced = glossary_forced_on_junk(src, dub, glossary)
        if forced:
            cn, dst = forced
            issues.append(f'原文對不上大綱「{cn}」，不要硬套 {dst}')
        if _mostly_cjk_episode(transcript) and _foreign_asr_source(src):
            if _rewrites_foreign_asr(src, dub):
                issues.append('原文被辨成外文，不要改成另一句英文')
            else:
                issues.append('原文被辨成外文，不要另造一種語言')
        if not issues:
            continue
        out.append({
            'index': i,
            'start': _line_start(line),
            'speaker': line.get('speaker') or '',
            'source': src,
            'translation': dub,
            'severity': 'high' if (
                broken or forced or any(
                    key in ''.join(issues) for key in ('殘留中文', '沒有配音', '兩人口吻')
                )
            ) else 'mid',
            'issue': '；'.join(issues),
            'suggest': suggest,
            'via': 'rule',
        })
    return out


def _suspect_indices(transcript, language, rules):
    from tools.line_roles import should_skip_dub

    marked = {item['index'] for item in rules}
    for i, line in enumerate(transcript or []):
        if i in marked:
            continue
        src = line.get('text') or ''
        dub = line.get('translation') or ''
        if should_skip_dub(line) and not _needs_empty_vocal(line):
            continue
        if not dub:
            if is_particle_card(src) or src.strip():
                marked.add(i)
            continue
        if leftover_chinese_reason(dub, language):
            marked.add(i)
        if is_particle_card(src):
            marked.add(i)
        if _short_slot_overflow(line, language):
            marked.add(i)
        if _glued_two_mouths(dub) or _speaks_both_mouths(src, dub) or _technique_crushed(src, dub):
            marked.add(i)
        if _guesses_unfinished(src, dub):
            marked.add(i)
        if _sense_broken(src, dub):
            marked.add(i)
        if _mostly_cjk_episode(transcript) and _foreign_asr_source(src):
            marked.add(i)
    return sorted(marked)


def _short_slot_overflow(line, language):
    from tools.translation_quality import _slot_seconds, _spoken_word_budget
    from tools.target_language import uses_word_budget

    dub = (line.get('translation') or '').strip()
    if not dub or not uses_word_budget(language):
        return False
    duration = _slot_seconds(line)
    return len(dub.split()) > _spoken_word_budget(duration, language) + 2


def _review_model_kwargs(timeout=None):
    model = (os.getenv('REVIEW_MODEL_NAME') or '').strip()
    effort = (os.getenv('REVIEW_REASONING_EFFORT') or 'high').strip() or 'high'
    if timeout is None:
        try:
            timeout = float(os.getenv('REVIEW_TIMEOUT') or 180)
        except (TypeError, ValueError):
            timeout = 120
    kwargs = {'reasoning_effort': effort, 'timeout': float(timeout), 'purpose': 'review'}
    if model:
        kwargs['model'] = model
    return kwargs


def _continuity_review(transcript, language, summary, method, progress_callback=None, folder=None):
    if not transcript:
        return []
    from tools.translation_backends import llm_translate
    from tools.translation_bible import bible_context, review_bible_context

    bible = review_bible_context(folder, summary) or bible_context(summary) or ''
    system = (
        review_system_preamble(language) +
        'This pass is whole-episode continuity. '
        'The Chinese source outline is the plot of record. The English outline may omit beats. '
        'Do not change a fact that appears in the Chinese line or source outline '
        'just to match a shorter English outline. '
        'Also flag inconsistent name spellings, wrong addressee, plot contradiction, '
        'or wording the Chinese line does not say. '
        'You MAY flag a wrong speaker ID when Chinese wording + outline + neighbors '
        'show this card cannot be the labeled person '
        '(e.g. "you this X" said by X, a companion tease glued onto a clerk). '
        'When the ID must change, set suggest_speaker to an EXISTING id already in the script '
        '(SPEAKER_A / SPEAKER_B / SPEAKER_SYS / SPEAKER_NARR / …). '
        'Do not invent a new id. Do not swap two leads\' entire IDs. '
        'Do not rewrite a line into another character\'s mouth without changing the id. '
        'Do not re-flag leftover Chinese, particles, ASR junk, or isolated slang. '
        'Cards shorter than 0.3s with mixed script or a single leftover glyph are ASR tears; '
        'do not invent a keep-alive dub for them. '
        'issue must be Traditional Chinese, one short sentence. '
        'Output JSON only: {"findings":[{"index":0,"severity":"high|mid|low","issue":"…","suggest":"…","suggest_speaker":"SPEAKER_A"}]}'
    )
    findings = []
    batches = max(1, (len(transcript) + _CONT_BATCH - 1) // _CONT_BATCH)
    for batch_i, start in enumerate(range(0, len(transcript), _CONT_BATCH), 1):
        chunk = transcript[start:start + _CONT_BATCH]
        allowed = list(range(start, start + len(chunk)))
        _progress(
            progress_callback,
            f'整集連貫 {batch_i}/{batches}（句 #{start}–#{start + len(chunk) - 1}）…',
        )
        payload = [f'Language: {language}', bible or '(no outline)', '', 'Script:']
        for index, line in enumerate(chunk, start=start):
            payload.append(_ship_line_row(index, line, language, transcript, folder))
        try:
            from tools.job_control import check_stop
            check_stop()
            raw = llm_translate(
                method,
                [
                    {'role': 'system', 'content': system},
                    {'role': 'user', 'content': '\n'.join(payload)},
                ],
                **_review_model_kwargs(timeout=180),
            )
        except Exception as exc:
            logger.warning(f'整集連貫審核失敗：{exc}')
            continue
        for item in _parse_ai_findings(raw, allowed):
            item['via'] = 'continuity'
            issue = (item.get('issue') or '').strip()
            known = {str(line.get('speaker') or '') for line in transcript}
            current = ''
            if 0 <= item['index'] < len(transcript):
                current = str(transcript[item['index']].get('speaker') or '')
            dest = _valid_suggest_speaker(item.get('suggest_speaker'), current, known)
            if dest:
                item['suggest_speaker'] = dest
            else:
                item.pop('suggest_speaker', None)
            if _SPEAKER_LABEL_ISSUE.search(issue) and not dest:
                continue
            if issue and not issue.startswith('連貫') and not issue.startswith('講者'):
                item['issue'] = f'連貫：{issue}'
            findings.append(item)
    _progress(progress_callback, f'整集連貫完成，標出 {len(findings)} 條')
    return findings


def _ship_qc_review(transcript, language, summary, method, progress_callback=None, folder=None):
    """Whole-table shipping QC: same view a human editor uses after a first pass."""
    if not transcript:
        return []
    from tools.translation_backends import llm_translate
    from tools.translation_bible import bible_context, review_bible_context

    bible = review_bible_context(folder, summary) or bible_context(summary) or ''
    voices = _voice_table(folder)
    system = (
        review_system_preamble(language) +
        'This pass is shipping QC on the same table the desktop editor uses. '
        'Each row has duration, spoken-word count, max words, optional wav/late, and flags '
        '(RUSH=too many words for the slot, EMPTY=no dub, STUMP/SKIP=ASR scrap, HAN=leftover Chinese, '
        'GLUE=two mouths in one line, TECHNIQUE=spell/move name crushed, '
        'GUESS=trail-off or yes-then-no stammer was finished, '
        'SENSE=dropped verb, gun for a spear, river for a realm, or Heaven without Earth, '
        'JUNK=glossary name stamped on ASR hash, '
        'FOREIGN=Latin-script or kana source in a Chinese episode). '
        'The Chinese source outline is the plot of record. Use the glossary for names. '
        'Flag only problems that would fail a watch-through. '
        'RUSH: rewrite to max words without wrecking the person; Fish is slower than reading. '
        'Do not invent dub for STUMP/SKIP cards. '
        'Do not flag an intentional repeat if the outline says the line is shouted more than once. '
        'Voice-bank role labels are stock-voice age tags, not character sheets; '
        'only mention them if the outline clearly contradicts adult vs child. '
        'Leave suggest empty ONLY for STUMP/SKIP scrap with no honest dub. '
        'issue must be Traditional Chinese, one short sentence starting with 品管： '
        'Output JSON only: {"findings":[{"index":0,"severity":"high|mid|low","issue":"品管：…","suggest":"…","suggest_speaker":"SPEAKER_A"}]}'
    )
    findings = []
    batches = max(1, (len(transcript) + _CONT_BATCH - 1) // _CONT_BATCH)
    for batch_i, start in enumerate(range(0, len(transcript), _CONT_BATCH), 1):
        chunk = transcript[start:start + _CONT_BATCH]
        allowed = list(range(start, start + len(chunk)))
        _progress(
            progress_callback,
            f'整表品管 {batch_i}/{batches}（句 #{start}–#{start + len(chunk) - 1}）…',
        )
        payload = [
            f'Language: {language}',
            bible or '(no outline)',
            voices,
            '',
            'Table:',
        ]
        for index, line in enumerate(chunk, start=start):
            payload.append(_ship_line_row(index, line, language, transcript, folder))
        try:
            from tools.job_control import check_stop
            check_stop()
            raw = llm_translate(
                method,
                [
                    {'role': 'system', 'content': system},
                    {'role': 'user', 'content': '\n'.join(part for part in payload if part is not None)},
                ],
                **_review_model_kwargs(timeout=180),
            )
        except Exception as exc:
            logger.warning(f'整表品管失敗：{exc}')
            continue
        for item in _parse_ai_findings(raw, allowed):
            item['via'] = 'ship'
            issue = (item.get('issue') or '').strip()
            known = {str(line.get('speaker') or '') for line in transcript}
            current = ''
            if 0 <= item['index'] < len(transcript):
                current = str(transcript[item['index']].get('speaker') or '')
            dest = _valid_suggest_speaker(item.get('suggest_speaker'), current, known)
            if dest:
                item['suggest_speaker'] = dest
            else:
                item.pop('suggest_speaker', None)
            if _SPEAKER_LABEL_ISSUE.search(issue) and not dest:
                continue
            if issue and not issue.startswith('品管'):
                item['issue'] = f'品管：{issue}'
            findings.append(item)
    _progress(progress_callback, f'整表品管完成，標出 {len(findings)} 條')
    return findings


def _ai_review(transcript, language, summary, suspects, method, progress_callback=None, folder=None):
    if not suspects:
        _progress(progress_callback, '單句 AI：沒有可疑句，略過')
        return []
    from tools.translation_backends import llm_translate
    from tools.translation_bible import review_bible_context

    system = (
        review_system_preamble(language) +
        'This pass is per-line QC. '
        'Do not treat Japanese kanji as leftover Chinese. '
        'Do not flag speaker IDs or Chinese segmentation. Those are already fixed. '
        'Particles: 呵=ふふ/Hề, 哼=ふん/Hừ, 嘿嘿=へへ/He he, 嗯=ん/Ừ. '
        'Use the episode outline for names. '
        'Leave suggest empty ONLY if the Chinese source is broken ASR and no honest dub exists. '
        'Do not put the explanation in suggest. '
        'Output JSON only: {"findings":[{"index":0,"severity":"high|mid|low","issue":"繁體中文","suggest":"replacement or empty"}]}'
    )
    glossary = (summary or {}).get('glossary') or ''
    title = (summary or {}).get('title') or ''
    plot = review_bible_context(folder, summary)
    findings = []
    batches = max(1, (len(suspects) + _BATCH - 1) // _BATCH)
    for batch_i, start in enumerate(range(0, len(suspects), _BATCH), 1):
        chunk = suspects[start:start + _BATCH]
        _progress(
            progress_callback,
            f'單句 AI {batch_i}/{batches}（句 #{chunk[0]}–#{chunk[-1]}）…',
        )
        payload = [f'Title: {title}', f'Language: {language}', f'Glossary: {glossary}', plot or '', '']
        for index in chunk:
            prev_src = transcript[index - 1].get('text') if index else ''
            next_src = transcript[index + 1].get('text') if index + 1 < len(transcript) else ''
            payload.append(
                _ship_line_row(index, transcript[index], language, transcript, folder)
                + f'\nPREV: {prev_src}\nNEXT: {next_src}'
            )
        try:
            from tools.job_control import check_stop
            check_stop()
            raw = llm_translate(
                method,
                [
                    {'role': 'system', 'content': system},
                    {'role': 'user', 'content': '\n\n'.join(payload)},
                ],
                **_review_model_kwargs(),
            )
        except Exception as exc:
            logger.warning(f'審稿模型失敗：{exc}')
            continue
        findings.extend(_parse_ai_findings(raw, chunk))
    _progress(progress_callback, f'單句 AI 完成，標出 {len(findings)} 條')
    return findings


def _parse_ai_findings(raw, allowed):
    text = (raw or '').strip()
    if text.startswith('```'):
        text = re.sub(r'^```(?:json)?', '', text).rstrip('`').strip()
    start = text.find('{')
    end = text.rfind('}')
    if start < 0 or end <= start:
        return []
    try:
        data = json.loads(text[start:end + 1])
    except Exception:
        logger.warning('審稿回傳不是 JSON，略過這批')
        return []
    rows = data.get('findings') if isinstance(data, dict) else data
    if not isinstance(rows, list):
        return []
    allowed_set = set(allowed)
    out = []
    for item in rows:
        if not isinstance(item, dict):
            continue
        try:
            index = int(item.get('index'))
        except (TypeError, ValueError):
            continue
        if index not in allowed_set:
            continue
        issue = str(item.get('issue') or '').strip()
        if not issue:
            continue
        severity = str(item.get('severity') or 'mid').lower()
        if severity not in _SEVERITY_RANK:
            severity = 'mid'
        row = {
            'index': index,
            'severity': severity,
            'issue': issue,
            'suggest': str(item.get('suggest') or '').strip(),
            'via': 'ai',
        }
        dest = str(item.get('suggest_speaker') or '').strip()
        if dest.startswith('SPEAKER_') or dest == 'NARRATOR':
            row['suggest_speaker'] = dest
        if item.get('merge') not in (None, '', [], ()):
            row['merge'] = item.get('merge')
        if str(item.get('suggest_source') or '').strip():
            row['suggest_source'] = str(item.get('suggest_source')).strip()
        out.append(row)
    return out


def _merge_findings(transcript, rules, ai_findings):
    by_index = {}
    for item in rules:
        by_index[item['index']] = dict(item)
    for item in ai_findings:
        index = item['index']
        line = transcript[index] if 0 <= index < len(transcript) else {}
        current = by_index.get(index)
        if current is None:
            by_index[index] = {
                'index': index,
                'start': _line_start(line),
                'speaker': line.get('speaker') or '',
                'source': line.get('text') or '',
                'translation': line.get('translation') or '',
                'severity': item.get('severity') or 'mid',
                'issue': item.get('issue') or '',
                'suggest': item.get('suggest') or '',
                'via': item.get('via') or 'ai',
            }
            if item.get('suggest_speaker'):
                by_index[index]['suggest_speaker'] = item['suggest_speaker']
            if item.get('merge'):
                by_index[index]['merge'] = item['merge']
            if item.get('suggest_source'):
                by_index[index]['suggest_source'] = item['suggest_source']
            continue
        issues = [current.get('issue') or '', item.get('issue') or '']
        current['issue'] = '；'.join(part for part in issues if part)
        if _SEVERITY_RANK.get(item.get('severity'), 9) < _SEVERITY_RANK.get(current.get('severity'), 9):
            current['severity'] = item.get('severity')
        if item.get('suggest') and not current.get('suggest'):
            current['suggest'] = item['suggest']
        elif item.get('suggest') and item.get('suggest') != current.get('suggest'):
            current['suggest'] = item['suggest']
        if item.get('merge'):
            current['merge'] = item['merge']
        if item.get('suggest_source'):
            current['suggest_source'] = item['suggest_source']
        if item.get('suggest_speaker'):
            current['suggest_speaker'] = item['suggest_speaker']
        current['via'] = 'ai+rule' if current.get('via') == 'rule' else current.get('via') or 'ai'
    return sorted(by_index.values(), key=lambda row: (_SEVERITY_RANK.get(row.get('severity'), 9), row.get('index', 0)))


def _wav_line_stats(folder, index, line):
    from tools.translation import _tts_wav_seconds
    from tools.translation_quality import _slot_seconds

    slot = _slot_seconds(line)
    wav = _tts_wav_seconds(folder, index)
    try:
        placed_start = float(line.get('start') or 0)
    except (TypeError, ValueError):
        placed_start = 0.0
    try:
        orig_start = float(
            line.get('orig_start') if line.get('orig_start') is not None else line.get('start') or 0
        )
    except (TypeError, ValueError):
        orig_start = placed_start
    late = max(0.0, placed_start - orig_start)
    return slot, wav, late


def _after_dub_flags(folder, index, line, language, transcript=None):
    flags = [
        flag for flag in _line_qc_flags(
            line, language, transcript=transcript, glossary=_folder_glossary_pairs(folder),
        ) if flag != 'RUSH'
    ]
    slot, wav, late = _wav_line_stats(folder, index, line)
    particle = is_particle_card(line.get('text'))
    if (
        wav is not None and slot and wav > max(slot * 1.15, slot + 0.35)
        and not particle
    ):
        flags.append('WAV_RUSH')
    if late >= 0.8:
        prev = transcript[index - 1] if transcript and index > 0 else None
        laugh_cascade = bool(prev and is_particle_card(prev.get('text')))
        if particle or (laugh_cascade and 'WAV_RUSH' not in flags):
            pass
        else:
            flags.append('LATE')
    return flags, slot, wav, late


def _after_dub_row(folder, index, line, language, transcript=None):
    from tools.translation_quality import _spoken_word_budget
    from tools.target_language import uses_word_budget

    flags, slot, wav, late = _after_dub_flags(
        folder, index, line, language, transcript=transcript,
    )
    dub = (line.get('translation') or '').strip()
    words = len(dub.split()) if dub else 0
    budget = _spoken_word_budget(slot, language) if uses_word_budget(language) else 0
    wav_s = f'{wav:.2f}' if wav is not None else '?'
    return (
        f'#{index} {line.get("speaker") or ""} slot={slot:.2f}s wav={wav_s}s late={late:.2f}s '
        f'words={words} max={budget} [{",".join(flags) or "-"}] | '
        f'SRC: {line.get("text") or ""} | DUB: {dub}'
    )


def _after_dub_suspects(folder, transcript, language):
    from tools.line_roles import should_skip_dub

    marked = set()
    n = len(transcript or [])
    for i, line in enumerate(transcript or []):
        if should_skip_dub(line) and not _needs_empty_vocal(line):
            continue
        flags, _slot, _wav, _late = _after_dub_flags(
            folder, i, line, language, transcript=transcript,
        )
        if is_particle_card(line.get('text')) and not any(
            flag in flags for flag in ('HAN', 'EMPTY')
        ):
            continue
        if not any(
            flag in flags
            for flag in ('WAV_RUSH', 'LATE', 'HAN', 'EMPTY', 'GLUE', 'FOREIGN', 'TECHNIQUE', 'GUESS', 'SENSE', 'JUNK')
        ):
            continue
        for j in (i - 1, i, i + 1):
            if 0 <= j < n:
                marked.add(j)
    return sorted(marked)


def _after_dub_ai(folder, transcript, language, summary, suspects, method, progress_callback=None):
    if not suspects:
        _progress(progress_callback, '配音後 AI：沒有可疑句，略過')
        return []
    from tools.translation_backends import llm_translate
    from tools.translation_bible import review_bible_context

    system = (
        review_system_preamble(language) +
        'This pass is after Fish. Each row has picture slot, actual WAV length, and late versus the Chinese. '
        'WAV_RUSH: wav longer than the picture — shorten THIS spoken line only if a natural shorter line exists. '
        'Do not rewrite particle-only cards (嘿嘿/呵/哼/嗯/哈哈). The mixer fades those. '
        'LATE: starts after the picture. Shorten only if it is also WAV_RUSH. '
        'If it is late only because a previous laugh overran, leave suggest empty. '
        'HAN: leftover Chinese. EMPTY: fill a short vocalization. '
        'GLUE: do not keep two mouths in one suggest. '
        'FOREIGN: do not invent another language. '
        'TECHNIQUE: restore the distinctive name/clauses; do not crush further. '
        'GUESS: restore the trail-off or yes-then-no stammer; do not replace it with a confession. '
        'SENSE: put the verb back; a spear is not a gun; a realm homophone is not a river; '
        'Heaven and Earth stay together; a titled hero keeps the place. '
        'JUNK: do not stamp a glossary name onto ASR hash. '
        'Do not rewrite a line that already fits (no WAV_RUSH, late < 0.8s) unless '
        'HAN/EMPTY/GLUE/FOREIGN/TECHNIQUE/GUESS/SENSE/JUNK. '
        'Do not flag speaker IDs or Chinese segmentation. '
        'WAV_RUSH and LATE are mid or high, never low. '
        'Leave suggest empty if the only shorter option is a bark or broken English; '
        'the mixer will delay the next line instead. '
        'issue must be Traditional Chinese, one short sentence starting with 配音後： '
        'Output JSON only: {"findings":[{"index":0,"severity":"high|mid|low","issue":"配音後：…","suggest":"…"}]}'
    )
    glossary = (summary or {}).get('glossary') or ''
    title = (summary or {}).get('title') or ''
    plot = review_bible_context(folder, summary)
    findings = []
    batches = max(1, (len(suspects) + _CONT_BATCH - 1) // _CONT_BATCH)
    for batch_i, start in enumerate(range(0, len(suspects), _CONT_BATCH), 1):
        chunk = suspects[start:start + _CONT_BATCH]
        _progress(
            progress_callback,
            f'配音後審稿 {batch_i}/{batches}（句 #{chunk[0]}–#{chunk[-1]}）…',
        )
        payload = [
            f'Title: {title}',
            f'Language: {language}',
            f'Glossary: {glossary}',
            plot or '',
            '',
        ]
        for index in chunk:
            payload.append(
                _after_dub_row(folder, index, transcript[index], language, transcript)
            )
        try:
            from tools.job_control import check_stop
            check_stop()
            raw = llm_translate(
                method,
                [
                    {'role': 'system', 'content': system},
                    {'role': 'user', 'content': '\n'.join(part for part in payload if part)},
                ],
                **_review_model_kwargs(timeout=240),
            )
        except Exception as exc:
            logger.warning(f'配音後審稿失敗：{exc}')
            continue
        for item in _parse_ai_findings(raw, chunk):
            item['via'] = 'after_dub'
            issue = (item.get('issue') or '').strip()
            if issue and not issue.startswith('配音後'):
                item['issue'] = f'配音後：{issue}'
            if _keep_after_dub_finding(folder, transcript, item, language):
                findings.append(item)
    _progress(progress_callback, f'配音後審稿完成，標出 {len(findings)} 條')
    return findings


def _keep_after_dub_finding(folder, transcript, item, language):
    """Drop neighbor nits and bark-style rewrites; layout can delay instead."""
    try:
        index = int(item.get('index'))
    except (TypeError, ValueError):
        return False
    line = transcript[index] if 0 <= index < len(transcript) else {}
    flags, _slot, _wav, _late = _after_dub_flags(
        folder, index, line, language, transcript=transcript,
    )
    suggest = str(item.get('suggest') or '').strip()
    if is_particle_card(line.get('text')) and not any(
        flag in flags for flag in ('HAN', 'EMPTY')
    ):
        return False
    src = line.get('text')
    current = line.get('translation')
    if restores_voice(src, current, suggest, language):
        return True
    if any(flag in flags for flag in ('HAN', 'EMPTY', 'GLUE', 'FOREIGN', 'TECHNIQUE', 'GUESS', 'SENSE', 'JUNK')):
        if suggest and 'EMPTY' not in flags and (
            _glued_two_mouths(suggest)
            or (
                suggest_wrecks_voice(src, current, suggest, language)
                and not restores_voice(src, current, suggest, language)
            )
        ):
            item['suggest'] = ''
        return True
    needs = any(flag in flags for flag in ('WAV_RUSH', 'LATE'))
    if not needs:
        return False
    if not suggest:
        return False
    if suggest_wrecks_voice(src, current, suggest, language) and not restores_voice(
        src, current, suggest, language,
    ):
        logger.info(f'配音後審稿略過沒人味的改法：#{index} → {suggest}')
        return False
    return True


def review_after_dub(folder, language=None, method='OpenAI', progress_callback=None):
    """QC after Fish using real wav length and placed start, not paper word counts."""
    from tools.translation_versions import load_lines_for_language, snapshot_active, write_review_report

    language = translation_language(
        language or load_dub_meta(folder).get('translation') or 'English'
    )
    if is_chinese_target(language):
        return {'language': language, 'stage': 'after_dub', 'findings': []}
    _progress(progress_callback, f'配音後審稿：載入 {language} 譯文與音檔…')
    snapshot_active(folder)
    transcript = load_lines_for_language(folder, language)
    if not transcript:
        raise FileNotFoundError(f'找不到 {language} 譯文可審稿：{folder}')
    from tools.line_roles import should_skip_dub

    suspects = _after_dub_suspects(folder, transcript, language)
    watch = [
        i for i, line in enumerate(transcript)
        if not should_skip_dub(line) or _needs_empty_vocal(line)
    ]
    _progress(
        progress_callback,
        f'配音後審稿：整表 {len(watch)} 句（旗標 {len(suspects)}）',
    )
    ai_findings = []
    if watch and method in {'OpenAI', 'LLM', '阿里云-通义千问', 'Ernie', 'Ollama'}:
        ai_findings = _after_dub_ai(
            folder, transcript, language, _load_summary(folder), watch, method,
            progress_callback=progress_callback,
        )
    findings = [_enrich_finding(item, language) for item in _merge_findings(transcript, [], ai_findings)]
    report = {
        'language': language,
        'reviewed_at': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'stage': 'after_dub',
        'line_count': len(transcript),
        'suspect_count': len(suspects),
        'findings': findings,
    }
    write_review_report(folder, language, report)
    return report


def _after_dub_fingerprint(transcript):
    import hashlib

    parts = []
    for i, line in enumerate(transcript or []):
        parts.append(
            f"{i}\t{line.get('speaker') or ''}\t{(line.get('translation') or '').strip()}"
        )
    return hashlib.sha256('\n'.join(parts).encode('utf-8')).hexdigest()[:16]


def _after_dub_stamp_lang(folder, language):
    return translation_language(
        language or load_dub_meta(folder).get('translation') or 'English'
    )


def after_dub_already_done(folder, language=None):
    """True when this language's current translations already had a post-TTS review."""
    from tools.translation_versions import load_lines_for_language

    lang = _after_dub_stamp_lang(folder, language)
    transcript = load_lines_for_language(folder, lang)
    if not transcript:
        return False
    stamps = load_dub_meta(folder).get('after_dub') or {}
    return str(stamps.get(lang) or '') == _after_dub_fingerprint(transcript)


def stamp_after_dub(folder, language=None):
    """Remember this draft so the next redub does not call the review API again."""
    from tools.translation_versions import load_lines_for_language

    lang = _after_dub_stamp_lang(folder, language)
    transcript = load_lines_for_language(folder, lang)
    if not transcript:
        return
    path = os.path.join(folder, 'dub_meta.json')
    meta = load_dub_meta(folder)
    stamps = dict(meta.get('after_dub') or {})
    stamps[lang] = _after_dub_fingerprint(transcript)
    meta['after_dub'] = stamps
    os.makedirs(folder, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(meta, handle, indent=2, ensure_ascii=False)


def review_and_redub_after_tts(folder, language, method='OpenAI'):
    """One post-TTS QC pass. Never loops. Caller resynths missing wavs once."""
    try:
        if after_dub_already_done(folder, language):
            logger.info(f'配音後審稿：同一稿已審過，略過：{folder}')
            return False
        from tools.cost_tracker import mark_stage
        mark_stage('配音後審稿...')
        report = review_after_dub(folder, language, method=method)
        n = len((report or {}).get('findings') or [])
        picks = auto_apply_review_picks(folder, language)
        if picks:
            _, applied, _skipped, message, _ = apply_selected_review(
                folder, language, picks,
            )
            logger.info(f'配音後審稿標出 {n} 句，已改譯並將重配：{message}')
            return applied > 0
        stamp_after_dub(folder, language)
        if n:
            logger.info(f'配音後審稿標出 {n} 句，沒有可套用的建議：{folder}')
        else:
            logger.info(f'配音後審稿未標異常：{folder}')
        return False
    except Exception as exc:
        from tools.job_control import JobStopped
        if isinstance(exc, JobStopped):
            raise
        logger.warning(f'配音後審稿略過：{exc}')
        return False
