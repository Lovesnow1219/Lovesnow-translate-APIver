# -*- coding: utf-8 -*-
import json
import os
import re

from loguru import logger


def _is_episode_dir(folder):
    return (
        os.path.isfile(os.path.join(folder, 'translation.json'))
        or os.path.isfile(os.path.join(folder, 'summary.json'))
        or os.path.isfile(os.path.join(folder, 'transcript.json'))
        or os.path.isdir(os.path.join(folder, 'translations'))
    )


def episode_folder(folder):
    folder = os.path.normpath(folder or '')
    if _is_episode_dir(folder):
        return folder
    for root, dirs, files in os.walk(folder):
        dirs[:] = [name for name in dirs if name not in {'translations', 'wavs', 'SPEAKER', '__pycache__'}]
        if 'translation.json' in files or 'summary.json' in files or 'transcript.json' in files:
            return root
    return folder


def _load_transcript(folder):
    from tools.target_language import drop_asr_junk_lines, purge_asr_junk_folder

    dropped = purge_asr_junk_folder(folder)
    if dropped:
        logger.info(f'已從講者表清掉 {dropped} 條辨識幻聽：{folder}')
    path = os.path.join(folder, 'translation.json')
    if not os.path.isfile(path):
        raise FileNotFoundError(f'找不到 translation.json：{folder}')
    with open(path, 'r', encoding='utf-8') as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError('translation.json 格式不正確')
    kept, n = drop_asr_junk_lines(data)
    if n:
        data = kept
    return path, data


def _load_summary(folder):
    path = os.path.join(folder, 'summary.json')
    if not os.path.isfile(path):
        return path, {}
    try:
        with open(path, 'r', encoding='utf-8') as handle:
            data = json.load(handle)
        if isinstance(data, dict):
            return path, data
    except Exception:
        pass
    return path, {}


def load_speaker_table(folder, language=None):
    from tools.target_language import load_dub_meta, ui_language_label
    from tools.translation_versions import (
        activate_language,
        has_version,
        list_saved_languages,
        load_lines_for_language,
        snapshot_active,
    )

    folder = episode_folder(folder)
    snapshot_active(folder)
    meta = load_dub_meta(folder)
    lang = language or meta.get('translation') or 'English'
    exists = has_version(folder, lang)
    if exists:
        activate_language(folder, lang)
        transcript = load_lines_for_language(folder, lang)
    else:
        transcript = load_lines_for_language(folder, lang)
    if not transcript:
        raise FileNotFoundError(f'找不到對白：{folder}')
    rows = []
    for i, line in enumerate(transcript):
        start = line.get('orig_start', line.get('start') or 0)
        rows.append([
            '',
            i,
            round(float(start), 2),
            str(line.get('speaker') or 'SPEAKER_00'),
            line.get('text') or '',
            line.get('translation') or '',
            '翻',
        ])
    speakers = sorted({row[3] for row in rows})
    saved = list_saved_languages(folder)
    saved_text = '、'.join(ui_language_label(item) for item in saved) if saved else '尚無'
    note = '' if exists else f'｜{ui_language_label(lang)} 尚未翻譯'
    status = (
        f'{folder}｜語言 {ui_language_label(lang)}｜已有版本 {saved_text}{note}'
        f'｜{len(rows)} 句｜講者 {", ".join(speakers)}'
    )
    return folder, rows, status


def load_bible_fields(folder):
    folder = episode_folder(folder)
    _path, summary = _load_summary(folder)
    locked = '已鎖' if summary.get('outline_locked') else '未鎖'
    status = f'{folder}｜大綱{locked}'
    return (
        summary.get('title') or '',
        summary.get('summary') or '',
        summary.get('outline') or '',
        summary.get('glossary') or '',
        summary.get('voices') or '',
        status,
    )


def save_bible_fields(folder, title, plot, outline, glossary, voices):
    from tools.translation import OUTLINE_VERSION

    folder = episode_folder(folder)
    path, summary = _load_summary(folder)
    summary['title'] = (title or '').strip() or summary.get('title') or os.path.basename(folder)
    summary['summary'] = (plot or '').strip()
    summary['outline'] = (outline or '').strip()
    summary['glossary'] = (glossary or '').strip()
    summary['voices'] = (voices or '').strip()
    summary['outline_locked'] = True
    try:
        current = int(summary.get('outline_version') or 0)
    except (TypeError, ValueError):
        current = 0
    summary['outline_version'] = max(current, OUTLINE_VERSION)
    os.makedirs(folder, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    logger.info(f'已儲存配音大綱：{folder}')
    return folder, f'已儲存大綱並鎖定，之後自動流程不會覆寫。{folder}'


def _rows_from_table(table):
    if table is None:
        return []
    if hasattr(table, 'values'):
        return [list(row) for row in table.values.tolist()]
    return [list(row) for row in table]


_CHECKED_TOKENS = {'true', '1', 'yes', 'y', '☑', '✓', '✔', 'x', 'on', '勾'}


def _looks_like_index(value):
    if isinstance(value, bool):
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return number == int(number) and int(number) >= 0


def _is_checked_value(value):
    if value is True:
        return True
    if value is False or value is None:
        return False
    return str(value).strip().lower() in _CHECKED_TOKENS


def _speaker_row_fields(row):
    if not row:
        return None
    cells = list(row)
    check_first = False
    if len(cells) >= 6:
        check_first = True
    elif len(cells) >= 5 and (
        isinstance(cells[0], bool) or not _looks_like_index(cells[0])
    ):
        check_first = True
    if check_first:
        return {
            'checked': _is_checked_value(cells[0]),
            'index': cells[1] if len(cells) > 1 else None,
            'start': cells[2] if len(cells) > 2 else None,
            'speaker': cells[3] if len(cells) > 3 else '',
            'source': cells[4] if len(cells) > 4 else '',
            'translation': cells[5] if len(cells) > 5 else '',
        }
    return {
        'checked': False,
        'index': cells[0],
        'start': cells[1] if len(cells) > 1 else None,
        'speaker': cells[2] if len(cells) > 2 else '',
        'source': cells[3] if len(cells) > 3 else '',
        'translation': cells[4] if len(cells) > 4 else '',
    }


def save_speaker_table(folder, table, language=None):
    from tools.asr import generate_speaker_audio
    from tools.translation import _clear_line_wavs, _clear_mix_outputs
    from tools.target_language import load_dub_meta, save_dub_meta, translation_language, ui_language_label
    from tools.translation_versions import has_version, load_lines_for_language, snapshot_active, write_translation

    folder = episode_folder(folder)
    snapshot_active(folder)
    lang = translation_language(language or load_dub_meta(folder).get('translation') or 'English')
    transcript = load_lines_for_language(folder, lang)
    if not transcript:
        raise FileNotFoundError(f'找不到對白：{folder}')
    rows = _rows_from_table(table)
    speaker_changed = 0
    translation_changed = 0
    source_changed = 0
    changed_indices = []
    for row in rows:
        fields = _speaker_row_fields(row)
        if not fields:
            continue
        try:
            index = int(float(fields['index']))
        except (TypeError, ValueError):
            continue
        if index < 0 or index >= len(transcript):
            continue
        speaker = str(fields['speaker'] or '').strip() or transcript[index].get('speaker')
        source = '' if fields['source'] is None else str(fields['source']).strip()
        translation = '' if fields['translation'] is None else str(fields['translation']).strip()
        old_speaker = str(transcript[index].get('speaker') or '')
        old_source = str(transcript[index].get('text') or '').strip()
        old_translation = str(transcript[index].get('translation') or '').strip()
        line_changed = False
        if speaker != old_speaker:
            transcript[index]['speaker'] = speaker
            speaker_changed += 1
            line_changed = True
        if source != old_source:
            transcript[index]['text'] = source
            source_changed += 1
            line_changed = True
        if translation != old_translation:
            transcript[index]['translation'] = translation
            translation_changed += 1
            line_changed = True
        if line_changed:
            changed_indices.append(index)
    if not has_version(folder, lang) and not any((line.get('translation') or '').strip() for line in transcript):
        message = f'{ui_language_label(lang)} 還沒有譯文，沒有寫入。請先翻譯或貼上譯文。{folder}'
        logger.info(message)
        return folder, message, False, False
    write_translation(folder, transcript, lang)
    if source_changed:
        _sync_transcript_json(folder, transcript)
    save_dub_meta(folder, speakers_locked=True)
    if speaker_changed:
        voices = os.path.join(folder, 'speaker_voices.json')
        if os.path.isfile(voices):
            os.remove(voices)
        if os.path.isfile(os.path.join(folder, 'audio_vocals.wav')):
            generate_speaker_audio(folder, transcript)
        from tools.target_language import clear_tts_cache
        clear_tts_cache(folder)
    elif translation_changed or source_changed:
        for index in changed_indices:
            _clear_line_wavs(folder, index)
        _clear_mix_outputs(folder)
    parts = []
    if speaker_changed:
        parts.append(f'{speaker_changed} 句講者')
    if source_changed:
        parts.append(f'{source_changed} 句原文')
    if translation_changed:
        parts.append(f'{translation_changed} 句譯文')
    if not parts:
        message = f'沒有改動：{folder}'
    else:
        message = f'已儲存 {folder}，更新 {"、".join(parts)}。按「只重配音」會只重做有改的句子。'
    logger.info(message)
    return folder, message, speaker_changed > 0, translation_changed > 0


def redub_speakers(folder, language, table=None):
    from tools.tts import generate_wavs
    from tools.synthesize import synthesize_video
    from tools.target_language import clear_tts_cache, tts_language, ui_language_label
    from tools.translation_versions import activate_language, has_version

    speaker_changed = False
    if table is not None:
        folder, message, speaker_changed, _trans_changed = save_speaker_table(folder, table, language)
    else:
        folder = episode_folder(folder)
        message = f'使用已儲存的講者：{folder}'
    if not has_version(folder, language):
        return f'{message}\n{ui_language_label(language)} 尚未翻譯，無法配音。', None, None
    activate_language(folder, language)
    if speaker_changed:
        clear_tts_cache(folder)
    from tools.translation import tighten_overlong_lines
    from tools.translation_review import applied_review_indices
    applied_idx = applied_review_indices(folder, language)
    if applied_idx:
        tighten_overlong_lines(folder, language, indices=applied_idx, slack=2)
    tts_lang = tts_language(language)
    result = generate_wavs('Fish', folder, tts_lang)
    if isinstance(result, str):
        return f'{message}\n配音失敗：{result}', None, None
    wav_combined, _wav_ori = result
    status = f'{message}\n配音完成：{wav_combined}'
    video_path = None
    download = os.path.join(folder, 'download.mp4')
    if os.path.isfile(download) and wav_combined:
        video_path = synthesize_video(folder, subtitles=True)
        status += f'\n影片：{video_path}'
    return status, wav_combined, video_path


def retranslate_from_bible(folder, language, title=None, plot=None, outline=None, glossary=None, voices=None):
    from tools.translation import refresh_translation_lines
    from tools.tts import generate_wavs
    from tools.synthesize import synthesize_video
    from tools.target_language import split_target_language

    folder = episode_folder(folder)
    if any(value is not None for value in (title, plot, outline, glossary, voices)):
        folder, bible_status = save_bible_fields(folder, title, plot, outline, glossary, voices)
    else:
        bible_status = f'使用已儲存的大綱：{folder}'
    trans_lang, tts_lang = split_target_language(language)
    refresh_translation_lines(folder, trans_lang)
    result = generate_wavs('Fish', folder, tts_lang)
    if isinstance(result, str):
        return f'{bible_status}\n依大綱重翻後配音失敗：{result}', None, None
    wav_combined, _wav_ori = result
    status = f'{bible_status}\n已依大綱重翻並重配：{wav_combined}'
    video_path = None
    download = os.path.join(folder, 'download.mp4')
    if os.path.isfile(download) and wav_combined:
        video_path = synthesize_video(folder, subtitles=True)
        status += f'\n影片：{video_path}'
    return status, wav_combined, video_path


def line_pick_choices_from_rows(rows):
    choices = []
    for row in rows or []:
        fields = _speaker_row_fields(row)
        if not fields:
            continue
        try:
            index = int(float(fields['index']))
        except (TypeError, ValueError):
            continue
        start = fields['start']
        preview = re.sub(r'\s+', ' ', str(fields['source'] or '')).strip()[:24]
        choices.append((f'{index}｜{start}s｜{preview}', str(index)))
    return choices


def selected_line_indices(selected):
    if selected is None or selected is False:
        return []
    if isinstance(selected, str):
        items = [part for part in re.split(r'[,，、\s]+', selected) if part]
    elif isinstance(selected, (list, tuple, set)):
        items = list(selected)
    else:
        items = [selected]
    out = []
    for item in items:
        text = str(item).strip()
        if not text:
            continue
        token = text.split('｜', 1)[0]
        try:
            out.append(int(float(token)))
        except (TypeError, ValueError):
            continue
    return sorted(set(out))


def checked_indices_from_table(table):
    out = []
    for row in _rows_from_table(table):
        fields = _speaker_row_fields(row)
        if not fields or not fields['checked']:
            continue
        try:
            out.append(int(float(fields['index'])))
        except (TypeError, ValueError):
            continue
    return sorted(set(out))


def resolve_checked_indices(table, selected=None):
    checked = selected_line_indices(selected)
    if checked:
        return checked
    return checked_indices_from_table(table)


def _consecutive_runs(indices):
    runs = []
    for index in indices or []:
        if runs and index == runs[-1][-1] + 1:
            runs[-1].append(index)
        else:
            runs.append([index])
    return runs


def _sync_transcript_json(folder, transcript):
    path = os.path.join(folder, 'transcript.json')
    cards = []
    for line in transcript:
        start = float(line.get('orig_start', line.get('start') or 0))
        end = float(line.get('orig_end', line.get('end') or start))
        cards.append({
            'start': round(start, 3),
            'end': round(end, 3),
            'text': line.get('text') or '',
            'speaker': line.get('speaker') or 'SPEAKER_00',
        })
    os.makedirs(folder, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(cards, handle, indent=4, ensure_ascii=False)


def _reindex_review_after_delete(folder, language, deleted):
    if not deleted:
        return
    from tools.translation_review import _reindex_findings, load_review
    from tools.translation_versions import write_review_report

    report = load_review(folder, language)
    if not report:
        return
    by_index = {}
    for finding in report.get('findings') or []:
        try:
            by_index[int(finding.get('index'))] = finding
        except (TypeError, ValueError):
            continue
    report['findings'] = list(_reindex_findings(by_index, deleted).values())
    write_review_report(folder, language, report)


def retranslate_checked_lines(folder, language, table, selected=None, method='OpenAI'):
    """Merge consecutive checked cards, then re-translate only those hosts."""
    from tools.translation import (
        _clear_line_wavs,
        _clear_mix_outputs,
        _dubbing_fixed_message,
        _translate_one,
        _usable_target_text,
    )
    from tools.target_language import (
        clear_tts_cache,
        load_dub_meta,
        translation_language,
        ui_language_label,
    )
    from tools.translation_quality import _forbids_chinese_script
    from tools.translation_review import _join_lines
    from tools.translation_versions import load_lines_for_language, snapshot_active, write_translation
    from tools.vocal_particles import ensure_particle_translation_lead

    checked = resolve_checked_indices(table, selected)
    folder, message, *_ = save_speaker_table(folder, table, language)
    snapshot_active(folder)
    lang = translation_language(language or load_dub_meta(folder).get('translation') or 'English')
    transcript = load_lines_for_language(folder, lang)
    if not transcript:
        raise FileNotFoundError(f'找不到 {lang} 譯文：{folder}')
    checked = [index for index in checked if 0 <= index < len(transcript)]
    if not checked:
        return folder, '請先在對白表最左欄打勾要處理的句子。', False
    runs = _consecutive_runs(checked)
    deleted = []
    hosts = []
    merged_hosts = set()
    merged_n = 0
    for run in reversed(runs):
        host = run[0]
        absorbed = run[1:]
        if absorbed:
            _join_lines(transcript, host, absorbed, '')
            deleted.extend(absorbed)
            merged_n += len(absorbed)
            merged_hosts.add(host)
        hosts.append(host)
    for index in sorted(set(deleted), reverse=True):
        del transcript[index]
    hosts = sorted({index - sum(1 for gone in deleted if gone < index) for index in hosts})
    merged_hosts = {
        index - sum(1 for gone in deleted if gone < index)
        for index in merged_hosts
    }
    _sync_transcript_json(folder, transcript)
    _reindex_review_after_delete(folder, lang, deleted)
    if deleted:
        clear_tts_cache(folder, keep_video=True)
    _path, summary = _load_summary(folder)
    fixed_message = _dubbing_fixed_message(summary, lang)
    merge_note = (
        'Adjacent subtitle cards were merged into one Chinese line. '
        'Translate this complete line from Chinese. Do not reuse the old split translation. '
    )
    done = 0
    failed = []
    for index in hosts:
        line = transcript[index]
        line['translation'] = ''
        source = (line.get('text') or '').strip()
        logger.info(f'依原文重翻第 {index} 句：{source[:80]}')
        raw, _user = _translate_one(
            summary, line, lang, method, fixed_message, history=[],
            budget_scale=1.15 if index in merged_hosts else 1.0,
            extra_note=merge_note if index in merged_hosts else '',
        )
        if _forbids_chinese_script(lang):
            raw = _usable_target_text(raw, lang)
        if raw:
            line['translation'] = raw
            ensure_particle_translation_lead(line, lang)
            done += 1
        else:
            failed.append(index)
        if not deleted:
            _clear_line_wavs(folder, index)
    write_translation(folder, transcript, lang)
    _clear_mix_outputs(folder, keep_video=True)
    parts = [f'已依合併後原文重翻 {done} 句{ui_language_label(lang)}' if merged_n else f'已重翻 {done} 句{ui_language_label(lang)}']
    if merged_n:
        parts.append(f'合併刪掉 {merged_n} 張卡片，時間已併到留下的那一句')
    if failed:
        parts.append(f'第 {", ".join(str(i) for i in failed)} 句重翻失敗，請再按一次')
        logger.error(f'勾選句重翻失敗：{failed}')
    parts.append('尚未重配。要聽新聲音請按「只重配音」。')
    status = '。'.join(parts)
    logger.info(status)
    return folder, f'{message}\n{status}', done > 0


def parse_one_line_payload(payload):
    text = '' if payload is None else str(payload)
    if '\n' in text:
        idx_text, source = text.split('\n', 1)
    else:
        idx_text, source = text, ''
    try:
        index = int(float(idx_text.strip()))
    except (TypeError, ValueError):
        return None, ''
    return index, source.strip()


def retranslate_one_line(folder, language, payload, method='OpenAI'):
    """Retranslate one card from the Chinese currently in the table / payload."""
    from tools.translation import (
        _clear_line_wavs,
        _clear_mix_outputs,
        _dubbing_fixed_message,
        _translate_one,
        _usable_target_text,
    )
    from tools.target_language import load_dub_meta, translation_language, ui_language_label
    from tools.translation_quality import _forbids_chinese_script
    from tools.translation_versions import load_lines_for_language, snapshot_active, write_translation
    from tools.vocal_particles import ensure_particle_translation_lead

    index, source = parse_one_line_payload(payload)
    if index is None:
        return folder, '請按該列右邊的「單句翻」。', False
    if not source:
        return folder, '這一句沒有中文原文，請先改「原文」再按單句翻。', False
    folder = episode_folder(folder)
    snapshot_active(folder)
    lang = translation_language(language or load_dub_meta(folder).get('translation') or 'English')
    transcript = load_lines_for_language(folder, lang)
    if not transcript:
        raise FileNotFoundError(f'找不到 {lang} 譯文：{folder}')
    if index < 0 or index >= len(transcript):
        return folder, f'沒有第 {index} 句。', False
    line = transcript[index]
    old_source = (line.get('text') or '').strip()
    line['text'] = source
    line['translation'] = ''
    _path, summary = _load_summary(folder)
    extra_note = (
        'The user edited this Chinese line. Translate this exact Chinese. '
        'Do not reuse the previous translation.'
    )
    raw, _user = _translate_one(
        summary, line, lang, method, _dubbing_fixed_message(summary, lang),
        history=[], extra_note=extra_note,
    )
    if _forbids_chinese_script(lang):
        raw = _usable_target_text(raw, lang)
    if not raw:
        line['text'] = old_source or source
        write_translation(folder, transcript, lang)
        return folder, f'第 {index} 句重翻失敗，請再按一次單句翻。', False
    line['translation'] = raw
    ensure_particle_translation_lead(line, lang)
    write_translation(folder, transcript, lang)
    if source != old_source:
        _sync_transcript_json(folder, transcript)
    _clear_line_wavs(folder, index)
    _clear_mix_outputs(folder, keep_video=True)
    status = (
        f'已依修改後原文重翻第 {index} 句{ui_language_label(lang)}。'
        '尚未重配。要聽新聲音請按「只重配音」。'
    )
    logger.info(f'{status}｜{source[:80]}')
    return folder, status, True
