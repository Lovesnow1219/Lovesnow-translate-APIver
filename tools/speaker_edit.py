# -*- coding: utf-8 -*-
import json
import os

from loguru import logger


def episode_folder(folder):
    folder = os.path.normpath(folder or '')
    if os.path.isfile(os.path.join(folder, 'translation.json')) or os.path.isfile(os.path.join(folder, 'summary.json')):
        return folder
    for root, _dirs, files in os.walk(folder):
        if 'translation.json' in files or 'summary.json' in files:
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


def load_speaker_table(folder):
    folder = episode_folder(folder)
    _path, transcript = _load_transcript(folder)
    rows = []
    for i, line in enumerate(transcript):
        start = line.get('orig_start', line.get('start') or 0)
        rows.append([
            i,
            round(float(start), 2),
            str(line.get('speaker') or 'SPEAKER_00'),
            line.get('text') or '',
            line.get('translation') or '',
        ])
    speakers = sorted({row[2] for row in rows})
    return folder, rows, f'{folder}｜{len(rows)} 句｜講者 {", ".join(speakers)}'


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
    from tools.step030_translation import OUTLINE_VERSION

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


def save_speaker_table(folder, table):
    from tools.step020_asr import generate_speaker_audio
    from tools.step030_translation import _clear_line_wavs, _clear_mix_outputs
    from tools.target_language import save_dub_meta

    folder = episode_folder(folder)
    path, transcript = _load_transcript(folder)
    rows = _rows_from_table(table)
    speaker_changed = 0
    translation_changed = 0
    changed_indices = []
    for row in rows:
        if not row:
            continue
        try:
            index = int(float(row[0]))
        except (TypeError, ValueError, IndexError):
            continue
        if index < 0 or index >= len(transcript):
            continue
        speaker = str(row[2] if len(row) > 2 else '').strip() or transcript[index].get('speaker')
        translation = '' if len(row) < 5 or row[4] is None else str(row[4]).strip()
        old_speaker = str(transcript[index].get('speaker') or '')
        old_translation = str(transcript[index].get('translation') or '').strip()
        line_changed = False
        if speaker != old_speaker:
            transcript[index]['speaker'] = speaker
            speaker_changed += 1
            line_changed = True
        if translation != old_translation:
            transcript[index]['translation'] = translation
            translation_changed += 1
            line_changed = True
        if line_changed:
            changed_indices.append(index)
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(transcript, handle, indent=2, ensure_ascii=False)
    save_dub_meta(folder, speakers_locked=True)
    if speaker_changed:
        voices = os.path.join(folder, 'speaker_voices.json')
        if os.path.isfile(voices):
            os.remove(voices)
        if os.path.isfile(os.path.join(folder, 'audio_vocals.wav')):
            generate_speaker_audio(folder, transcript)
        from tools.target_language import clear_tts_cache
        clear_tts_cache(folder)
    elif translation_changed:
        for index in changed_indices:
            _clear_line_wavs(folder, index)
        _clear_mix_outputs(folder)
    parts = []
    if speaker_changed:
        parts.append(f'{speaker_changed} 句講者')
    if translation_changed:
        parts.append(f'{translation_changed} 句譯文')
    if not parts:
        message = f'沒有改動：{folder}'
    else:
        message = f'已儲存 {folder}，更新 {"、".join(parts)}。按「只重配音」會只重做有改的句子。'
    logger.info(message)
    return folder, message, speaker_changed > 0, translation_changed > 0


def redub_speakers(folder, language, table=None):
    from tools.step040_tts import generate_wavs
    from tools.step050_synthesize_video import synthesize_video
    from tools.target_language import clear_tts_cache, tts_language

    speaker_changed = False
    if table is not None:
        folder, message, speaker_changed, _trans_changed = save_speaker_table(folder, table)
    else:
        folder = episode_folder(folder)
        message = f'使用已儲存的講者：{folder}'
    if speaker_changed:
        clear_tts_cache(folder)
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
    from tools.step030_translation import refresh_translation_lines
    from tools.step040_tts import generate_wavs
    from tools.step050_synthesize_video import synthesize_video
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
