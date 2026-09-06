# -*- coding: utf-8 -*-
"""Re-transcribe vocal holes that the first ASR pass skipped."""
import os
import tempfile

import numpy as np
from loguru import logger

from tools.vocal_particles import (
    _bursts,
    _nearest_speaker,
    _overlap,
    card_end,
    card_start,
    is_particle_card,
)

_MIN_HOLE = 0.45
_MIN_CLUSTER = 0.28
_MAX_CLUSTER = 12.0
_MIN_PEAK = 0.32
_MERGE = 0.75
_MAX_CLIPS = 20
_LEADING_MIN = 1.8
_LEADING_CHUNK = 16.0


def _windows(transcript):
    return [(card_start(line), card_end(line)) for line in transcript or []]


def uncovered_speech_clusters(samples, sample_rate, transcript):
    """Voiced clusters that sit in a hole between existing cards."""
    cards = _windows(transcript)
    cards.sort()
    holes = []
    cursor = 0.0
    duration = len(samples) / float(sample_rate)
    for start, end in cards:
        if start - cursor >= _MIN_HOLE:
            holes.append((cursor, start))
        cursor = max(cursor, end)
    if duration - cursor >= _MIN_HOLE:
        holes.append((cursor, duration))

    bursts = [
        (start, end, peak)
        for start, end, peak in _bursts(samples, sample_rate)
        if end - start >= 0.12 and peak >= 0.18
    ]
    clusters = []
    for hole_start, hole_end in holes:
        inside = [
            item for item in bursts
            if item[0] < hole_end - 0.02 and item[1] > hole_start + 0.02
        ]
        if not inside:
            continue
        inside.sort()
        group = [inside[0]]
        flushed = []

        def flush():
            start = group[0][0]
            end = max(item[1] for item in group)
            peak = max(item[2] for item in group)
            flushed.append((start, end, peak))

        for item in inside[1:]:
            if item[0] - group[-1][1] <= _MERGE:
                group.append(item)
            else:
                flush()
                group = [item]
        flush()
        for start, end, peak in flushed:
            if end - start < _MIN_CLUSTER or end - start > _MAX_CLUSTER:
                continue
            if peak < _MIN_PEAK:
                continue
            covered = sum(_overlap(start, end, a, b) for a, b in cards)
            if covered / max(end - start, 0.01) >= 0.40:
                continue
            clusters.append((
                max(hole_start, start - 0.05),
                min(hole_end, end + 0.08),
                peak,
            ))
    clusters.sort()
    return clusters[:_MAX_CLIPS]


def _transcribe_clip(samples, sample_rate, start, end, language=None):
    clip = np.asarray(samples[int(start * sample_rate):int(end * sample_rate)], dtype=np.float32)
    if clip.size < int(0.20 * sample_rate):
        return ''
    handle = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    handle.close()
    try:
        import soundfile as sf
        sf.write(handle.name, clip, sample_rate)
        from tools.asr_openai import _transcribe_file
        model = os.getenv('OPENAI_ASR_GAP_MODEL') or 'gpt-4o-transcribe'
        segments = _transcribe_file(handle.name, model, language=language)
    except Exception as exc:
        logger.warning(f'空隙辨識失敗 {start:.2f}-{end:.2f}s：{exc}')
        return ''
    finally:
        try:
            os.remove(handle.name)
        except OSError:
            pass
    parts = [(item.get('text') or '').strip() for item in segments or []]
    return ''.join(parts).strip()


def _load_mono(path):
    import librosa
    return librosa.load(path, sr=16000, mono=True)


def _han_n(text):
    return sum(1 for ch in (text or '') if '\u4e00' <= ch <= '\u9fff')


def _is_substantial(line):
    text = (line.get('text') if isinstance(line, dict) else line) or ''
    if is_particle_card(text):
        return False
    return _han_n(text) >= 8


def _first_substantial_start(transcript, duration):
    for line in sorted(transcript or [], key=card_start):
        if _is_substantial(line):
            return card_start(line)
    return duration


def _leading_chunks(transcript, duration):
    """0 → first real line. A 3-character stub at 0s does not count as coverage."""
    if duration < _LEADING_MIN:
        return []
    first = _first_substantial_start(transcript, duration)
    if first < _LEADING_MIN:
        return []
    chunks = []
    cursor = 0.0
    while cursor < first - 0.35:
        nxt = min(first, cursor + _LEADING_CHUNK)
        chunks.append((cursor, nxt))
        cursor = nxt
    return chunks


def _rms(samples, sample_rate, start, end):
    i0 = int(max(0, start) * sample_rate)
    i1 = int(min(len(samples), end * sample_rate))
    if i1 - i0 < int(0.20 * sample_rate):
        return 0.0
    clip = np.asarray(samples[i0:i1], dtype=np.float32)
    return float(np.sqrt(np.mean(np.square(clip))))


def _mix_tracks(folder):
    tracks = []
    for name in ('audio.wav', 'audio_instruments.wav'):
        path = os.path.join(folder, name)
        if not os.path.isfile(path):
            continue
        try:
            tracks.append(_load_mono(path))
        except Exception as exc:
            logger.warning(f'開頭旁白備援音軌讀不到 {name}：{exc}')
    return tracks


def _spoken_zh(raw):
    from tools.asr_source import cleanup_asr_source
    from tools.target_language import is_asr_junk

    text = cleanup_asr_source(raw)
    compact = ''.join(ch for ch in (text or '') if not ch.isspace())
    if not compact or is_asr_junk(text) or is_particle_card(text):
        return ''
    if not any('\u4e00' <= ch <= '\u9fff' for ch in compact):
        return ''
    return text


def _transcribe_leading(samples, sample_rate, start, end, language, extras):
    sources = [(samples, sample_rate)]
    sources.extend(extras)
    for wav, rate in sources:
        if _rms(wav, rate, start, end) < 0.008:
            continue
        text = _spoken_zh(_transcribe_clip(wav, rate, start, end, language=language))
        if text:
            return text
    return ''


def _apply_leading(folder, lines, samples, sample_rate, language):
    from tools.line_roles import SPEAKER_NARR

    duration = len(samples) / float(sample_rate)
    chunks = _leading_chunks(lines, duration)
    if not chunks:
        return lines, []
    until = chunks[-1][1]
    extras = _mix_tracks(folder)
    recovered = []
    for start, end in chunks:
        text = _transcribe_leading(samples, sample_rate, start, end, language, extras)
        if not text:
            continue
        recovered.append({
            'start': round(start, 3),
            'end': round(max(end, start + 0.28), 3),
            'orig_start': round(start, 3),
            'orig_end': round(end, 3),
            'text': text,
            'speaker': SPEAKER_NARR,
        })
    if not recovered:
        return lines, []
    old_han = sum(
        _han_n(item.get('text'))
        for item in lines
        if card_start(item) < until
    )
    new_han = sum(_han_n(item.get('text')) for item in recovered)
    if new_han <= old_han + 2:
        return lines, []
    kept = [
        item for item in lines
        if card_start(item) >= until - 0.02 or _is_substantial(item)
    ]
    logger.info(
        '開頭短卡重聽：{} → {}'.format(
            old_han,
            '、'.join(item['text'][:16] for item in recovered),
        )
    )
    return kept, recovered


def recover_leading_speech(folder, transcript, language=None):
    """Re-transcribe an opening that only got a short stub (speech in the music bed)."""
    if not transcript:
        return list(transcript or [])
    wav_path = os.path.join(folder, 'audio_vocals.wav')
    if not os.path.isfile(wav_path):
        return [dict(item) for item in transcript]
    try:
        samples, sample_rate = _load_mono(wav_path)
    except Exception as exc:
        logger.warning(f'無法重聽開頭：{exc}')
        return [dict(item) for item in transcript]
    from tools.target_language import load_dub_meta

    lang = language or (load_dub_meta(folder) or {}).get('asr_language')
    lines = [dict(item) for item in transcript]
    lines, added = _apply_leading(folder, lines, samples, sample_rate, lang)
    if added:
        lines.extend(added)
        lines.sort(key=card_start)
    return lines


def recover_missing_speech(folder, transcript, language=None):
    """Insert cards for spoken holes the first ASR pass dropped."""
    if not transcript:
        return list(transcript or [])
    wav_path = os.path.join(folder, 'audio_vocals.wav')
    if not os.path.isfile(wav_path):
        return [dict(item) for item in transcript]
    try:
        samples, sample_rate = _load_mono(wav_path)
    except Exception as exc:
        logger.warning(f'無法掃描漏句空隙：{exc}')
        return [dict(item) for item in transcript]

    from tools.asr_source import cleanup_asr_source
    from tools.target_language import is_asr_junk, load_dub_meta

    lang = language or (load_dub_meta(folder) or {}).get('asr_language')
    lines = [dict(item) for item in transcript]
    lines, added = _apply_leading(folder, lines, samples, sample_rate, lang)
    for start, end, _peak in uncovered_speech_clusters(samples, sample_rate, lines + added):
        if any(abs(card_start(item) - start) < 0.12 for item in lines + added):
            continue
        raw = _transcribe_clip(samples, sample_rate, start, end, language=lang)
        low = ''.join(ch for ch in (raw or '') if ch.strip() and ch not in '，,。．.!！？?、…').lower()
        if low in {'hehe', 'heehee', 'hehheh'}:
            raw = '嘿嘿'
        elif low in {'haha', 'hahaha'}:
            raw = '哈哈'
        text = cleanup_asr_source(raw)
        compact = ''.join(ch for ch in (text or '') if not ch.isspace())
        duration = max(0.01, end - start)
        has_cjk = any('\u4e00' <= ch <= '\u9fff' for ch in compact)
        if not compact or is_asr_junk(text):
            continue
        if is_particle_card(text):
            continue
        if not has_cjk:
            continue
        if len(compact) > duration * 8 + 2:
            continue
        neighbors = [
            (item.get('text') or '')
            for item in lines
            if abs(card_start(item) - start) < 2.5 or abs(card_end(item) - end) < 2.5
        ]
        if any(compact in ''.join(part.split()) or ''.join(part.split()) in compact for part in neighbors if part):
            continue
        added.append({
            'start': round(start, 3),
            'end': round(max(end, start + 0.28), 3),
            'orig_start': round(start, 3),
            'orig_end': round(end, 3),
            'text': text,
            'speaker': _nearest_speaker(lines, (start + end) / 2),
        })
    if added:
        lines.extend(added)
        lines.sort(key=card_start)
        logger.info(
            '補上 {} 句被辨識漏掉的對白：{}'.format(
                len(added),
                '、'.join(f"{item['text'][:12]}@{item['start']:.1f}s" for item in added),
            )
        )
    return lines
