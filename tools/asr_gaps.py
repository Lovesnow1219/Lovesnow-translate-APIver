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
_MAX_CLUSTER = 3.2
_MIN_PEAK = 0.32
_MERGE = 0.75
_MAX_CLIPS = 12


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


def _transcribe_clip(samples, sample_rate, start, end):
    clip = np.asarray(samples[int(start * sample_rate):int(end * sample_rate)], dtype=np.float32)
    if clip.size < int(0.20 * sample_rate):
        return ''
    handle = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    handle.close()
    try:
        import soundfile as sf
        sf.write(handle.name, clip, sample_rate)
        from tools.step023_asr_openai import _transcribe_file
        model = os.getenv('OPENAI_ASR_GAP_MODEL') or 'gpt-4o-transcribe'
        segments = _transcribe_file(handle.name, model)
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


def recover_missing_speech(folder, transcript):
    """Insert cards for spoken holes the first ASR pass dropped."""
    if not transcript:
        return list(transcript or [])
    wav_path = os.path.join(folder, 'audio_vocals.wav')
    if not os.path.isfile(wav_path):
        return [dict(item) for item in transcript]
    try:
        import librosa
        samples, sample_rate = librosa.load(wav_path, sr=16000, mono=True)
    except Exception as exc:
        logger.warning(f'無法掃描漏句空隙：{exc}')
        return [dict(item) for item in transcript]

    from tools.asr_source import cleanup_asr_source
    from tools.target_language import is_asr_junk

    lines = [dict(item) for item in transcript]
    added = []
    for start, end, _peak in uncovered_speech_clusters(samples, sample_rate, lines):
        if any(abs(card_start(item) - start) < 0.12 for item in lines + added):
            continue
        raw = _transcribe_clip(samples, sample_rate, start, end)
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
