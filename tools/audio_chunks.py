# -*- coding: utf-8 -*-
"""Split and stitch long audio so cloud APIs stay under size/duration caps."""
import json
import os
import subprocess

import numpy as np
from loguru import logger


def media_duration(path):
    if not path or not os.path.exists(path):
        return 0.0
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'json', path],
            capture_output=True, text=True, check=False,
        )
        data = json.loads(result.stdout or '{}')
        return max(0.0, float((data.get('format') or {}).get('duration') or 0))
    except Exception:
        return 0.0


def plan_chunks(duration, hop, overlap=1.0):
    total = max(0.0, float(duration or 0))
    hop = max(30.0, float(hop))
    overlap = max(0.0, min(float(overlap), hop / 4.0))
    if total <= hop + 1e-6:
        return [(0.0, total)] if total > 0 else []
    chunks = []
    index = 0
    while True:
        start = 0.0 if index == 0 else max(0.0, index * hop - overlap)
        end = min(total, (index + 1) * hop + overlap)
        if end <= start + 0.05:
            break
        chunks.append((round(start, 3), round(end, 3)))
        if end >= total - 1e-6:
            break
        index += 1
        if index > 2000:
            break
    return chunks


def plan_chunks_for_file(path, hop, overlap=1.0, max_bytes=None, sample_rate=44100, channels=2):
    duration = media_duration(path)
    hop = float(hop)
    if max_bytes:
        bytes_per_sec = max(1, int(sample_rate) * int(channels) * 2)
        max_seconds = max(60.0, (float(max_bytes) / bytes_per_sec) - float(overlap) - 1.0)
        hop = min(hop, max_seconds)
        try:
            size = os.path.getsize(path)
        except OSError:
            size = 0
        if size > float(max_bytes) and duration > hop:
            parts = max(2, int(size / float(max_bytes)) + 1)
            hop = min(hop, max(60.0, duration / parts + float(overlap)))
    return plan_chunks(duration, hop, overlap), duration


def keep_chunk_segment(abs_start, abs_end, chunk_start, chunk_end, overlap, is_first, is_last):
    """Keep a transcript line if its midpoint sits in this chunk's unique span."""
    mid = (float(abs_start) + float(abs_end)) / 2.0
    lo = chunk_start if is_first else chunk_start + float(overlap)
    if is_last:
        return mid >= lo
    hi = chunk_end - float(overlap)
    return lo <= mid < hi


def cut_audio(src, dest, start, end, sample_rate=None, channels=None):
    os.makedirs(os.path.dirname(dest) or '.', exist_ok=True)
    cmd = ['ffmpeg', '-y', '-i', src, '-ss', f'{start:.3f}', '-to', f'{end:.3f}']
    if sample_rate:
        cmd.extend(['-ar', str(int(sample_rate))])
    if channels:
        cmd.extend(['-ac', str(int(channels))])
    cmd.extend(['-c:a', 'pcm_s16le', dest])
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0 or not os.path.isfile(dest):
        raise RuntimeError(f'ffmpeg 切段失敗: {(result.stderr or "")[-300:]}')
    return dest


def stitch_wavs(parts, dest, total_duration, sample_rate, overlap_seconds=2.0):
    """parts: list of (start, end, wav_path). Overlap-add with linear fade."""
    import soundfile as sf
    try:
        import librosa
    except Exception:
        librosa = None
    acc = None
    weight = None
    fade = max(1, int(round(float(overlap_seconds) * sample_rate)))
    total = max(1, int(round(float(total_duration) * sample_rate)))
    for start, end, path in parts:
        if librosa is not None:
            audio, _sr = librosa.load(path, sr=sample_rate, mono=False)
            if audio.ndim == 1:
                audio = audio.reshape(1, -1)
            audio = audio.T
        else:
            audio, sr = sf.read(path, always_2d=True)
            if sr != sample_rate:
                raise RuntimeError(f'取樣率不符: {sr} vs {sample_rate}')
        piece = np.asarray(audio, dtype=np.float64)
        if acc is None:
            acc = np.zeros((total, piece.shape[1]), dtype=np.float64)
            weight = np.zeros((total, 1), dtype=np.float64)
        start_i = max(0, int(round(start * sample_rate)))
        count = min(piece.shape[0], total - start_i)
        if count <= 0:
            continue
        piece = piece[:count]
        window = np.ones((count, 1), dtype=np.float64)
        if start > 0.01 and fade > 1:
            fade_n = min(fade, count)
            window[:fade_n, 0] = np.linspace(0.0, 1.0, fade_n)
        if end < float(total_duration) - 0.01 and fade > 1:
            fade_n = min(fade, count)
            window[-fade_n:, 0] = np.linspace(1.0, 0.0, fade_n)
        acc[start_i:start_i + count] += piece * window
        weight[start_i:start_i + count] += window
    if acc is None:
        raise RuntimeError('沒有可拼接的音訊段')
    mixed = np.clip(acc / np.maximum(weight, 1e-8), -1.0, 1.0).astype(np.float32)
    os.makedirs(os.path.dirname(dest) or '.', exist_ok=True)
    from tools.utils import save_wav
    save_wav(mixed if mixed.shape[1] > 1 else mixed[:, 0], dest, sample_rate=sample_rate)
    logger.info(f'已拼接 {len(parts)} 段 -> {dest}')
    return dest
