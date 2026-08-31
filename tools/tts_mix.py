# -*- coding: utf-8 -*-
"""Fit line audio, duck BGM, and assemble the dubbed timeline."""
import json
import os

import librosa
from loguru import logger
import numpy as np

from .utils import save_wav
from tools.target_language import AUDIO_LAYOUT_VERSION, AUDIO_MIX_VERSION, save_dub_meta


def mix_speech_and_bed(speech, bed, ceiling=0.99):
    """Keep accompaniment level; duck speech only where the sum would clip."""
    speech = np.asarray(speech, dtype=np.float64).reshape(-1)
    bed = np.asarray(bed, dtype=np.float64).reshape(-1)
    n = max(len(speech), len(bed))
    if len(speech) < n:
        speech = np.pad(speech, (0, n - len(speech)))
    if len(bed) < n:
        bed = np.pad(bed, (0, n - len(bed)))
    mix = speech + bed
    if n == 0:
        return mix.astype(np.float32)
    peak = float(np.max(np.abs(mix)))
    if peak <= ceiling:
        return mix.astype(np.float32)
    overflow = np.abs(mix) > ceiling
    target = np.sign(mix) * ceiling
    speech = speech.copy()
    speech[overflow] = target[overflow] - bed[overflow]
    mix = speech + bed
    mix_peak = float(np.max(np.abs(mix)))
    if mix_peak > ceiling:
        mix = mix * (ceiling / mix_peak)
    return mix.astype(np.float32)


def _speech_rms(wav, floor=0.02):
    wav = np.asarray(wav, dtype=np.float64).reshape(-1)
    if wav.size == 0:
        return 0.0
    active = wav[np.abs(wav) > floor]
    if active.size < 80:
        return float(np.sqrt(np.mean(np.square(wav))))
    return float(np.sqrt(np.mean(np.square(active))))


def compress_speech(wav, sample_rate=24000, threshold=0.16, ratio=3.0, attack_ms=8, release_ms=60):
    wav = np.asarray(wav, dtype=np.float64).reshape(-1)
    if wav.size == 0:
        return wav.astype(np.float32)
    from scipy.ndimage import maximum_filter1d, uniform_filter1d
    env = maximum_filter1d(np.abs(wav), size=max(1, int(sample_rate * attack_ms / 1000.0)))
    env = uniform_filter1d(env, size=max(1, int(sample_rate * release_ms / 1000.0)))
    over = np.maximum(env, 1e-8)
    compressed = threshold + (over - threshold) / ratio
    gain = np.where(over > threshold, compressed / over, 1.0)
    return (wav * gain).astype(np.float32)


def match_line_loudness(tts, original=None, ceiling=0.95):
    tts = compress_speech(tts)
    tts_rms = _speech_rms(tts)
    if tts_rms < 1e-5:
        return np.asarray(tts, dtype=np.float32)
    orig_rms = _speech_rms(original) if original is not None and len(original) else 0.0
    if orig_rms < 0.03:
        orig_rms = max(tts_rms, 0.08)
    scale = min(orig_rms / tts_rms, 10.0 ** (12.0 / 20.0))
    out = np.asarray(tts, dtype=np.float64) * scale
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > ceiling:
        out = out * (ceiling / peak)
    return out.astype(np.float32)


def limit_peak(wav, ceiling=0.95):
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    if wav.size == 0:
        return wav
    peak = float(np.max(np.abs(wav)))
    if peak > ceiling:
        wav = wav * (ceiling / peak)
    return wav.astype(np.float32)


def _audio_rms(chunk, min_samples):
    if chunk.size < min_samples:
        return None
    return float(np.sqrt(np.mean(np.square(np.asarray(chunk, dtype=np.float64)))))


def estimate_dialogue_duck_db(audio, transcript, sample_rate=24000):
    """How many dB quieter the bed is during a line vs the short gap after it."""
    audio = np.asarray(audio, dtype=np.float64).reshape(-1)
    if audio.size == 0 or not transcript or len(transcript) < 2:
        return 0.0
    min_samples = int(0.08 * sample_rate)
    ducks = []
    for i, line in enumerate(transcript[:-1]):
        start = float(line.get('orig_start', line.get('start') or 0))
        end = float(line.get('orig_end', line.get('end') or 0))
        nxt = float(transcript[i + 1].get('orig_start', transcript[i + 1].get('start') or 0))
        if end <= start or nxt - end < 0.18 or nxt - end > 1.8:
            continue
        line_rms = _audio_rms(audio[int(start * sample_rate):int(end * sample_rate)], min_samples)
        gap_rms = _audio_rms(audio[int(end * sample_rate):int(nxt * sample_rate)], min_samples)
        if not line_rms or not gap_rms:
            continue
        ducks.append(20.0 * np.log10(gap_rms / max(line_rms, 1e-8)))
    if len(ducks) < 8:
        return 0.0
    return float(np.median(ducks))


def original_mix_duck_db(folder, transcript, sample_rate=24000):
    mix_path = os.path.join(folder, 'audio.wav')
    voc_path = os.path.join(folder, 'audio_vocals.wav')
    if not transcript or not os.path.isfile(mix_path) or not os.path.isfile(voc_path):
        return 0.0
    mix, _ = librosa.load(mix_path, sr=sample_rate, mono=True)
    vocals, _ = librosa.load(voc_path, sr=sample_rate, mono=True)
    n = min(len(mix), len(vocals))
    residual = mix[:n].astype(np.float64) - vocals[:n].astype(np.float64)
    return estimate_dialogue_duck_db(residual, transcript, sample_rate)


def duck_db_for_original(folder, bed, sample_rate=24000):
    """Match original mix ducking. If the original does not duck BGM, return 0."""
    transcript_path = os.path.join(folder, 'translation.json')
    if not os.path.isfile(transcript_path):
        return 0.0
    try:
        with open(transcript_path, 'r', encoding='utf-8') as handle:
            transcript = json.load(handle)
    except Exception:
        return 0.0
    orig_db = original_mix_duck_db(folder, transcript, sample_rate)
    stem_db = estimate_dialogue_duck_db(bed, transcript, sample_rate)
    extra = orig_db - max(stem_db, 0.0)
    if orig_db < 3.0 or extra < 2.0:
        logger.info(
            f'原作對白未明顯壓低 BGM（約 {orig_db:.1f} dB），不另外閃避'
        )
        return 0.0
    duck_db = float(min(8.0, extra))
    logger.info(
        f'原作對白有壓 BGM（約 {orig_db:.1f} dB），配音再閃避 {duck_db:.1f} dB'
    )
    return duck_db


def duck_bed_under_speech(speech, bed, sample_rate=24000, duck_db=8.0, attack_ms=40, release_ms=280):
    """Lower accompaniment while dialogue is present, restore in gaps."""
    if duck_db < 0.5:
        return np.asarray(bed, dtype=np.float32).reshape(-1)
    speech = np.asarray(speech, dtype=np.float64).reshape(-1)
    bed = np.asarray(bed, dtype=np.float64).reshape(-1)
    n = max(len(speech), len(bed))
    if len(speech) < n:
        speech = np.pad(speech, (0, n - len(speech)))
    if len(bed) < n:
        bed = np.pad(bed, (0, n - len(bed)))
    if n == 0:
        return bed.astype(np.float32)
    from scipy.ndimage import maximum_filter1d, uniform_filter1d
    win = max(1, int(0.025 * sample_rate))
    rms = np.sqrt(np.maximum(uniform_filter1d(np.square(speech), size=win), 0.0))
    hold = maximum_filter1d(rms, size=max(1, int(sample_rate * release_ms / 1000.0)))
    env = uniform_filter1d(hold, size=max(1, int(sample_rate * attack_ms / 1000.0)))
    speech_on = np.clip((env - 0.02) / 0.07, 0.0, 1.0)
    min_gain = float(10.0 ** (-abs(duck_db) / 20.0))
    gain = 1.0 - (1.0 - min_gain) * speech_on
    return (bed * gain).astype(np.float32)


def suppress_leak_in_bed(bed, vocals, sample_rate=24000, reduce_db=12.0):
    """Dip accompaniment when original vocals are loud so leaked speech is quieter."""
    if bed is None or vocals is None or len(bed) == 0 or len(vocals) == 0:
        return bed
    n = min(len(bed), len(vocals))
    from scipy.ndimage import maximum_filter1d, uniform_filter1d
    vocal = np.asarray(vocals[:n], dtype=np.float32)
    win = max(1, int(0.02 * sample_rate))
    rms = np.sqrt(np.maximum(uniform_filter1d(np.square(vocal), size=win), 0.0))
    hold = maximum_filter1d(rms, size=max(1, int(sample_rate * 0.08)))
    env = uniform_filter1d(hold, size=max(1, int(sample_rate * 0.012)))
    speech_on = np.clip((env - 0.015) / 0.05, 0.0, 1.0)
    min_gain = float(10.0 ** (-abs(reduce_db) / 20.0))
    gain = 1.0 - (1.0 - min_gain) * speech_on
    out = np.asarray(bed, dtype=np.float32).copy()
    out[:n] = (out[:n] * gain).astype(np.float32)
    return out


def write_combined_mix(folder, speech, bed, target_language='中文'):
    from tools.target_language import AUDIO_LAYOUT_VERSION, AUDIO_MIX_VERSION, save_dub_meta
    ceiling = 0.99
    orig_path = os.path.join(folder, 'audio.wav')
    if os.path.isfile(orig_path):
        orig, _ = librosa.load(orig_path, sr=24000, mono=True)
        if orig is not None and len(orig):
            ceiling = float(min(0.99, max(0.89, np.max(np.abs(orig)))))
    vocals = _load_vocals(folder)
    if vocals is not None:
        bed = suppress_leak_in_bed(bed, vocals)
    duck_db = duck_db_for_original(folder, bed)
    if duck_db:
        bed = duck_bed_under_speech(speech, bed, duck_db=duck_db)
    combined = mix_speech_and_bed(speech, bed, ceiling=ceiling)
    save_wav(combined, os.path.join(folder, 'audio_combined.wav'))
    save_dub_meta(
        folder,
        tts=target_language,
        mix_version=AUDIO_MIX_VERSION,
        layout_version=AUDIO_LAYOUT_VERSION,
    )
    return os.path.join(folder, 'audio_combined.wav')


def remix_combined_audio(folder, target_language='中文'):
    tts_path = os.path.join(folder, 'audio_tts.wav')
    inst_path = os.path.join(folder, 'audio_instruments.wav')
    if not os.path.isfile(tts_path) or not os.path.isfile(inst_path):
        return False
    speech, _ = librosa.load(tts_path, sr=24000, mono=True)
    bed, _ = librosa.load(inst_path, sr=24000, mono=True)
    write_combined_mix(folder, speech, bed, target_language)
    video_path = os.path.join(folder, 'video.mp4')
    if os.path.isfile(video_path):
        os.remove(video_path)
    logger.info(f'已重混伴奏: {os.path.join(folder, "audio_combined.wav")}')
    return True


def _orig_start(line):
    if line.get('orig_start') is not None:
        return float(line['orig_start'])
    return float(line.get('start') or 0)


def stamp_orig_times(transcript):
    for line in transcript or []:
        if line.get('orig_start') is None:
            line['orig_start'] = float(line.get('start') or 0)
            line['orig_end'] = float(line.get('end') or 0)
    return transcript


def fit_tts_to_gap(wav_path, max_length, sample_rate=24000):
    """Keep natural TTS length. Compress only if it would overlap the next line. Never pad."""
    try:
        wav, sample_rate = librosa.load(wav_path, sr=sample_rate, mono=True)
    except Exception:
        alt = wav_path.replace('.wav', '.mp3') if wav_path.endswith('.wav') else wav_path + '.mp3'
        wav, sample_rate = librosa.load(alt, sr=sample_rate, mono=True)
    current = len(wav) / sample_rate if sample_rate else 0.0
    if current <= 0:
        return np.zeros((int(0.05 * sample_rate),), dtype=np.float32), 0.05
    if max_length is None or max_length <= 0.05 or current <= max_length + 0.03:
        return np.asarray(wav, dtype=np.float32), current
    cap = int(max(1, max_length * sample_rate))
    wav = np.asarray(wav[:cap], dtype=np.float32)
    return wav, len(wav) / sample_rate


def assemble_dub_timeline(transcript, wav_paths, sample_rate=24000, vocals=None):
    from tools.vocal_particles import available_tts_seconds, is_inner_overlapping_particle

    stamp_orig_times(transcript)
    starts = [_orig_start(line) for line in transcript]
    full = np.zeros((0,), dtype=np.float32)
    vocals = None if vocals is None else np.asarray(vocals, dtype=np.float32).reshape(-1)
    for i, (line, wav_path) in enumerate(zip(transcript, wav_paths)):
        start = starts[i]
        if is_inner_overlapping_particle(transcript, i):
            logger.warning(
                f'跳過重疊語氣詞以免壓扁原句：{(line.get("text") or "")[:8]}@{start:.2f}s'
            )
            line['start'] = start
            line['end'] = start
            continue
        last_end = len(full) / sample_rate
        if start < last_end - 0.02:
            start = last_end
        max_len = available_tts_seconds(transcript, i, start)
        if start > last_end:
            pad = int(round((start - last_end) * sample_rate))
            if pad > 0:
                full = np.concatenate((full, np.zeros((pad,), dtype=np.float32)))
        wav, length = fit_tts_to_gap(wav_path, max_len, sample_rate)
        orig_start = float(line.get('orig_start') or start)
        orig_end = float(line.get('orig_end') or (orig_start + length))
        orig_slice = None
        if vocals is not None and vocals.size:
            orig_slice = vocals[int(orig_start * sample_rate):int(orig_end * sample_rate)]
        wav = match_line_loudness(wav, orig_slice)
        line['start'] = len(full) / sample_rate
        full = np.concatenate((full, wav))
        line['end'] = line['start'] + length
    return limit_peak(full)


def _load_vocals(folder, sample_rate=24000):
    vocal_path = os.path.join(folder, 'audio_vocals.wav')
    if not os.path.isfile(vocal_path):
        return None
    vocals, _ = librosa.load(vocal_path, sr=sample_rate, mono=True)
    return vocals


def restitch_tts_timeline(folder, target_language='中文'):
    transcript_path = os.path.join(folder, 'translation.json')
    wav_dir = os.path.join(folder, 'wavs')
    if not os.path.isfile(transcript_path) or not os.path.isdir(wav_dir):
        return False
    with open(transcript_path, 'r', encoding='utf-8') as handle:
        transcript = json.load(handle)
    from tools.vocal_particles import repair_overlapping_particle_cards
    repaired = repair_overlapping_particle_cards(transcript)
    if len(repaired) != len(transcript):
        return False
    transcript = repaired
    wav_paths = []
    for i, _line in enumerate(transcript):
        path = os.path.join(wav_dir, f'{str(i).zfill(4)}.wav')
        if not os.path.isfile(path):
            alt = path.replace('.wav', '.mp3')
            if os.path.isfile(alt):
                path = alt
            else:
                return False
        wav_paths.append(path)
    full_wav = assemble_dub_timeline(transcript, wav_paths, vocals=_load_vocals(folder))
    save_wav(full_wav, os.path.join(folder, 'audio_tts.wav'))
    with open(transcript_path, 'w', encoding='utf-8') as handle:
        json.dump(transcript, handle, indent=2, ensure_ascii=False)
    inst_path = os.path.join(folder, 'audio_instruments.wav')
    if not os.path.isfile(inst_path):
        return False
    instruments_wav, _ = librosa.load(inst_path, sr=24000, mono=True)
    write_combined_mix(folder, full_wav, instruments_wav, target_language)
    video_path = os.path.join(folder, 'video.mp4')
    if os.path.isfile(video_path):
        os.remove(video_path)
    logger.info(f'已重排配音時長: {os.path.join(folder, "audio_combined.wav")}')
    return True

