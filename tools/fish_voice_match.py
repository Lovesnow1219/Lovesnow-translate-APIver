# -*- coding: utf-8 -*-
"""Match each speaker to the closest Fish official stock voice and lock #1."""
import json
import os
import time

import numpy as np
import requests
from dotenv import load_dotenv
from loguru import logger

from tools.fish_official_voices import FISH_OFFICIAL_EN_VOICES

load_dotenv()

MATCH_QUALITY = 2
_PROBE_TEXT = 'Hello, this is a short voice sample for matching.'
_CACHE_DIR = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'temp', 'fish_voice_bank')
_PITCH_MALE = 175.0
_PITCH_FEMALE = 190.0
_EMBED_SR = 16000
_EMBED_KIND = 'ge2e'
_OFFICIAL_SEED_ID = '9a9cf47702da476aa4629e2506d4a857'
_OFFICIAL_CACHE = 'official_en.json'
_OFFICIAL_TTL_SEC = 7 * 24 * 3600
_ge2e_encoder = None
_ge2e_failed = False

# Baked Fish Official English extras. Live catalog refresh may add newer official voices.
FISH_EXTRA_MATCH_VOICES = FISH_OFFICIAL_EN_VOICES

def match_enabled():
    raw = (os.getenv('FISH_VOICE_MATCH') or '1').strip().lower()
    return raw not in ('0', 'false', 'no', 'off')


def speaker_needs_match(speaker):
    from tools.line_roles import SPEAKER_BGM, SPEAKER_NARR, SPEAKER_SYS
    name = str(speaker or '')
    if name in {SPEAKER_BGM, SPEAKER_SYS, SPEAKER_NARR, 'SYSTEM', 'NARRATOR'} or name.endswith('_SYS'):
        return False
    return True


def gender_from_pitch(pitch):
    if pitch is None:
        return None
    if pitch >= _PITCH_FEMALE:
        return 'female'
    if pitch <= _PITCH_MALE:
        return 'male'
    return None


def cosine(left, right):
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    if a.size == 0 or b.size == 0 or a.size != b.size:
        return 0.0
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def pick_closest(speaker_vec, voices, prints, used_ids=None, gender=None, reserved_ids=None):
    """Return the closest unused voice dict, or None."""
    used = {item for item in (used_ids or set()) if item}
    reserved = {item for item in (reserved_ids or set()) if item}
    ranked = []
    for voice in voices or ():
        vid = voice.get('id')
        if not vid or vid in used or vid in reserved:
            continue
        if gender and voice.get('gender') and voice['gender'] != gender:
            continue
        vec = prints.get(vid)
        if vec is None:
            continue
        ranked.append((cosine(speaker_vec, vec), voice))
    ranked.sort(key=lambda item: (-item[0], item[1].get('name') or ''))
    if not ranked and gender:
        return pick_closest(speaker_vec, voices, prints, used_ids=used, reserved_ids=reserved)
    if not ranked:
        return None
    score, voice = ranked[0]
    return {
        'id': voice['id'],
        'name': voice.get('name') or voice['id'],
        'role': voice.get('role') or 'male',
        'gender': voice.get('gender'),
        'score': float(score),
        'top': [
            {'name': item.get('name') or item.get('id'), 'id': item['id'], 'score': round(sim, 3)}
            for sim, item in ranked[:3]
        ],
    }


def extra_match_voices():
    """Official English extras: refreshed catalog when possible, else the baked list."""
    refreshed = _refresh_official_en_voices()
    return refreshed or FISH_OFFICIAL_EN_VOICES


def all_match_voices(bank):
    seen = set()
    out = []
    for item in tuple(bank or ()) + tuple(extra_match_voices()):
        vid = item.get('id')
        if not vid or vid in seen or item.get('role') == 'system':
            continue
        if vid == 'e3cd384158934cc9a01029cd7d278634':
            continue
        seen.add(vid)
        out.append(item)
    return out


def _official_cache_path():
    return os.path.join(_CACHE_DIR, _OFFICIAL_CACHE)


def _read_official_cache(max_age=None):
    path = _official_cache_path()
    if not os.path.isfile(path):
        return ()
    try:
        if max_age is not None and (time.time() - os.path.getmtime(path)) > max_age:
            return ()
        with open(path, 'r', encoding='utf-8') as handle:
            data = json.load(handle)
        items = data.get('items') if isinstance(data, dict) else data
        if not isinstance(items, list):
            return ()
        return tuple(item for item in items if isinstance(item, dict) and item.get('id'))
    except Exception:
        return ()


def _write_official_cache(items):
    os.makedirs(_CACHE_DIR, exist_ok=True)
    with open(_official_cache_path(), 'w', encoding='utf-8') as handle:
        json.dump({'items': list(items)}, handle, indent=2, ensure_ascii=False)


def _voice_from_official_model(item):
    tags = [str(tag).lower() for tag in (item.get('tags') or [])]
    title = str(item.get('title') or '').replace('\u2014', '-').replace('\u2013', '-')
    name = title.split(' - ')[0].split('(')[0].strip() or (item.get('_id') or '')
    gender = 'female' if 'female' in tags else ('male' if 'male' in tags else '')
    if 'old' in tags or 'elder' in tags:
        age = 'elder'
    elif 'young' in tags:
        age = 'young'
    else:
        age = 'adult'
    if gender == 'female':
        role = 'young_female' if age == 'young' else (
            'mature_female' if ('narration' in tags or 'asmr' in tags) else 'female'
        )
    elif age == 'elder':
        role = 'elder_male'
    elif age == 'young':
        role = 'young_male'
    elif 'narration' in tags or 'storytelling' in tags:
        role = 'narrator'
    else:
        role = 'male'
    return {
        'id': item.get('_id') or item.get('id'),
        'name': name,
        'gender': gender or 'male',
        'age': age,
        'role': role,
    }


def _refresh_official_en_voices():
    cached = _read_official_cache(_OFFICIAL_TTL_SEC)
    if cached:
        return cached
    headers = _auth_headers()
    if not headers:
        return _read_official_cache()
    try:
        seed = requests.get(
            f'https://api.fish.audio/model/{_OFFICIAL_SEED_ID}',
            headers=headers,
            timeout=30,
        )
        if seed.status_code != 200:
            return _read_official_cache()
        author_id = ((seed.json() or {}).get('author') or {}).get('_id')
        if not author_id:
            return _read_official_cache()
        items = []
        page = 1
        while page <= 8:
            data = requests.get(
                'https://api.fish.audio/model',
                headers=headers,
                params={
                    'author_id': author_id,
                    'language': 'en',
                    'page_size': 100,
                    'page_number': page,
                    'sort_by': 'task_count',
                },
                timeout=45,
            ).json() or {}
            batch = data.get('items') or []
            for raw in batch:
                if raw.get('type') not in (None, '', 'tts'):
                    continue
                if raw.get('state') and raw.get('state') != 'trained':
                    continue
                voice = _voice_from_official_model(raw)
                if voice.get('id'):
                    items.append(voice)
            if not data.get('has_more') or not batch:
                break
            page += 1
        if len(items) < 20:
            return _read_official_cache() or FISH_OFFICIAL_EN_VOICES
        _write_official_cache(items)
        logger.info(f'已更新官方英文聲庫 {len(items)} 條')
        return tuple(items)
    except Exception as exc:
        logger.warning(f'更新官方聲庫失敗，改用內建清單：{exc}')
        return _read_official_cache() or FISH_OFFICIAL_EN_VOICES


def _load_mono(path, sr=_EMBED_SR):
    if not path or not os.path.isfile(path):
        return None, None
    try:
        import librosa
        y, loaded = librosa.load(path, sr=sr, mono=True)
    except Exception as exc:
        logger.warning(f'讀取聲紋音檔失敗 {path}: {exc}')
        return None, None
    if y is None or len(y) < int(0.4 * loaded):
        return None, None
    return y, loaded


def _mfcc_vector(y, sr):
    import librosa
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    delta = librosa.feature.delta(mfcc)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    f0 = librosa.yin(y, fmin=70, fmax=400, sr=sr)
    f0 = np.asarray(f0)
    f0 = f0[np.isfinite(f0) & (f0 > 0)]
    pitch = float(np.median(f0)) if len(f0) >= 4 else 0.0
    pitch_std = float(np.std(f0)) if len(f0) >= 4 else 0.0
    parts = [
        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1),
        np.mean(delta, axis=1),
        np.std(delta, axis=1),
        np.array([
            float(np.mean(centroid)) / 2000.0,
            float(np.std(centroid)) / 1000.0,
            float(np.mean(zcr)),
            pitch / 300.0,
            pitch_std / 50.0,
        ], dtype=np.float64),
    ]
    vec = np.concatenate([np.asarray(p, dtype=np.float64).reshape(-1) for p in parts])
    norm = np.linalg.norm(vec)
    if norm > 1e-8:
        vec = vec / norm
    return vec


def _stub_webrtcvad():
    """Resemblyzer imports webrtcvad only to trim silence; we skip that on Windows."""
    import sys
    import types
    if 'webrtcvad' in sys.modules:
        return
    module = types.ModuleType('webrtcvad')

    class Vad:
        def __init__(self, mode=3):
            pass

        def is_speech(self, buf, sample_rate=16000):
            return True

    module.Vad = Vad
    sys.modules['webrtcvad'] = module


def _ge2e_encoder_model():
    global _ge2e_encoder, _ge2e_failed
    if _ge2e_failed:
        return None
    if _ge2e_encoder is not None:
        return _ge2e_encoder
    try:
        _stub_webrtcvad()
        from resemblyzer.voice_encoder import VoiceEncoder
        _ge2e_encoder = VoiceEncoder(device='cpu', verbose=False)
        return _ge2e_encoder
    except Exception as exc:
        _ge2e_failed = True
        logger.warning(f'GE2E 聲紋模型不可用，改用 MFCC：{exc}')
        return None


def _ge2e_vector(y, sr):
    encoder = _ge2e_encoder_model()
    if encoder is None:
        return None
    import librosa
    from resemblyzer.hparams import audio_norm_target_dBFS, sampling_rate
    wav = np.asarray(y, dtype=np.float32)
    if sr != sampling_rate:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=sampling_rate)
    peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    if peak > 1.0:
        wav = wav / peak
    rms = float(np.sqrt(np.mean(np.square(wav)))) if wav.size else 0.0
    if rms > 1e-8:
        wave_db = 20.0 * np.log10(rms)
        change = audio_norm_target_dBFS - wave_db
        if change > 0:
            wav = wav * (10.0 ** (change / 20.0))
    embed = encoder.embed_utterance(wav)
    return np.asarray(embed, dtype=np.float64).reshape(-1)


def _vector_from_audio(y, sr):
    vec = _ge2e_vector(y, sr)
    if vec is not None:
        return vec
    return _mfcc_vector(y, sr)


def audio_fingerprint(path):
    y, sr = _load_mono(path)
    if y is None:
        return None
    try:
        return _vector_from_audio(y, sr)
    except Exception as exc:
        logger.warning(f'計算聲紋失敗 {path}: {exc}')
        return None


def speaker_print(folder, speaker):
    path = os.path.join(folder, 'SPEAKER', f'{speaker}.wav')
    return audio_fingerprint(path)


def _meta_path():
    return os.path.join(_CACHE_DIR, 'meta.json')


def _load_meta():
    path = _meta_path()
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, 'r', encoding='utf-8') as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_meta(meta):
    os.makedirs(_CACHE_DIR, exist_ok=True)
    with open(_meta_path(), 'w', encoding='utf-8') as handle:
        json.dump(meta, handle, indent=2, ensure_ascii=False)


def _auth_headers():
    from tools.api_keys import collect_api_keys
    keys = collect_api_keys('FISH_API_KEY')
    if not keys:
        return {}
    return {'Authorization': f'Bearer {keys[0]}'}


def _download_sample(voice_id):
    wav_path = os.path.join(_CACHE_DIR, f'{voice_id}.wav')
    if os.path.isfile(wav_path) and os.path.getsize(wav_path) > 2048:
        return wav_path
    headers = _auth_headers()
    try:
        response = requests.get(
            f'https://api.fish.audio/model/{voice_id}',
            headers=headers,
            timeout=30,
        )
        if response.status_code != 200:
            return None
        samples = (response.json() or {}).get('samples') or []
        url = str((samples[0] or {}).get('audio') or '').strip() if samples else ''
        if not url:
            return None
        audio = requests.get(url, headers=headers, timeout=60)
        if audio.status_code != 200 or not audio.content:
            audio = requests.get(url, timeout=60)
        if audio.status_code != 200 or len(audio.content or b'') < 2048:
            return None
        os.makedirs(_CACHE_DIR, exist_ok=True)
        raw_path = wav_path
        if '.mp3' in url.lower() or audio.headers.get('Content-Type', '').find('mpeg') >= 0:
            raw_path = os.path.join(_CACHE_DIR, f'{voice_id}.mp3')
        with open(raw_path, 'wb') as handle:
            handle.write(audio.content)
        y, sr = _load_mono(raw_path)
        if y is None:
            return None
        from tools.utils import save_wav_norm
        save_wav_norm(np.asarray(y, dtype=np.float32), wav_path, sample_rate=int(sr))
        if raw_path != wav_path:
            try:
                os.remove(raw_path)
            except OSError:
                pass
        return wav_path
    except Exception as exc:
        logger.warning(f'下載聲線樣本失敗 {voice_id}: {exc}')
        return None


def _synthesize_probe(voice_id, tts_fn):
    wav_path = os.path.join(_CACHE_DIR, f'{voice_id}.wav')
    if os.path.isfile(wav_path) and os.path.getsize(wav_path) > 2048:
        return wav_path
    if not tts_fn:
        return None
    os.makedirs(_CACHE_DIR, exist_ok=True)
    try:
        tts_fn(_PROBE_TEXT, wav_path, target_language='English', voice=voice_id, speed=1.0)
    except Exception as exc:
        logger.warning(f'合成聲線樣本失敗 {voice_id}: {exc}')
        return None
    return wav_path if os.path.isfile(wav_path) else None


def ensure_stock_prints(voices, tts_fn=None):
    """Fingerprints for stock voices. Prefer official samples, else a short TTS probe."""
    os.makedirs(_CACHE_DIR, exist_ok=True)
    meta = _load_meta()
    prints = {}
    pending = []
    for voice in voices or ():
        vid = voice.get('id')
        if not vid:
            continue
        npy_path = os.path.join(_CACHE_DIR, f'{vid}.npy')
        if os.path.isfile(npy_path) and meta.get('kind') == _EMBED_KIND:
            try:
                prints[vid] = np.load(npy_path)
                continue
            except Exception:
                pass
        pending.append(voice)

    def _build(voice):
        vid = voice.get('id')
        npy_path = os.path.join(_CACHE_DIR, f'{vid}.npy')
        wav_path = _download_sample(vid) or _synthesize_probe(vid, tts_fn)
        vec = audio_fingerprint(wav_path) if wav_path else None
        if vec is None:
            logger.warning(f'聲線 {voice.get("name") or vid} 無法建立聲紋，略過')
            return vid, None
        np.save(npy_path, vec)
        return vid, vec

    rebuilt = False
    if pending:
        from tools.parallel import env_workers, map_parallel
        workers = env_workers('FISH_VOICE_BANK_WORKERS', 6, maximum=8)
        logger.info(f'補建官方聲紋 {len(pending)} 條')
        for vid, vec in map_parallel(_build, pending, workers):
            if vec is None:
                continue
            prints[vid] = vec
            rebuilt = True
    if rebuilt:
        meta['kind'] = _EMBED_KIND if (_ge2e_encoder is not None and not _ge2e_failed) else 'mfcc'
        meta['ids'] = sorted(prints)
        _save_meta(meta)
    logger.info(f'聲線庫已就緒（{len(prints)} 條，{meta.get("kind") or "mfcc"}）')
    return prints
