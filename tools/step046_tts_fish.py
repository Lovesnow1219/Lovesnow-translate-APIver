# -*- coding: utf-8 -*-
import hashlib
import json
import os
import re
import tempfile
import time

import numpy as np
import requests
from dotenv import load_dotenv
from loguru import logger

from tools.fish_emotion import apply_emotion_tags, spoken_text_for_timing

load_dotenv()

FISH_TTS_FREE = 's2.1-pro-free'
FISH_TTS_PAID = 's2.1-pro'
_use_paid_model = False

# Fish Official default library. Used when cloning is off or a speaker sample is too short.
# IDs verified on fish.audio/m/<id> and /discovery/ (2026-08-27).
# https://fish.audio/m/<id>
FISH_VOICE_BANK = (
    {'role': 'male', 'id': '79d0bd3e4e5444b18f7b6d89b5927bf1', 'name': 'Jordan', 'gender': 'male', 'age': 'elder'},
    {'role': 'female', 'id': '9a9cf47702da476aa4629e2506d4a857', 'name': 'Hannah', 'gender': 'female', 'age': 'adult'},
    {'role': 'young_female', 'id': '933563129e564b19a115bedd57b7406a', 'name': 'Sarah', 'gender': 'female', 'age': 'young'},
    {'role': 'young_male', 'id': '536d3a5e000945adb7038665781a4aca', 'name': 'Ethan', 'gender': 'male', 'age': 'adult'},
    {'role': 'system', 'id': 'e3cd384158934cc9a01029cd7d278634', 'name': 'Laura', 'gender': 'female', 'age': 'adult'},
    {'role': 'narrator', 'id': 'bf322df2096a46f18c579d0baa36f41d', 'name': 'Adrian', 'gender': 'male', 'age': 'adult'},
    {'role': 'mature_female', 'id': 'b347db033a6549378b48d00acb0d06cd', 'name': 'Selene', 'gender': 'female', 'age': 'adult'},
)
FISH_STOCK_VOICES = {item['role']: item['id'] for item in FISH_VOICE_BANK}
FISH_VOICE_NAMES = {item['id']: item['name'] for item in FISH_VOICE_BANK}
FISH_IDS_MALE = tuple(item['id'] for item in FISH_VOICE_BANK if item['gender'] == 'male')
FISH_IDS_FEMALE = tuple(item['id'] for item in FISH_VOICE_BANK if item['gender'] == 'female')
_ROLE_ALIASES = {
    'girl': 'young_female',
    'boy': 'young_male',
    'elder_male': 'male',
    'narrator_f': 'mature_female',
}
ROLE_GENDER = {
    'male': 'male', 'young_male': 'male', 'boy': 'male', 'elder_male': 'male', 'narrator': 'male',
    'female': 'female', 'young_female': 'female', 'mature_female': 'female', 'girl': 'female',
    'system': 'female', 'narrator_f': 'female',
}
ROLE_FALLBACKS = {
    'male': ('male', 'young_male', 'elder_male', 'boy', 'narrator'),
    'young_male': ('young_male', 'boy', 'male', 'elder_male'),
    'boy': ('boy', 'young_male', 'male'),
    'elder_male': ('elder_male', 'male', 'narrator', 'young_male'),
    'female': ('female', 'young_female', 'mature_female', 'girl'),
    'young_female': ('young_female', 'girl', 'female', 'mature_female'),
    'girl': ('girl', 'young_female', 'female'),
    'mature_female': ('mature_female', 'female', 'young_female', 'system'),
    'system': ('system', 'mature_female', 'female'),
    'narrator': ('narrator', 'elder_male', 'male'),
    'narrator_f': ('narrator_f', 'mature_female', 'female'),
}
OPENAI_STOCK_VOICES = {
    'male': 'onyx',
    'young_male': 'ash',
    'boy': 'fable',
    'elder_male': 'onyx',
    'narrator': 'sage',
    'female': 'nova',
    'young_female': 'shimmer',
    'girl': 'nova',
    'mature_female': 'coral',
    'system': 'echo',
    'narrator_f': 'coral',
}
OPENAI_IDS_MALE = ('onyx', 'ash', 'fable', 'sage', 'echo')
OPENAI_IDS_FEMALE = ('nova', 'shimmer', 'coral', 'alloy')
EDGE_STOCK_EN = {
    'male': 'en-US-GuyNeural',
    'young_male': 'en-US-AndrewNeural',
    'boy': 'en-US-AndrewNeural',
    'elder_male': 'en-US-DavisNeural',
    'narrator': 'en-US-EricNeural',
    'female': 'en-US-JennyNeural',
    'young_female': 'en-US-AriaNeural',
    'girl': 'en-US-AnaNeural',
    'mature_female': 'en-US-JennyNeural',
    'system': 'en-US-AriaNeural',
    'narrator_f': 'en-US-JennyNeural',
}
EDGE_IDS_EN_MALE = ('en-US-GuyNeural', 'en-US-AndrewNeural', 'en-US-DavisNeural', 'en-US-EricNeural', 'en-US-ChristopherNeural')
EDGE_IDS_EN_FEMALE = ('en-US-JennyNeural', 'en-US-AriaNeural', 'en-US-AnaNeural')
EDGE_STOCK_ZH = {
    'male': 'zh-CN-YunxiNeural',
    'young_male': 'zh-CN-YunjianNeural',
    'boy': 'zh-CN-YunxiaNeural',
    'elder_male': 'zh-CN-YunyangNeural',
    'narrator': 'zh-CN-YunjianNeural',
    'female': 'zh-CN-XiaoxiaoNeural',
    'young_female': 'zh-CN-XiaoyiNeural',
    'girl': 'zh-CN-XiaoxiaoNeural',
    'mature_female': 'zh-CN-XiaoyiNeural',
    'system': 'zh-CN-YunyangNeural',
    'narrator_f': 'zh-CN-XiaoxiaoNeural',
}
EDGE_IDS_ZH_MALE = ('zh-CN-YunxiNeural', 'zh-CN-YunjianNeural', 'zh-CN-YunyangNeural', 'zh-CN-YunxiaNeural')
EDGE_IDS_ZH_FEMALE = ('zh-CN-XiaoxiaoNeural', 'zh-CN-XiaoyiNeural')


def _simple_edge(male, female):
    stock = {
        'male': male,
        'young_male': male,
        'boy': male,
        'elder_male': male,
        'narrator': male,
        'female': female,
        'young_female': female,
        'girl': female,
        'mature_female': female,
        'system': female,
        'narrator_f': female,
    }
    return stock, (male,), (female,)


EDGE_LANG_VOICES = {
    'English': (EDGE_STOCK_EN, EDGE_IDS_EN_MALE, EDGE_IDS_EN_FEMALE),
    '中文': (EDGE_STOCK_ZH, EDGE_IDS_ZH_MALE, EDGE_IDS_ZH_FEMALE),
    '粤语': _simple_edge('zh-HK-WanLungNeural', 'zh-HK-HiuMaanNeural'),
    'Japanese': _simple_edge('ja-JP-KeitaNeural', 'ja-JP-NanamiNeural'),
    'Korean': _simple_edge('ko-KR-InJoonNeural', 'ko-KR-SunHiNeural'),
    'Spanish': _simple_edge('es-ES-AlvaroNeural', 'es-ES-ElviraNeural'),
    'French': _simple_edge('fr-FR-HenriNeural', 'fr-FR-DeniseNeural'),
    'Polish': _simple_edge('pl-PL-MarekNeural', 'pl-PL-ZofiaNeural'),
    'Vietnamese': _simple_edge('vi-VN-NamMinhNeural', 'vi-VN-HoaiMyNeural'),
    'Thai': _simple_edge('th-TH-NiwatNeural', 'th-TH-PremwadeeNeural'),
    'Indonesian': _simple_edge('id-ID-ArdiNeural', 'id-ID-GadisNeural'),
    'Malay': _simple_edge('ms-MY-OsmanNeural', 'ms-MY-YasminNeural'),
    'Filipino': _simple_edge('fil-PH-AngeloNeural', 'fil-PH-BlessicaNeural'),
}


def _edge_voices_for_language(target_language):
    key = str(target_language or '').strip()
    if key in EDGE_LANG_VOICES:
        return EDGE_LANG_VOICES[key]
    lower = key.lower()
    if lower in {'en', 'eng', 'english'}:
        return EDGE_LANG_VOICES['English']
    if '中文' in key or key in {'zh', 'zh-cn', 'zh-tw'}:
        return EDGE_LANG_VOICES['中文']
    if '粤' in key or lower in {'cantonese', 'yue'}:
        return EDGE_LANG_VOICES['粤语']
    return EDGE_LANG_VOICES['English']
_SYSTEM_MARKERS = ('特殊能力', '技能', '效果', '单位', '指數', '指数', '状态', '狀態', '施加', 'HP', 'MP', 'Lv')
_NARRATOR_MARKERS = ('轻轻', '心想', '只见', '说道', '了声', '他大概', '她心里', '旁白')
_YOUNG_FEMALE_MARKERS = ('淑女', '人家', '领主大人', '領主大人', '讨厌', '討厭')
_GIRL_MARKERS = ('呜呜', '嗚嗚', '好怕', '妈妈', '媽媽', '姐姐', '不要啦')
_BOY_MARKERS = ('爸爸', '哥哥', '我才不怕')
_MALE_MARKERS = ('掩护', '掩護', '老子', '兄弟', '实验一下', '實驗一下')
_ELDER_MARKERS = ('老夫', '吾', '罢了', '罷了')
_SKIP_UI_VOICES = {
    'alloy', 'ash', 'coral', 'echo', 'fable', 'nova', 'onyx', 'sage', 'shimmer',
}


def _next_unused(preferred, extras, used):
    if preferred and preferred not in used:
        return preferred
    for item in extras:
        if item and item not in used:
            return item
    return preferred


def _role_gender(role):
    return ROLE_GENDER.get(role) or 'male'


def _pick_available_role(guessed, used_roles):
    for name in ROLE_FALLBACKS.get(guessed, (guessed,)):
        if name not in used_roles:
            return name
    return guessed


def _voice_for_role(role, used, stock, male_ids, female_ids, fallback):
    preferred = stock.get(role) or stock.get(_ROLE_ALIASES.get(role, ''), fallback)
    extras = female_ids if _role_gender(role) == 'female' else male_ids
    return _next_unused(preferred, extras, used)


FISH_MODEL_URL = 'https://api.fish.audio/model'
CLONE_QUALITY = 2
_CLONE_SR = 24000
_CLONE_MIN_SEC = 3.0
_CLONE_TARGET_SEC = 20.0
_CLONE_MAX_SEC = 25.0
_PITCH_LOW = 165.0
_PITCH_HIGH = 220.0


def _clone_enabled():
    raw = (os.getenv('FISH_CLONE') or '1').strip().lower()
    return raw not in ('0', 'false', 'no', 'off')


def _clip_to_wav_bytes(y, sr):
    fd, path = tempfile.mkstemp(suffix='.wav')
    os.close(fd)
    try:
        from tools.utils import save_wav_norm
        save_wav_norm(np.asarray(y, dtype=np.float32), path, sample_rate=int(sr))
        with open(path, 'rb') as handle:
            return handle.read()
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def _load_vocals(folder, sr=_CLONE_SR):
    path = os.path.join(folder, 'audio_vocals.wav')
    if not path or not os.path.isfile(path):
        return None, None
    try:
        import librosa
        y, loaded_sr = librosa.load(path, sr=sr, mono=True)
        return y, loaded_sr
    except Exception as exc:
        logger.warning(f'讀取人聲失敗 {path}: {exc}')
        return None, None


def _clip_pitch(clip, sr):
    if clip is None or len(clip) < int(0.25 * sr):
        return None
    try:
        import librosa
        f0 = librosa.yin(clip, fmin=70, fmax=400, sr=sr)
        f0 = np.asarray(f0)
        f0 = f0[np.isfinite(f0) & (f0 > 0)]
        if len(f0) < 4:
            return None
        return float(np.median(f0))
    except Exception:
        return None


def _loudest_window(clip, sr, max_sec):
    max_len = int(max_sec * sr)
    if clip is None or len(clip) <= max_len:
        return clip
    win = max_len
    hop = max(1, sr // 4)
    best_i, best_e = 0, -1.0
    for i in range(0, len(clip) - win + 1, hop):
        energy = float(np.mean(clip[i:i + win] ** 2))
        if energy > best_e:
            best_e, best_i = energy, i
    return clip[best_i:best_i + win]


def _rms(clip):
    if clip is None or len(clip) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(clip))))


def _line_audio_clips(folder, transcript, speaker):
    y, sr = _load_vocals(folder)
    if y is None:
        return []
    speaker = str(speaker)
    others = [
        (float(item.get('start') or 0), float(item.get('end') or 0), str(item.get('speaker') or ''))
        for item in (transcript or [])
    ]
    items = []
    for line in transcript or []:
        if str(line.get('speaker') or '') != speaker:
            continue
        start = float(line.get('start') or 0)
        end = float(line.get('end') or 0)
        duration = end - start
        if duration < 0.7:
            continue
        overlap = 0.0
        for other_start, other_end, other_speaker in others:
            if other_speaker == speaker:
                continue
            overlap += max(0.0, min(end, other_end) - max(start, other_start))
        if duration > 0 and overlap > 0.3 * duration:
            continue
        i0 = int(max(0, start * sr))
        i1 = int(min(len(y), end * sr))
        clip = y[i0:i1]
        if _rms(clip) < 0.004:
            continue
        items.append({
            'clip': clip,
            'sr': sr,
            'duration': duration,
            'pitch': _clip_pitch(clip, sr),
        })
    return items


def _majority_pitch_clips(items):
    pitched = [item for item in items if item.get('pitch')]
    if not pitched:
        return items
    low = [item for item in pitched if item['pitch'] <= _PITCH_LOW]
    high = [item for item in pitched if item['pitch'] >= _PITCH_HIGH]
    low_dur = sum(item['duration'] for item in low)
    high_dur = sum(item['duration'] for item in high)
    if low_dur >= 3 and high_dur >= 3:
        keep = low if low_dur >= high_dur else high
        logger.info(
            f'克隆樣本去掉混雜音高（低 {low_dur:.1f}s / 高 {high_dur:.1f}s，留 {"低" if keep is low else "高"}）'
        )
        return keep
    return pitched or items


def prepare_clone_sample(wav_path):
    """Pick a loud 10–25s window from concatenated speaker audio. Returns (wav_bytes, seconds) or (None, 0)."""
    if not wav_path or not os.path.isfile(wav_path):
        return None, 0.0
    try:
        import librosa
        y, sr = librosa.load(wav_path, sr=_CLONE_SR, mono=True)
    except Exception as exc:
        logger.warning(f'讀取克隆樣本失敗 {wav_path}: {exc}')
        return None, 0.0
    if y is None or len(y) < int(_CLONE_MIN_SEC * sr):
        return None, 0.0
    clip = _loudest_window(y, sr, _CLONE_MAX_SEC)
    if _rms(clip) < 0.004:
        return None, 0.0
    try:
        return _clip_to_wav_bytes(clip, sr), len(clip) / float(sr)
    except Exception as exc:
        logger.warning(f'寫入克隆樣本失敗: {exc}')
        return None, 0.0


def prepare_clone_wavs(folder, speaker, transcript):
    """Return (list of wav bytes, total seconds) from pitch-consistent line clips."""
    items = _majority_pitch_clips(_line_audio_clips(folder, transcript, speaker))
    items.sort(key=lambda item: -item['duration'])
    blobs = []
    total = 0.0
    for item in items:
        if total >= _CLONE_TARGET_SEC and blobs:
            break
        clip = _loudest_window(item['clip'], item['sr'], 8.0)
        if clip is None or len(clip) < int(0.7 * item['sr']):
            continue
        try:
            blobs.append(_clip_to_wav_bytes(clip, item['sr']))
            total += len(clip) / float(item['sr'])
        except Exception:
            continue
        if len(blobs) >= 5:
            break
    if total < _CLONE_MIN_SEC:
        fallback, seconds = prepare_clone_sample(
            os.path.join(folder, 'SPEAKER', f'{speaker}.wav')
        )
        if fallback:
            return [fallback], seconds
        return [], 0.0
    return blobs, total


def split_mixed_pitch_speakers(folder, transcript):
    """If one diarized speaker contains both low and high pitch, split the minority out."""
    y, sr = _load_vocals(folder, sr=16000)
    if y is None or not transcript:
        return transcript, False
    changed = False
    speakers = []
    for line in transcript:
        speaker = str(line.get('speaker') or 'SPEAKER_00')
        if speaker not in speakers:
            speakers.append(speaker)
    for speaker in speakers:
        if speaker.endswith('_F') or speaker.endswith('_M'):
            continue
        low, high = [], []
        for line in transcript:
            if str(line.get('speaker') or '') != speaker:
                continue
            start = float(line.get('start') or 0)
            end = float(line.get('end') or 0)
            duration = max(0.0, end - start)
            i0 = int(max(0, start * sr))
            i1 = int(min(len(y), end * sr))
            pitch = _clip_pitch(y[i0:i1], sr)
            if pitch is None:
                continue
            row = (line, duration)
            if pitch <= _PITCH_LOW:
                low.append(row)
            elif pitch >= _PITCH_HIGH:
                high.append(row)
        low_dur = sum(duration for _, duration in low)
        high_dur = sum(duration for _, duration in high)
        if not (low_dur >= 6 and high_dur >= 6 and len(low) >= 2 and len(high) >= 2):
            continue
        if high_dur > low_dur:
            minority, tag = low, '_M'
        else:
            minority, tag = high, '_F'
        new_id = f'{speaker}{tag}'
        for line, _duration in minority:
            line['speaker'] = new_id
        changed = True
        logger.info(
            f'{speaker} 音高混雜（低 {low_dur:.1f}s / 高 {high_dur:.1f}s），拆出 {new_id}（{len(minority)} 句）'
        )
    return transcript, changed


def _clone_title(folder, speaker, wav_bytes):
    digest = hashlib.sha1(wav_bytes[:8192] + str(len(wav_bytes)).encode('utf-8')).hexdigest()[:8]
    base = re.sub(r'[^\w\-]+', '_', os.path.basename(os.path.abspath(folder)))[:24]
    return f'Linly_{base}_{speaker}_{digest}'[:80]


def _wait_clone_trained(model_id, api_key, timeout=45):
    deadline = time.time() + timeout
    state = ''
    while time.time() < deadline:
        try:
            response = requests.get(
                f'{FISH_MODEL_URL}/{model_id}',
                headers={'Authorization': f'Bearer {api_key}'},
                timeout=30,
            )
            if response.status_code == 200:
                state = str((response.json() or {}).get('state') or '').lower()
                if state == 'trained':
                    return True
                if state == 'failed':
                    return False
        except Exception:
            pass
        time.sleep(2)
    # Fast-mode clones are often usable before the poller sees trained.
    return state != 'failed'


def _create_fish_clone(title, wav_bytes):
    from tools.api_keys import next_api_key
    wav_list = wav_bytes if isinstance(wav_bytes, (list, tuple)) else [wav_bytes]
    wav_list = [blob for blob in wav_list if blob]
    if not wav_list:
        return None
    last_error = None
    delays = (0.0, 1.5, 3.0)
    for attempt, delay in enumerate(delays, start=1):
        if delay:
            time.sleep(delay)
        api_key, n_keys, key_index = next_api_key(
            'FISH_API_KEY', error='請先在 .env 設定 FISH_API_KEY'
        )
        try:
            response = requests.post(
                FISH_MODEL_URL,
                headers={'Authorization': f'Bearer {api_key}'},
                data={
                    'type': 'tts',
                    'title': title,
                    'description': 'Linly-Dubbing speaker clone',
                    'visibility': 'private',
                    'train_mode': 'fast',
                    'enhance_audio_quality': 'true',
                },
                files=[
                    ('voices', (f'speaker_{index}.wav', blob, 'audio/wav'))
                    for index, blob in enumerate(wav_list)
                ],
                timeout=120,
            )
            if response.status_code in (200, 201) and response.content:
                try:
                    data = response.json()
                except Exception:
                    data = {}
                model_id = data.get('_id') or data.get('id')
                state = str(data.get('state') or '').lower()
                if not model_id:
                    last_error = f'no id: {response.text[:200]}'
                    continue
                if state == 'failed':
                    last_error = f'state=failed: {response.text[:200]}'
                    continue
                if state in ('created', 'training'):
                    if not _wait_clone_trained(model_id, api_key):
                        last_error = f'clone not ready: {model_id}'
                        continue
                return str(model_id)
            last_error = f'HTTP {response.status_code}: {response.text[:200]}'
            if response.status_code in (401, 402, 403):
                from tools.api_keys import mark_api_key_dead
                mark_api_key_dead(api_key, f'http {response.status_code}')
                logger.warning(f'Fish 克隆被拒絕 ({response.status_code})，改試另一把 key')
                continue
            if response.status_code == 429:
                extra = '，改試另一把 key' if n_keys > 1 else ''
                logger.warning(
                    f'Fish 克隆限流 ({attempt}/{len(delays)} key {key_index}/{n_keys}){extra}'
                )
                continue
            logger.warning(f'Fish 克隆失敗 ({attempt}/{len(delays)}): {last_error}')
        except Exception as exc:
            last_error = str(exc)
            logger.warning(f'Fish 克隆失敗 ({attempt}/{len(delays)}): {exc}')
    logger.warning(f'Fish 克隆失敗: {last_error}')
    return None


def _median_pitch(wav_path):
    if not wav_path or not os.path.isfile(wav_path):
        return None
    try:
        import librosa
        y, sr = librosa.load(wav_path, sr=16000, mono=True)
        if y is None or len(y) < sr // 2:
            return None
        f0 = librosa.yin(y, fmin=70, fmax=400, sr=sr)
        f0 = np.asarray(f0)
        f0 = f0[np.isfinite(f0) & (f0 > 0)]
        if len(f0) < 8:
            return None
        return float(np.median(f0))
    except Exception:
        return None


def _guess_role(text, pitch=None):
    blob = text or ''
    scores = {
        'system': sum(blob.count(m) for m in _SYSTEM_MARKERS),
        'narrator': sum(blob.count(m) for m in _NARRATOR_MARKERS),
        'young_female': sum(blob.count(m) for m in _YOUNG_FEMALE_MARKERS),
        'girl': sum(blob.count(m) for m in _GIRL_MARKERS),
        'boy': sum(blob.count(m) for m in _BOY_MARKERS),
        'male': sum(blob.count(m) for m in _MALE_MARKERS),
        'elder_male': sum(blob.count(m) for m in _ELDER_MARKERS),
    }
    if scores['system'] or scores['narrator']:
        if pitch is None or _PITCH_LOW < pitch < _PITCH_HIGH:
            role = 'system' if scores['system'] >= scores['narrator'] else 'narrator'
            return role, scores[role]
    female_kw = scores['young_female'] + scores['girl']
    male_kw = scores['male'] + scores['boy'] + scores['elder_male']
    if pitch is not None:
        if pitch >= 250:
            return ('boy' if male_kw > female_kw else 'girl'), 1
        if pitch >= 220:
            return 'young_female', 1
        if pitch >= 200:
            return 'female', 1
        if pitch <= 115:
            return 'elder_male', 1
        if pitch <= 155:
            return 'male', 1
    if scores['elder_male'] and scores['elder_male'] >= max(female_kw, scores['male'], scores['boy']):
        return 'elder_male', scores['elder_male']
    if scores['girl'] and scores['girl'] >= max(scores['young_female'], male_kw):
        return 'girl', scores['girl']
    if scores['boy'] and scores['boy'] > female_kw:
        return 'boy', scores['boy']
    if scores['young_female'] > male_kw:
        return 'young_female', scores['young_female']
    if scores['male'] > female_kw:
        return 'male', scores['male']
    if female_kw > male_kw:
        return 'female', female_kw
    return 'male', 0


def _clone_quality(info):
    try:
        return int((info or {}).get('clone_quality') or 0)
    except (TypeError, ValueError):
        return 0


def fish_clone_pending(folder, transcript):
    """True when Fish TTS should clone before reusing audio_combined.wav."""
    if not _clone_enabled():
        return False
    assignment = {}
    path = os.path.join(folder, 'speaker_voices.json')
    if os.path.isfile(path):
        try:
            with open(path, 'r', encoding='utf-8') as handle:
                raw = json.load(handle)
            if isinstance(raw, dict):
                assignment = {str(k): v for k, v in raw.items() if isinstance(v, dict)}
        except Exception:
            assignment = {}
    speakers = {str(line.get('speaker') or 'SPEAKER_00') for line in (transcript or [])}
    if not speakers:
        return False
    for speaker in speakers:
        info = assignment.get(speaker) or {}
        if _clone_quality(info) >= CLONE_QUALITY:
            continue
        return True
    return False


def assign_speaker_voices(folder, transcript, target_language='English', clone=True):
    """Assign Fish voices: clone SPEAKER/*.wav when possible, else stock."""
    path = os.path.join(folder, 'speaker_voices.json')
    assignment = {}
    if os.path.isfile(path):
        try:
            with open(path, 'r', encoding='utf-8') as handle:
                raw = json.load(handle)
            if isinstance(raw, dict):
                assignment = {str(k): dict(v) for k, v in raw.items() if isinstance(v, dict)}
        except Exception:
            assignment = {}

    grouped = {}
    for line in transcript or []:
        speaker = str(line.get('speaker') or 'SPEAKER_00')
        grouped.setdefault(speaker, []).append(line.get('text') or '')

    used_fish = {item.get('reference_id') for item in assignment.values() if item.get('reference_id')}
    used_openai = {item.get('openai') for item in assignment.values() if item.get('openai')}
    used_edge = {item.get('edge') for item in assignment.values() if item.get('edge')}
    used_roles = {item.get('role') for item in assignment.values() if item.get('role')}
    for info in assignment.values():
        if info.get('cloned'):
            continue
        role = info.get('role')
        rid = info.get('reference_id')
        if role in FISH_STOCK_VOICES and rid:
            expected = FISH_STOCK_VOICES[role]
            opposite = FISH_STOCK_VOICES['female' if role == 'male' else 'male'] if role in ('male', 'female') else None
            if opposite and rid == opposite:
                used_fish.discard(rid)
                info['reference_id'] = expected
                used_fish.add(expected)
        if info.get('reference_id') and not info.get('voice_name'):
            info['voice_name'] = FISH_VOICE_NAMES.get(info['reference_id'], '')
    env_voice = (os.getenv('FISH_VOICE_ID') or '').strip()
    edge_stock, edge_male, edge_female = _edge_voices_for_language(target_language)
    do_clone = bool(clone) and _clone_enabled()

    ranked = []
    for speaker, texts in grouped.items():
        info = assignment.get(speaker) or {}
        if _clone_quality(info) >= CLONE_QUALITY:
            continue
        if not do_clone and info.get('reference_id'):
            continue
        pitch = _median_pitch(os.path.join(folder, 'SPEAKER', f'{speaker}.wav'))
        role, score = _guess_role(''.join(texts), pitch)
        ranked.append((sum(len(t) for t in texts), speaker, role, score, pitch))
    ranked.sort(key=lambda item: (-item[0], item[1]))

    if env_voice and ranked and not do_clone:
        main_speaker = ranked[0][1]
        if main_speaker not in assignment:
            assignment[main_speaker] = {}
        assignment[main_speaker]['reference_id'] = env_voice
        assignment[main_speaker].setdefault('role', ranked[0][2])
        used_fish.add(env_voice)

    cloned_now = False
    main_speaker = ranked[0][1] if ranked else None
    for _chars, speaker, role, _score, pitch in ranked:
        info = assignment.setdefault(speaker, {})
        role = _pick_available_role(role, used_roles)
        if _clone_quality(info) < CLONE_QUALITY or not info.get('role'):
            info['role'] = role
        else:
            info['role'] = info.get('role') or role
        used_roles.add(info['role'])
        if pitch is not None:
            info['pitch_hz'] = round(pitch, 1)

        clone_id = None
        clone_sec = 0.0
        sample = None
        wav_path = os.path.join(folder, 'SPEAKER', f'{speaker}.wav')
        if do_clone:
            info['clone_attempted'] = True
            info['clone_quality'] = CLONE_QUALITY
            wavs, clone_sec = prepare_clone_wavs(folder, speaker, transcript)
            sample = wavs[0] if wavs else None
            if wavs:
                logger.info(f'克隆 {speaker}（{len(wavs)} 段共 {clone_sec:.1f}s）')
                clone_id = _create_fish_clone(_clone_title(folder, speaker, wavs[0]), wavs)
                if not clone_id:
                    logger.warning(f'{speaker} 克隆失敗，再試一次')
                    clone_id = _create_fish_clone(_clone_title(folder, speaker, wavs[0]), wavs)
            elif not os.path.isfile(wav_path):
                logger.warning(f'{speaker} 沒有 SPEAKER 音檔，改用庫存聲線')
            else:
                logger.warning(f'{speaker} 樣本過短或過靜，改用庫存聲線')

        if clone_id:
            old_id = info.get('reference_id')
            if old_id and old_id not in {clone_id}:
                used_fish.discard(old_id)
            info['reference_id'] = clone_id
            info['cloned'] = True
            info['voice_name'] = 'clone'
            info['clone_seconds'] = round(clone_sec, 1)
            used_fish.add(clone_id)
            cloned_now = True
            logger.info(f'{speaker} 克隆完成')
        else:
            info['cloned'] = False
            if not info.get('reference_id'):
                if env_voice and speaker == main_speaker:
                    info['reference_id'] = env_voice
                else:
                    info['reference_id'] = _voice_for_role(
                        info['role'], used_fish, FISH_STOCK_VOICES,
                        FISH_IDS_MALE, FISH_IDS_FEMALE, FISH_STOCK_VOICES['male'],
                    )
            used_fish.add(info['reference_id'])
            info['voice_name'] = FISH_VOICE_NAMES.get(info['reference_id'], info.get('voice_name') or '')
            if do_clone and sample and not clone_id:
                logger.warning(
                    f'{speaker} 克隆失敗，改用 {info.get("voice_name") or info.get("reference_id")}'
                )

        if not info.get('openai'):
            info['openai'] = _voice_for_role(
                info['role'], used_openai, OPENAI_STOCK_VOICES,
                OPENAI_IDS_MALE, OPENAI_IDS_FEMALE, OPENAI_STOCK_VOICES['male'],
            )
        used_openai.add(info['openai'])
        if not info.get('edge'):
            info['edge'] = _voice_for_role(
                info['role'], used_edge, edge_stock,
                edge_male, edge_female, edge_stock['male'],
            )
        used_edge.add(info['edge'])

    os.makedirs(folder, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(assignment, handle, indent=2, ensure_ascii=False)
    roles = {
        k: f"{v.get('role')}/{'clone' if v.get('cloned') else (v.get('voice_name') or v.get('reference_id'))}"
        for k, v in assignment.items()
    }
    logger.info(f'角色聲線: {roles}')
    return assignment, cloned_now


def voice_for_method(assignment, speaker, method, ui_voice=None):
    info = (assignment or {}).get(str(speaker)) or {}
    if method == 'Fish':
        return info.get('reference_id') or (os.getenv('FISH_VOICE_ID') or '').strip() or None
    if method == 'OpenAI':
        return info.get('openai') or ui_voice
    if method == 'EdgeTTS':
        return info.get('edge') or ui_voice
    return ui_voice


def _fix_wav_header(path):
    """Fish often writes 0xFFFFFFFF chunk sizes; wave.getnframes then looks like hours."""
    try:
        with open(path, 'rb') as handle:
            data = handle.read()
        if len(data) < 44 or data[:4] != b'RIFF' or data[8:12] != b'WAVE':
            return
        size = len(data)
        data_idx = data.find(b'data', 12)
        if data_idx < 0:
            return
        riff_size = (size - 8).to_bytes(4, 'little')
        payload = (size - data_idx - 8).to_bytes(4, 'little')
        patched = bytearray(data)
        patched[4:8] = riff_size
        patched[data_idx + 4:data_idx + 8] = payload
        if patched != data:
            with open(path, 'wb') as handle:
                handle.write(patched)
    except Exception:
        return


def tts(text, output_path, target_language='中文', voice=None, speed=1.0):
    global _use_paid_model
    if os.path.exists(output_path):
        logger.info(f'TTS {text} 已存在')
        return
    from tools.api_keys import next_api_key
    url = os.getenv('FISH_API_URL', 'https://api.fish.audio/v1/tts')
    model = FISH_TTS_PAID if _use_paid_model else FISH_TTS_FREE
    env_voice = (os.getenv('FISH_VOICE_ID') or '').strip()
    reference_id = None
    if voice and not str(voice).endswith('Neural') and str(voice) not in _SKIP_UI_VOICES:
        reference_id = voice
    elif env_voice:
        reference_id = env_voice
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    wav_path = output_path if output_path.endswith('.wav') else output_path.replace('.mp3', '.wav')
    last_error = None
    delays = (0.0, 1.0, 2.0, 4.0, 8.0)
    from tools.job_control import check_stop
    for attempt, delay in enumerate(delays, start=1):
        check_stop()
        if delay:
            time.sleep(delay)
        api_key, n_keys, key_index = next_api_key(
            'FISH_API_KEY', error='請先在 .env 設定 FISH_API_KEY'
        )
        from tools.target_language import tts_language
        japanese = tts_language(target_language) == 'Japanese'
        payload = {
            'text': text,
            'format': 'wav',
            'sample_rate': 44100,
            # Chinese number normalization makes CJK kanji sound Mandarin.
            'normalize': not japanese,
            'latency': 'normal',
            'temperature': 0.85 if japanese else 0.7,
            'prosody': {'speed': float(min(2.0, max(0.5, speed or 1.0))), 'volume': 0, 'normalize_loudness': True},
        }
        if japanese:
            payload['language'] = 'ja'
        if reference_id:
            payload['reference_id'] = reference_id
        try:
            response = requests.post(
                url,
                headers={
                    'Authorization': f'Bearer {api_key}',
                    'Content-Type': 'application/json',
                    'model': model,
                },
                json=payload,
                timeout=120,
            )
            if response.status_code == 200 and response.content:
                with open(wav_path, 'wb') as f:
                    f.write(response.content)
                _fix_wav_header(wav_path)
                from tools.cost_tracker import record
                spoken = spoken_text_for_timing(text)
                record('fish', 'tts_fish', model, utf8_bytes=len(spoken.encode('utf-8')), characters=len(spoken))
                logger.info(f'Fish TTS {text}')
                return
            last_error = f'HTTP {response.status_code}: {response.text[:200]}'
            if model == FISH_TTS_FREE and response.status_code in (400, 402, 403, 404, 422):
                logger.warning(
                    f'Fish 免費模型不可用 ({response.status_code})，改用付費 {FISH_TTS_PAID}'
                )
                _use_paid_model = True
                model = FISH_TTS_PAID
                continue
            if response.status_code == 429:
                extra = f'，改試另一把 key' if n_keys > 1 else ''
                logger.warning(
                    f'Fish TTS 限流 ({attempt}/{len(delays)} key {key_index}/{n_keys}){extra}'
                )
                continue
            logger.warning(f'Fish TTS 失敗 ({attempt}/{len(delays)}): {last_error}')
        except Exception as e:
            last_error = str(e)
            logger.warning(f'Fish TTS 失敗 ({attempt}/{len(delays)}): {e}')
    raise RuntimeError(f'Fish TTS 失敗: {last_error}')
