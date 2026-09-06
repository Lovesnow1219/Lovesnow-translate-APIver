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

from tools.fish_emotion import apply_emotion_tags, delivery_temperature, spoken_text_for_timing
from tools.fish_voice_match import FISH_EXTRA_MATCH_VOICES

load_dotenv()

FISH_TTS_FREE = 's2.1-pro-free'
FISH_TTS_PAID = 's2.1-pro'
_use_paid_model = False

# Fish Official default roles. Matching also uses the full official English catalog
# in tools/fish_official_voices.py (~70 voices). Laura/system stays reserved.
# https://fish.audio/m/<id>
FISH_VOICE_BANK = (
    {'role': 'male', 'id': '79d0bd3e4e5444b18f7b6d89b5927bf1', 'name': 'Jordan', 'gender': 'male', 'age': 'elder'},
    {'role': 'female', 'id': '9a9cf47702da476aa4629e2506d4a857', 'name': 'Hannah', 'gender': 'female', 'age': 'adult'},
    {'role': 'young_female', 'id': '933563129e564b19a115bedd57b7406a', 'name': 'Sarah', 'gender': 'female', 'age': 'young'},
    {'role': 'young_male', 'id': '536d3a5e000945adb7038665781a4aca', 'name': 'Ethan', 'gender': 'male', 'age': 'adult'},
    {'role': 'boy', 'id': 'f983ef416c87413f986e1a5a000c385e', 'name': 'Miles', 'gender': 'male', 'age': 'young'},
    {'role': 'girl', 'id': 'f3a2b90078d54a65af4b95f63bc7798e', 'name': 'Sheila', 'gender': 'female', 'age': 'young'},
    {'role': 'system', 'id': 'e3cd384158934cc9a01029cd7d278634', 'name': 'Laura', 'gender': 'female', 'age': 'adult'},
    {'role': 'narrator', 'id': 'bf322df2096a46f18c579d0baa36f41d', 'name': 'Adrian', 'gender': 'male', 'age': 'adult'},
    {'role': 'mature_female', 'id': 'b347db033a6549378b48d00acb0d06cd', 'name': 'Selene', 'gender': 'female', 'age': 'adult'},
)
FISH_STOCK_VOICES = {item['role']: item['id'] for item in FISH_VOICE_BANK}
FISH_VOICE_NAMES = {item['id']: item['name'] for item in FISH_VOICE_BANK}
for _item in FISH_EXTRA_MATCH_VOICES:
    FISH_VOICE_NAMES[_item['id']] = _item['name']
FISH_IDS_MALE = tuple(dict.fromkeys(
    [item['id'] for item in FISH_VOICE_BANK if item['gender'] == 'male']
    + [item['id'] for item in FISH_EXTRA_MATCH_VOICES if item['gender'] == 'male']
))
FISH_IDS_FEMALE = tuple(dict.fromkeys(
    [item['id'] for item in FISH_VOICE_BANK if item['gender'] == 'female' and item['role'] != 'system']
    + [item['id'] for item in FISH_EXTRA_MATCH_VOICES if item['gender'] == 'female']
))
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
_SYSTEM_MARKERS = ('特殊能力', '技能', '效果', '单位', '指數', '指数', '状态', '狀態', '施加', 'HP', 'MP', 'Lv')
_NARRATOR_MARKERS = ('轻轻', '心想', '只见', '说道', '了声', '他大概', '她心里', '旁白')
_YOUNG_FEMALE_MARKERS = ('淑女', '人家', '讨厌', '討厭')
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
        from tools.line_roles import is_clerk_line, is_functional_speaker, is_system_line
        if is_functional_speaker(speaker):
            continue
        low, high = [], []
        for line in transcript:
            if str(line.get('speaker') or '') != speaker:
                continue
            src = line.get('text') or ''
            if is_system_line(src) or is_clerk_line(src):
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


def _looks_like_address(text):
    """Short vocative / greeting. Do not move by pitch — the name is the listener."""
    src = (text or '').strip()
    han = len(re.findall(r'[\u4e00-\u9fff]', src))
    if han == 0 or han > 8:
        return False
    if re.search(r'[，,].{1,4}$', src):
        return True
    return bool(re.match(r'^[早嘿喂嗨你您].{0,6}$', src))


def reassign_crossed_pitch_speakers(folder, transcript):
    """Move clearly high/low lines to the other lead instead of inventing SPEAKER_*_F."""
    from tools.asr_source import _is_particle_only
    from tools.line_roles import is_functional_speaker

    y, sr = _load_vocals(folder, sr=16000)
    if y is None or not transcript:
        return transcript, False
    grouped = {}
    for line in transcript:
        speaker = str(line.get('speaker') or 'SPEAKER_00')
        if is_functional_speaker(speaker) or speaker.endswith('_F') or speaker.endswith('_M'):
            continue
        grouped.setdefault(speaker, []).append(line)
    if len(grouped) < 2:
        return transcript, False

    def _line_pitch(line):
        start = float(line.get('start') or 0)
        end = float(line.get('end') or 0)
        if end - start < 0.4:
            return None
        if _is_particle_only(line.get('text')):
            return None
        i0 = int(max(0, start * sr))
        i1 = int(min(len(y), end * sr))
        return _clip_pitch(y[i0:i1], sr)

    stats = []
    for speaker, lines in grouped.items():
        pitches = []
        duration = 0.0
        for line in lines:
            duration += max(0.0, float(line.get('end') or 0) - float(line.get('start') or 0))
            pitch = _line_pitch(line)
            if pitch:
                pitches.append(pitch)
        if len(pitches) < 3 or duration < 6:
            continue
        stats.append((duration, speaker, float(np.median(pitches))))
    stats.sort(reverse=True)
    if len(stats) < 2:
        return transcript, False
    _dur_a, speaker_a, med_a = stats[0]
    _dur_b, speaker_b, med_b = stats[1]
    if abs(med_a - med_b) < 40:
        return transcript, False
    if med_a >= med_b:
        high_spk, low_spk, high_med, low_med = speaker_a, speaker_b, med_a, med_b
    else:
        high_spk, low_spk, high_med, low_med = speaker_b, speaker_a, med_b, med_a

    moved = 0
    mid = (high_med + low_med) / 2
    high_cut = max(_PITCH_HIGH, mid + 15)
    low_cut = min(_PITCH_LOW, mid - 15)
    for line in transcript:
        speaker = str(line.get('speaker') or '')
        parent = speaker[:-2] if speaker.endswith(('_F', '_M')) else speaker
        pitch = _line_pitch(line)
        if pitch is None:
            continue
        text = line.get('text') or ''
        if _looks_like_address(text):
            continue
        if parent == low_spk and pitch >= high_cut:
            line['speaker'] = high_spk
            moved += 1
        elif parent == high_spk and pitch <= low_cut:
            line['speaker'] = low_spk
            moved += 1
    if moved:
        logger.info(
            f'依音高把 {moved} 句改回兩個主角（高 {high_spk} {high_med:.0f}Hz／低 {low_spk} {low_med:.0f}Hz）'
        )
    return transcript, moved > 0


def _clone_title(folder, speaker, wav_bytes):
    digest = hashlib.sha1(wav_bytes[:8192] + str(len(wav_bytes)).encode('utf-8')).hexdigest()[:8]
    base = re.sub(r'[^\w\-]+', '_', os.path.basename(os.path.abspath(folder)))[:24]
    return f'Lovesnow_{base}_{speaker}_{digest}'[:80]


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
                    'description': 'Lovesnow-translate speaker clone',
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


def _lock_quality(info):
    try:
        return int((info or {}).get('match_quality') or 0)
    except (TypeError, ValueError):
        return 0


def _load_voice_assignment(folder):
    path = os.path.join(folder, 'speaker_voices.json')
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, 'r', encoding='utf-8') as handle:
            raw = json.load(handle)
        if isinstance(raw, dict):
            return {str(k): v for k, v in raw.items() if isinstance(v, dict)}
    except Exception:
        pass
    return {}


def fish_clone_pending(folder, transcript):
    """True when Fish voices are not locked yet (match or clone)."""
    from tools.fish_voice_match import MATCH_QUALITY, match_enabled, speaker_needs_match
    assignment = _load_voice_assignment(folder)
    speakers = {str(line.get('speaker') or 'SPEAKER_00') for line in (transcript or [])}
    if not speakers:
        return False
    if match_enabled():
        for speaker in speakers:
            if not speaker_needs_match(speaker):
                continue
            info = assignment.get(speaker) or {}
            if _lock_quality(info) >= MATCH_QUALITY and info.get('reference_id'):
                continue
            return True
        return False
    if not _clone_enabled():
        return False
    from tools.line_roles import is_functional_speaker
    for speaker in speakers:
        if is_functional_speaker(speaker):
            continue
        info = assignment.get(speaker) or {}
        if _clone_quality(info) >= CLONE_QUALITY:
            continue
        return True
    return False


def assign_speaker_voices(folder, transcript, target_language='English', clone=True):
    """Assign Fish voices: closest official stock voice, else clone, else role stock."""
    from tools.fish_voice_match import (
        MATCH_QUALITY,
        all_match_voices,
        ensure_stock_prints,
        gender_from_pitch,
        match_enabled,
        pick_closest,
        speaker_needs_match,
        speaker_print,
    )
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

    do_match = match_enabled()
    used_fish = set()
    used_roles = {item.get('role') for item in assignment.values() if item.get('role')}
    for speaker, info in assignment.items():
        rid = info.get('reference_id')
        if not rid:
            continue
        locked = _lock_quality(info) >= MATCH_QUALITY if do_match else True
        if do_match and speaker_needs_match(speaker) and not locked:
            continue
        used_fish.add(rid)
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
    do_clone = bool(clone) and _clone_enabled()

    from tools.line_roles import SPEAKER_BGM, SPEAKER_CLERK, SPEAKER_NARR, SPEAKER_SYS, is_functional_speaker
    ranked = []
    for speaker, texts in grouped.items():
        if speaker == SPEAKER_BGM:
            continue
        info = assignment.get(speaker) or {}
        if is_functional_speaker(speaker) and speaker != SPEAKER_CLERK:
            info = assignment.setdefault(speaker, {})
            if speaker == SPEAKER_SYS:
                role = 'system'
            elif speaker == SPEAKER_NARR:
                role = 'narrator'
            else:
                role = 'male'
            info['role'] = role
            info['cloned'] = False
            if not info.get('reference_id'):
                info['reference_id'] = FISH_STOCK_VOICES[role] if role in FISH_STOCK_VOICES else FISH_STOCK_VOICES['male']
            info['voice_name'] = FISH_VOICE_NAMES.get(info['reference_id'], '')
            used_fish.add(info['reference_id'])
            used_roles.add(info['role'])
            continue
        if do_match:
            if _lock_quality(info) >= MATCH_QUALITY and info.get('reference_id'):
                continue
        else:
            if _clone_quality(info) >= CLONE_QUALITY:
                continue
            if not do_clone and info.get('reference_id'):
                continue
        pitch = _median_pitch(os.path.join(folder, 'SPEAKER', f'{speaker}.wav'))
        role, score = _guess_role(''.join(texts), pitch)
        ranked.append((sum(len(t) for t in texts), speaker, role, score, pitch))
    ranked.sort(key=lambda item: (-item[0], item[1]))

    if env_voice and ranked and not do_clone and not do_match:
        main_speaker = ranked[0][1]
        if main_speaker not in assignment:
            assignment[main_speaker] = {}
        assignment[main_speaker]['reference_id'] = env_voice
        assignment[main_speaker].setdefault('role', ranked[0][2])
        used_fish.add(env_voice)

    cloned_now = False
    matched_now = False
    stock_prints = None
    match_voices = all_match_voices(FISH_VOICE_BANK)
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

        matched = None
        if do_match:
            if stock_prints is None:
                stock_prints = ensure_stock_prints(match_voices, tts_fn=tts)
            vec = speaker_print(folder, speaker)
            if vec is not None and stock_prints:
                matched = pick_closest(
                    vec,
                    match_voices,
                    stock_prints,
                    used_ids=used_fish,
                    gender=gender_from_pitch(pitch),
                    reserved_ids={FISH_STOCK_VOICES['system']},
                )
            elif vec is None:
                logger.warning(f'{speaker} 沒有可用原聲，無法擬合聲線')

        if matched:
            old_id = info.get('reference_id')
            if old_id:
                used_fish.discard(old_id)
            info['reference_id'] = matched['id']
            info['cloned'] = False
            info['voice_name'] = matched['name']
            info['match_quality'] = MATCH_QUALITY
            info['match_score'] = round(matched['score'], 3)
            info['match_top'] = matched['top']
            if matched.get('role'):
                info['role'] = _pick_available_role(matched['role'], used_roles - {info.get('role')})
                used_roles.add(info['role'])
            used_fish.add(matched['id'])
            matched_now = True
            nxt = matched['top'][1]['name'] if len(matched.get('top') or []) > 1 else ''
            extra = f'，其次 {nxt}' if nxt else ''
            logger.info(f'{speaker} 鎖定 {matched["name"]}（{matched["score"]:.2f}）{extra}')
        else:
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
                info['match_quality'] = MATCH_QUALITY
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
                info['match_quality'] = MATCH_QUALITY
                if do_clone and sample and not clone_id:
                    logger.warning(
                        f'{speaker} 克隆失敗，改用 {info.get("voice_name") or info.get("reference_id")}'
                    )

    os.makedirs(folder, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(assignment, handle, indent=2, ensure_ascii=False)
    roles = {
        k: f"{v.get('role')}/{'clone' if v.get('cloned') else (v.get('voice_name') or v.get('reference_id'))}"
        for k, v in assignment.items()
    }
    logger.info(f'角色聲線: {roles}')
    return assignment, cloned_now or matched_now


def voice_for_method(assignment, speaker, method='Fish', ui_voice=None):
    info = (assignment or {}).get(str(speaker)) or {}
    return info.get('reference_id') or (os.getenv('FISH_VOICE_ID') or '').strip() or ui_voice


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
            'temperature': delivery_temperature(text, japanese=japanese),
            'top_p': 0.8,
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
