"""Per-episode delivery settings and cache identities; never store API keys."""
import hashlib
import json
import math
import os
import tempfile


def read_json(path, default=None):
    try:
        with open(path, encoding='utf-8') as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return default


def write_json_atomic(path, data):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=directory, suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def fingerprint(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode('utf-8')).hexdigest()


def file_identity(path):
    if not path:
        return None
    path = os.path.abspath(os.fspath(path))
    try:
        stat = os.stat(path)
        return [path, stat.st_size, stat.st_mtime_ns]
    except OSError:
        return [path, None, None]


def validate_tts_speed(value):
    speed = float(value)
    if not math.isfinite(speed) or not 0.85 <= speed <= 1.15:
        raise ValueError('配音語速須介於 0.85–1.15 倍')
    return speed


def load_delivery_settings(folder):
    settings = read_json(os.path.join(folder, 'dubbing_settings.json'), {})
    if not isinstance(settings, dict):
        settings = {}
    return {'tts_speed': validate_tts_speed(settings.get('tts_speed', 1.0))}


def save_delivery_settings(folder, tts_speed):
    settings = {'tts_speed': validate_tts_speed(tts_speed)}
    write_json_atomic(os.path.join(folder, 'dubbing_settings.json'), settings)
    return settings


def render_fingerprint(folder, language, voice=None):
    from tools.target_language import translation_language
    lines = read_json(os.path.join(folder, 'translation.json'), [])
    # Actual placement changes during assembly; original picture times do not.
    source = []
    for line in lines:
        row = {key: value for key, value in line.items()
               if key not in {'start', 'end', 'dub_timing'}}
        row['orig_start'] = line.get('orig_start', line.get('start'))
        row['orig_end'] = line.get('orig_end', line.get('end'))
        source.append(row)
    wav_dir = os.path.join(folder, 'wavs')
    wavs = [file_identity(os.path.join(wav_dir, f'{i:04d}.wav')) for i in range(len(lines))]
    return fingerprint({
        'version': 1, 'language': translation_language(language), 'lines': source,
        'settings': load_delivery_settings(folder), 'voice': voice,
        'voices': read_json(os.path.join(folder, 'speaker_voices.json'), {}),
        'voice_override': os.getenv('FISH_VOICE_ID', ''),
        'endpoint': os.getenv('FISH_API_URL', 'https://api.fish.audio/v1/tts'),
        'wavs': wavs,
        'stems': [file_identity(os.path.join(folder, name)) for name in
                  ('audio.wav', 'audio_vocals.wav', 'audio_instruments.wav')],
    })


def stamp_render(folder, language, voice=None):
    write_json_atomic(os.path.join(folder, 'tts_render.json'),
                      {'fingerprint': render_fingerprint(folder, language, voice),
                       'outputs': [file_identity(os.path.join(folder, name))
                                   for name in ('audio_tts.wav', 'audio_combined.wav')]})


def render_is_current(folder, language, voice=None):
    stamp = read_json(os.path.join(folder, 'tts_render.json'), {})
    return (isinstance(stamp, dict)
            and stamp.get('fingerprint') == render_fingerprint(folder, language, voice)
            and stamp.get('outputs') == [file_identity(os.path.join(folder, name))
                                        for name in ('audio_tts.wav', 'audio_combined.wav')])


def save_style_note(folder, note):
    """The editable creative brief belongs to this episode, not global rules."""
    path = os.path.join(folder, 'dubbing_style.json')
    write_json_atomic(path, {'note': str(note or '').strip()[:4000]})
    summary_path = os.path.join(folder, 'summary.json')
    summary = read_json(summary_path, {})
    if isinstance(summary, dict) and summary:
        summary['dubbing_style'] = str(note or '').strip()[:4000]
        write_json_atomic(summary_path, summary)


def style_note(folder):
    data = read_json(os.path.join(folder, 'dubbing_style.json'), {})
    return str(data.get('note') or '') if isinstance(data, dict) else ''
