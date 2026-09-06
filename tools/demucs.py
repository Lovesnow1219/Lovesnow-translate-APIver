# -*- coding: utf-8 -*-
import os
import subprocess
import tempfile
import time

import httpx
import requests
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

# Replicate Demucs：完整 4-stem，不要 stem=vocals
REPLICATE_DEMUCS_REF = os.getenv(
    'REPLICATE_DEMUCS_MODEL',
    'cjwbw/demucs:abf8fe28e407afa6d8e41e86a759caccc0af8e49c3c68016006b62cb0968441e',
)
_STEM_KEYS = ('vocals', 'drums', 'bass', 'other', 'guitar', 'piano')
_BED_KEYS = ('drums', 'bass', 'other', 'guitar', 'piano')


def _token():
    token = (os.getenv('REPLICATE_API_TOKEN') or os.getenv('REPLICATE_API_KEY') or '').strip()
    if not token:
        raise ValueError('請先在 .env 設定 REPLICATE_API_TOKEN')
    return token


def use_replicate(model_name: str) -> bool:
    if model_name == 'SIG':
        return False
    return bool((os.getenv('REPLICATE_API_TOKEN') or os.getenv('REPLICATE_API_KEY') or '').strip())


def _output_map(output):
    data = {}
    if isinstance(output, dict):
        data = output
    else:
        for name in ('model_dump', 'dict'):
            converter = getattr(output, name, None)
            if callable(converter):
                try:
                    dumped = converter()
                    if isinstance(dumped, dict):
                        data = dumped
                        break
                except Exception:
                    pass
        if not data:
            items = getattr(output, 'items', None)
            if callable(items):
                try:
                    data = dict(items())
                except Exception:
                    data = {}
    return {k: v for k, v in data.items() if k in _STEM_KEYS and v not in (None, '', False)}


def _as_url(value):
    if value is None:
        return None
    if isinstance(value, str) and value.startswith('http'):
        return value
    url = getattr(value, 'url', None)
    if callable(url):
        try:
            url = url()
        except TypeError:
            url = None
    if isinstance(url, str) and url.startswith('http'):
        return url
    text = str(value)
    if text.startswith('http'):
        return text
    return None


def _download_to_wav(source, dest_wav, sample_rate=44100):
    os.makedirs(os.path.dirname(dest_wav) or '.', exist_ok=True)
    suffix = '.wav'
    url = _as_url(source)
    if url:
        suffix = os.path.splitext(url.split('?')[0])[1] or '.wav'
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name
    try:
        if url:
            resp = requests.get(url, timeout=300)
            resp.raise_for_status()
            with open(tmp_path, 'wb') as f:
                f.write(resp.content)
        elif hasattr(source, 'read'):
            with open(tmp_path, 'wb') as f:
                f.write(source.read())
        else:
            return False
        subprocess.run(
            ['ffmpeg', '-y', '-i', tmp_path, '-acodec', 'pcm_s16le', '-ar', str(sample_rate), '-ac', '2', dest_wav],
            check=True,
            capture_output=True,
        )
        return True
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


DEMUCS_CHUNK_SECONDS = float(os.getenv('DEMUCS_CHUNK_SECONDS') or 480)
DEMUCS_OVERLAP_SECONDS = float(os.getenv('DEMUCS_OVERLAP_SECONDS') or 2.0)
REPLICATE_MAX_UPLOAD_BYTES = int(os.getenv('REPLICATE_MAX_UPLOAD_BYTES') or (95 * 1024 * 1024))
DEMUCS_SAMPLE_RATE = 44100
DEMUCS_CHANNELS = 2


def _replicate_client():
    import replicate

    timeout = httpx.Timeout(connect=60.0, read=900.0, write=900.0, pool=60.0)
    return replicate.Client(api_token=_token(), timeout=timeout)


def _is_transient(exc):
    if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError)):
        return True
    text = str(exc).lower()
    return any(token in text for token in ('timed out', 'timeout', 'temporarily', 'connection reset', '503', '502'))


def _upload_audio(client, audio_path):
    size_mb = os.path.getsize(audio_path) / (1024 * 1024)
    logger.info(f'上傳音訊到 Replicate: {audio_path} ({size_mb:.1f} MB)')
    uploaded = client.files.create(audio_path)
    url = (getattr(uploaded, 'urls', None) or {}).get('get')
    if not url:
        raise RuntimeError('Replicate 檔案上傳成功但沒有下載網址')
    return url


def _mix_wavs(wav_paths, dest_wav):
    if not wav_paths:
        return False
    if len(wav_paths) == 1:
        os.replace(wav_paths[0], dest_wav)
        return True
    inputs = []
    for path in wav_paths:
        inputs.extend(['-i', path])
    subprocess.run(
        ['ffmpeg', '-y', *inputs, '-filter_complex', f'amix=inputs={len(wav_paths)}:duration=longest:normalize=0', dest_wav],
        check=True,
        capture_output=True,
    )
    return True


def _separate_one(audio_path, vocal_output_path, instruments_output_path, model_name='htdemucs_ft', shifts=2):
    client = _replicate_client()
    api_model = model_name if model_name in (
        'htdemucs', 'htdemucs_ft', 'htdemucs_6s', 'hdemucs_mmi', 'mdx', 'mdx_extra', 'mdx_q', 'mdx_extra_q'
    ) else 'htdemucs_ft'
    logger.info(f'Replicate Demucs 4-stem 分離: {audio_path} model={api_model}')
    audio_url = None
    output = None
    last_error = None
    gpu_seconds = 0.0
    for attempt in range(1, 4):
        from tools.job_control import check_stop
        check_stop()
        try:
            if audio_url is None:
                audio_url = _upload_audio(client, audio_path)
            logger.info(f'等待 Replicate Demucs 完成（第 {attempt} 次）')
            t0 = time.monotonic()
            output = client.run(
                REPLICATE_DEMUCS_REF,
                input={
                    'audio': audio_url,
                    'model_name': api_model,
                    'clip_mode': 'rescale',
                    'output_format': 'wav',
                    'shifts': max(1, min(int(shifts or 2), 5)),
                    'overlap': 0.5,
                },
                wait=False,
            )
            gpu_seconds = time.monotonic() - t0
            last_error = None
            break
        except Exception as e:
            last_error = e
            if attempt >= 3 or not _is_transient(e):
                raise
            sleep_s = min(30, 5 * attempt)
            logger.warning(f'Replicate 暫時失敗，{sleep_s}s 後重試: {e}')
            audio_url = None
            for _ in range(int(sleep_s)):
                check_stop()
                time.sleep(1)
    if output is None:
        raise last_error or RuntimeError('Replicate Demucs 沒有回傳結果')
    stems = _output_map(output)
    logger.info(f'Replicate stems: {list(stems.keys())}')
    if 'vocals' not in stems:
        raise RuntimeError(f'Replicate 未回傳人聲: {list(stems.keys())}')
    if not _download_to_wav(stems['vocals'], vocal_output_path):
        raise RuntimeError('下載人聲失敗')
    bed_parts = [stems[k] for k in _BED_KEYS if k in stems]
    if not bed_parts:
        raise RuntimeError('Replicate 未回傳伴奏 stems')
    tmp_dir = os.path.dirname(instruments_output_path) or '.'
    tmp_wavs = []
    try:
        for i, part in enumerate(bed_parts):
            tmp = os.path.join(tmp_dir, f'_bed_{i}.wav')
            if _download_to_wav(part, tmp):
                tmp_wavs.append(tmp)
        if not _mix_wavs(tmp_wavs, instruments_output_path):
            raise RuntimeError('混合伴奏失敗')
        tmp_wavs = []
    finally:
        for path in tmp_wavs:
            if os.path.exists(path):
                os.remove(path)
    logger.info(f'Replicate 人聲分離完成: {vocal_output_path}')
    from tools.cost_tracker import record
    record('replicate', 'demucs_replicate', api_model, seconds=gpu_seconds)
    return vocal_output_path, instruments_output_path


def separate_with_replicate(audio_path, vocal_output_path, instruments_output_path, model_name='htdemucs_ft', shifts=2):
    from tools.audio_chunks import cut_audio, plan_chunks_for_file, stitch_wavs

    hop = float(os.getenv('DEMUCS_CHUNK_SECONDS') or DEMUCS_CHUNK_SECONDS)
    overlap = float(os.getenv('DEMUCS_OVERLAP_SECONDS') or DEMUCS_OVERLAP_SECONDS)
    max_bytes = int(os.getenv('REPLICATE_MAX_UPLOAD_BYTES') or REPLICATE_MAX_UPLOAD_BYTES)
    chunks, duration = plan_chunks_for_file(
        audio_path,
        hop=hop,
        overlap=overlap,
        max_bytes=max_bytes,
        sample_rate=DEMUCS_SAMPLE_RATE,
        channels=DEMUCS_CHANNELS,
    )
    if len(chunks) <= 1:
        return _separate_one(
            audio_path, vocal_output_path, instruments_output_path,
            model_name=model_name, shifts=shifts,
        )

    logger.info(
        f'音訊 {duration:.0f}s 超過 Replicate 上傳上限，分成 {len(chunks)} 段 '
        f'(約 {hop:.0f}s + {overlap:.1f}s 重疊)'
    )
    chunk_dir = os.path.join(os.path.dirname(vocal_output_path) or '.', '_demucs_chunks')
    os.makedirs(chunk_dir, exist_ok=True)
    vocal_parts = []
    bed_parts = []
    from tools.job_control import check_stop
    for index, (start, end) in enumerate(chunks):
        check_stop()
        inp = os.path.join(chunk_dir, f'input_{index:03d}.wav')
        vox = os.path.join(chunk_dir, f'vocals_{index:03d}.wav')
        bed = os.path.join(chunk_dir, f'bed_{index:03d}.wav')
        if not os.path.isfile(inp):
            cut_audio(
                audio_path, inp, start, end,
                sample_rate=DEMUCS_SAMPLE_RATE, channels=DEMUCS_CHANNELS,
            )
        if not (os.path.isfile(vox) and os.path.isfile(bed)):
            tmp = os.path.join(chunk_dir, f'tmp_{index:03d}')
            os.makedirs(tmp, exist_ok=True)
            _separate_one(
                inp,
                os.path.join(tmp, 'vocals.wav'),
                os.path.join(tmp, 'bed.wav'),
                model_name=model_name,
                shifts=shifts,
            )
            os.replace(os.path.join(tmp, 'vocals.wav'), vox)
            os.replace(os.path.join(tmp, 'bed.wav'), bed)
        vocal_parts.append((start, end, vox))
        bed_parts.append((start, end, bed))
    stitch_wavs(
        vocal_parts, vocal_output_path, duration, DEMUCS_SAMPLE_RATE,
        overlap_seconds=overlap,
    )
    stitch_wavs(
        bed_parts, instruments_output_path, duration, DEMUCS_SAMPLE_RATE,
        overlap_seconds=overlap,
    )
    logger.info(f'Replicate 分段人聲分離完成: {vocal_output_path}')
    return vocal_output_path, instruments_output_path


def init_demucs():
    logger.info('人聲分離使用 Replicate，無需本機 Demucs')


def release_model():
    return


def separate_audio(folder: str, model_name: str = "htdemucs_ft", device: str = 'auto', progress: bool = True,
                   shifts: int = 5) -> None:
    audio_path = os.path.join(folder, 'audio.wav')
    if not os.path.exists(audio_path):
        return None, None
    vocal_output_path = os.path.join(folder, 'audio_vocals.wav')
    instruments_output_path = os.path.join(folder, 'audio_instruments.wav')

    from tools.target_language import (
        clear_asr_downstream,
        recorded_demucs_shifts,
        save_dub_meta,
    )

    wanted = max(1, int(shifts if shifts is not None else 1))
    if os.path.exists(vocal_output_path) and os.path.exists(instruments_output_path):
        if recorded_demucs_shifts(folder) == wanted:
            logger.info(f'音訊已分離: {folder}')
            return vocal_output_path, instruments_output_path
        logger.info(f'shifts {recorded_demucs_shifts(folder)}→{wanted}，重跑人聲分離與識別：{folder}')
        for name in ('audio_vocals.wav', 'audio_instruments.wav'):
            path = os.path.join(folder, name)
            if os.path.isfile(path):
                os.remove(path)
        chunk_dir = os.path.join(folder, '_demucs_chunks')
        if os.path.isdir(chunk_dir):
            import shutil
            shutil.rmtree(chunk_dir, ignore_errors=True)
        clear_asr_downstream(folder, keep_bible=True)

    logger.info(f'正在分離音訊: {folder}')
    if not use_replicate(model_name):
        raise RuntimeError('本機 Demucs 已移除。請在 API 設定填入 REPLICATE_API_TOKEN。')
    result = separate_with_replicate(
        audio_path, vocal_output_path, instruments_output_path,
        model_name=model_name, shifts=shifts)
    if result and result[0]:
        save_dub_meta(folder, demucs_shifts=wanted)
    return result


def extract_audio_from_video(folder: str) -> bool:
    video_path = os.path.join(folder, 'download.mp4')
    if not os.path.exists(video_path):
        return False
    audio_path = os.path.join(folder, 'audio.wav')
    if os.path.exists(audio_path):
        logger.info(f'音訊已提取: {folder}')
        return True
    logger.info(f'正在從影片提取音訊: {folder}')
    os.system(
        f'ffmpeg -loglevel error -i "{video_path}" -vn -acodec pcm_s16le -ar 44100 -ac 2 "{audio_path}"')
    time.sleep(1)
    logger.info(f'音訊提取完成: {folder}')
    return True


def separate_all_audio_under_folder(root_folder: str, model_name: str = "htdemucs_ft",
                                    progress: bool = True, shifts: int = 5, device: str = 'auto') -> None:
    vocal_output_path, instruments_output_path = None, None
    for subdir, dirs, files in os.walk(root_folder):
        dirs[:] = [name for name in dirs if name not in {'translations', 'wavs', 'SPEAKER', '__pycache__', '_demucs_chunks'}]
        if 'download.mp4' not in files:
            continue
        if 'audio.wav' not in files:
            extract_audio_from_video(subdir)
        vocal_output_path, instruments_output_path = separate_audio(
            subdir, model_name, device, progress, shifts)
    logger.info(f'已完成所有音訊分離: {root_folder}')
    return f'所有音訊分離完成: {root_folder}', vocal_output_path, instruments_output_path
