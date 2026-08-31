# -*- coding: utf-8 -*-
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

import httpx
from dotenv import load_dotenv
from loguru import logger
from tools.target_language import is_asr_junk

load_dotenv()

QWEN_POLL_TIMEOUT_SECONDS = 1800.0
QWEN_UPLOAD_TIMEOUT_SECONDS = 300.0


def dashscope_api_key():
    return (os.getenv('DASHSCOPE_API_KEY') or os.getenv('QWEN_API_KEY') or '').strip()


def dashscope_base_url():
    override = (os.getenv('DASHSCOPE_BASE_URL') or '').strip().rstrip('/')
    if override:
        return override
    region = (os.getenv('DASHSCOPE_REGION') or '').strip().lower()
    if region in {'intl', 'international', 'singapore', 'sg'}:
        return 'https://dashscope-intl.aliyuncs.com'
    return 'https://dashscope.aliyuncs.com'


def qwen_asr_model():
    return os.getenv('QWEN_ASR_MODEL') or 'qwen-audio-3.0-asr-flash-filetrans'


def qwen_transcribe_audio(wav_path, diarization=True, language=None, **kwargs):
    api_key = dashscope_api_key()
    if not api_key:
        raise ValueError('請先在 .env 設定 DASHSCOPE_API_KEY')
    from tools.audio_chunks import cut_audio, keep_chunk_segment, media_duration, plan_chunks

    source = Path(wav_path)
    prepared = _prepare_mono16k(source)
    hop = float(os.getenv('QWEN_ASR_MAX_SECONDS') or 1200)
    overlap = float(os.getenv('QWEN_ASR_OVERLAP_SECONDS') or 1.0)
    duration = media_duration(str(prepared))
    chunks = plan_chunks(duration, hop, overlap)
    if len(chunks) <= 1:
        file_url = (os.getenv('DASHSCOPE_FILE_URL') or '').strip() or None
        return _qwen_from_prepared(prepared, diarization=diarization, language=language, file_url=file_url)

    logger.info(f'音訊 {duration:.0f}s 超過通義 ASR 輪詢時限，分成 {len(chunks)} 段')
    tmpdir = tempfile.mkdtemp(prefix='qwen_asr_')
    transcript = []
    try:
        for index, (start, end) in enumerate(chunks):
            piece = Path(tmpdir) / f'chunk_{index:03d}.wav'
            cut_audio(str(prepared), str(piece), start, end, sample_rate=16000, channels=1)
            segs = _qwen_from_prepared(piece, diarization=diarization, language=language, file_url='')
            is_first = index == 0
            is_last = index == len(chunks) - 1
            for seg in segs:
                abs_start = float(seg.get('start') or 0) + start
                abs_end = float(seg.get('end') or 0) + start
                if not keep_chunk_segment(abs_start, abs_end, start, end, overlap, is_first, is_last):
                    continue
                seg['start'] = round(abs_start, 3)
                seg['end'] = round(abs_end, 3)
                transcript.append(seg)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    if not transcript:
        raise RuntimeError('通義千問 ASR 沒有回傳語音段落')
    transcript.sort(key=lambda item: item.get('start') or 0)
    logger.info(f'通義千問 ASR 分段完成，共 {len(transcript)} 段')
    return transcript


def _qwen_from_prepared(prepared, *, diarization=True, language=None, file_url=None):
    api_key = dashscope_api_key()
    model = qwen_asr_model()
    prepared = Path(prepared)
    url = (file_url or '').strip() or upload_dashscope_file(prepared, api_key=api_key, model=model)
    logger.info(f'通義千問 ASR: {prepared} model={model} diarize={bool(diarization)}')
    result = submit_and_wait_filetrans(
        url,
        api_key=api_key,
        language=language,
        diarize=bool(diarization),
        model=model,
    )
    transcript = segments_from_qwen_result(result)
    if not transcript:
        raise RuntimeError('通義千問 ASR 沒有回傳語音段落')
    logger.info(f'通義千問 ASR 完成，共 {len(transcript)} 段')
    try:
        from tools.cost_tracker import media_seconds, record
        record('qwen', 'asr_qwen', model, seconds=media_seconds(str(prepared)))
    except Exception:
        pass
    return transcript


def upload_dashscope_file(path, *, api_key, model):
    policy = _get_upload_policy(api_key, model)
    file_name = f'{uuid.uuid4().hex}_{path.name}'
    key = f"{policy['upload_dir'].rstrip('/')}/{file_name}"
    logger.info(f'上傳音訊到 DashScope: {path.name}')
    with path.open('rb') as handle:
        files = {
            'OSSAccessKeyId': (None, str(policy['oss_access_key_id'])),
            'Signature': (None, str(policy['signature'])),
            'policy': (None, str(policy['policy'])),
            'x-oss-object-acl': (None, str(policy['x_oss_object_acl'])),
            'x-oss-forbid-overwrite': (None, str(policy['x_oss_forbid_overwrite'])),
            'key': (None, key),
            'success_action_status': (None, '200'),
            'file': (file_name, handle, 'application/octet-stream'),
        }
        with httpx.Client(timeout=QWEN_UPLOAD_TIMEOUT_SECONDS, follow_redirects=True) as client:
            response = client.post(str(policy['upload_host']), files=files)
    if response.status_code != 200:
        raise RuntimeError(f'DashScope 上傳失敗 ({response.status_code}): {response.text[:300]}')
    return f'oss://{key}'


def submit_and_wait_filetrans(file_url, *, api_key, language=None, diarize=True, model=None):
    model = model or qwen_asr_model()
    task_id = _submit_filetrans(
        file_url, api_key=api_key, language=language, diarize=diarize, model=model
    )
    payload = _poll_task(task_id, api_key=api_key)
    result_url = _transcription_url(payload)
    if not result_url:
        raise RuntimeError('通義千問 ASR 成功但沒有 transcription_url')
    return _download_json(result_url)


def segments_from_qwen_result(result):
    transcript = []
    for item in _iter_sentences(result):
        text = str(item.get('text') or '').strip()
        if not text or is_asr_junk(text):
            continue
        start = _ms_to_seconds(item.get('begin_time'))
        end = _ms_to_seconds(item.get('end_time'))
        if end <= start or end - start < 0.12:
            continue
        speaker_id = item.get('speaker_id')
        if speaker_id in (None, '', '@', 'unknown', 'noise'):
            speaker = 'SPEAKER_00'
        else:
            try:
                speaker = f'SPEAKER_{int(speaker_id):02d}'
            except (TypeError, ValueError):
                speaker = 'SPEAKER_00'
        transcript.append({
            'start': round(start, 3),
            'end': round(end, 3),
            'text': text,
            'speaker': speaker,
        })
    return transcript


def _prepare_mono16k(path):
    dest = path.with_name(f'{path.stem}_qwen16k_mono.wav')
    if dest.is_file() and dest.stat().st_mtime >= path.stat().st_mtime:
        return dest
    subprocess.run(
        ['ffmpeg', '-y', '-i', str(path), '-ar', '16000', '-ac', '1', '-c:a', 'pcm_s16le', str(dest)],
        check=True,
        capture_output=True,
    )
    return dest


def _auth_headers(api_key):
    return {'Authorization': f'Bearer {api_key}'}


def _get_upload_policy(api_key, model):
    url = f'{dashscope_base_url()}/api/v1/uploads'
    with httpx.Client(timeout=30.0) as client:
        response = client.get(
            url,
            headers=_auth_headers(api_key),
            params={'action': 'getPolicy', 'model': model},
        )
    if response.status_code != 200:
        raise RuntimeError(f'DashScope getPolicy 失敗 ({response.status_code}): {response.text[:300]}')
    data = response.json().get('data') or {}
    required = (
        'upload_dir',
        'upload_host',
        'oss_access_key_id',
        'signature',
        'policy',
        'x_oss_object_acl',
        'x_oss_forbid_overwrite',
    )
    missing = [key for key in required if not data.get(key)]
    if missing:
        raise RuntimeError(f'DashScope 上傳策略缺少欄位: {", ".join(missing)}')
    return data


def _submit_filetrans(file_url, *, api_key, language, diarize, model):
    url = f'{dashscope_base_url()}/api/v1/services/audio/asr/transcription'
    headers = {
        **_auth_headers(api_key),
        'Content-Type': 'application/json',
        'X-DashScope-Async': 'enable',
    }
    if file_url.startswith('oss://'):
        headers['X-DashScope-OssResourceResolve'] = 'enable'
    parameters = {
        'channel_id': [0],
        'diarization_enabled': bool(diarize),
    }
    hints = _language_hints(language)
    if hints:
        parameters['language_hints'] = hints
    body = {
        'model': model,
        'input': {'file_urls': [file_url]},
        'parameters': parameters,
    }
    with httpx.Client(timeout=60.0) as client:
        response = client.post(url, headers=headers, json=body)
    if response.status_code >= 400:
        raise RuntimeError(f'通義千問 ASR 送出失敗 ({response.status_code}): {response.text[:300]}')
    payload = response.json()
    task_id = (payload.get('output') or {}).get('task_id') or payload.get('task_id')
    if not task_id:
        raise RuntimeError('通義千問 ASR 沒有回傳 task_id')
    return str(task_id)


def _poll_task(task_id, *, api_key):
    url = f'{dashscope_base_url()}/api/v1/tasks/{task_id}'
    deadline = time.monotonic() + QWEN_POLL_TIMEOUT_SECONDS
    delay = 5.0
    while time.monotonic() < deadline:
        with httpx.Client(timeout=30.0) as client:
            response = client.get(url, headers=_auth_headers(api_key))
        if response.status_code >= 400:
            raise RuntimeError(f'通義千問 ASR 輪詢失敗 ({response.status_code}): {response.text[:300]}')
        payload = response.json()
        status = str((payload.get('output') or {}).get('task_status') or payload.get('task_status') or '')
        if status in {'SUCCEEDED', 'FAILED', 'UNKNOWN'}:
            if status != 'SUCCEEDED':
                raise RuntimeError(_task_error_message(payload, status))
            _assert_subtask_ok(payload)
            return payload
        time.sleep(delay)
        delay = min(15.0, delay + 2.0)
    raise RuntimeError(f'通義千問 ASR 逾時 {int(QWEN_POLL_TIMEOUT_SECONDS)} 秒')


def _transcription_url(payload):
    output = payload.get('output') or {}
    results = output.get('results')
    if isinstance(results, list) and results:
        first = results[0] or {}
        return str(first.get('transcription_url') or '').strip()
    result = output.get('result') or payload.get('result') or {}
    if isinstance(result, dict):
        return str(result.get('transcription_url') or '').strip()
    return ''


def _assert_subtask_ok(payload):
    output = payload.get('output') or {}
    results = output.get('results')
    if not isinstance(results, list):
        return
    for item in results:
        status = str((item or {}).get('subtask_status') or '').upper()
        if status and status != 'SUCCEEDED':
            message = (item or {}).get('message') or (item or {}).get('code') or status
            raise RuntimeError(f'通義千問 ASR 子任務失敗: {message}')


def _task_error_message(payload, status):
    output = payload.get('output') or {}
    results = output.get('results') or []
    if isinstance(results, list) and results:
        first = results[0] or {}
        detail = first.get('message') or first.get('code')
        if detail:
            return f'通義千問 ASR {status}: {detail}'
    return f'通義千問 ASR 狀態 {status}'


def _download_json(url):
    with httpx.Client(timeout=120.0, follow_redirects=True) as client:
        response = client.get(url)
    if response.status_code >= 400:
        raise RuntimeError(f'下載通義千問轉寫失敗 ({response.status_code})')
    data = response.json()
    if not isinstance(data, dict):
        raise RuntimeError('通義千問轉寫 JSON 格式不正確')
    return data


def _iter_sentences(result):
    sentences = []
    transcripts = result.get('transcripts') or []
    if isinstance(transcripts, list):
        for transcript in transcripts:
            items = (transcript or {}).get('sentences') or []
            if isinstance(items, list):
                sentences.extend(item for item in items if isinstance(item, dict))
    if not sentences and isinstance(result.get('sentences'), list):
        sentences = [item for item in result['sentences'] if isinstance(item, dict)]
    return sentences


def _ms_to_seconds(value):
    if value in (None, ''):
        return 0.0
    try:
        return float(value) / 1000.0
    except (TypeError, ValueError):
        return 0.0


def _language_hints(language):
    if not language or str(language).lower() in {'auto', 'detect'}:
        return ['zh', 'en']
    mapped = {
        'zh-hant': 'zh',
        'zh-hans': 'zh',
        'zh-tw': 'zh',
        'zh-cn': 'zh',
        'zh': 'zh',
        'en': 'en',
        'ja': 'ja',
        'ko': 'ko',
    }
    code = mapped.get(str(language).lower(), str(language).lower().split('-')[0])
    return [code]
