# -*- coding: utf-8 -*-
import os
import re
import shutil
import subprocess
import tempfile

from dotenv import load_dotenv
from loguru import logger
from openai import OpenAI

load_dotenv()

LOCAL_WHISPER_SIZES = {'large', 'medium', 'small', 'base', 'tiny'}
DIARIZE_MODELS = {'gpt-4o-transcribe-diarize'}
JSON_ONLY_MODELS = {'gpt-4o-transcribe', 'gpt-4o-mini-transcribe', 'gpt-4o-mini-transcribe-2025-12-15'}
ASR_PROMPT = (
    'Only transcribe spoken words. Do not summarize, omit, clean up, or truncate. '
    'Output every spoken word.'
)


def _get_client():
    from tools.api_keys import next_api_key
    api_key, n_keys, index = next_api_key(
        'OPENAI_API_KEY', error='請先在 .env 設定有效的 OPENAI_API_KEY'
    )
    base_url = os.getenv('OPENAI_API_BASE') or 'https://api.openai.com/v1'
    if n_keys > 1:
        logger.info(f'OpenAI ASR 使用 key {index}/{n_keys}')
    return OpenAI(base_url=base_url, api_key=api_key), api_key


def _prepare_audio(wav_path):
    size_mb = os.path.getsize(wav_path) / (1024 * 1024)
    if size_mb <= 24:
        return wav_path, None
    tmp = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
    tmp.close()
    logger.info(f'音訊 {size_mb:.1f}MB 超過 API 限制，轉成 mp3: {tmp.name}')
    subprocess.run(
        ['ffmpeg', '-y', '-i', wav_path, '-ac', '1', '-ar', '16000', '-b:a', '64k', tmp.name],
        check=True,
        capture_output=True,
    )
    return tmp.name, tmp.name


def _resolve_asr_model(model_name):
    if model_name and model_name not in LOCAL_WHISPER_SIZES:
        return model_name
    return os.getenv('OPENAI_ASR_MODEL') or 'gpt-4o-transcribe-diarize'


def _speaker_id(raw):
    speaker = str(raw or '').strip()
    if not speaker:
        return 'SPEAKER_00'
    if speaker.upper().startswith('SPEAKER'):
        return speaker.replace(' ', '_')
    return f'SPEAKER_{speaker}'


def _item_value(item, key, default=None):
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def _segments_from_result(result):
    transcript = []
    for segment in getattr(result, 'segments', None) or []:
        text = (_item_value(segment, 'text') or '').strip()
        if not text:
            continue
        start = float(_item_value(segment, 'start', 0) or 0)
        end = float(_item_value(segment, 'end', start) or start)
        transcript.append({
            'start': start,
            'end': end,
            'text': text,
            'speaker': _speaker_id(_item_value(segment, 'speaker')),
        })
    return transcript


def _timed_sentences(text, duration):
    parts = [p.strip() for p in re.split(r'(?<=[。！？.!?])\s*', (text or '').strip()) if p.strip()]
    if not parts:
        return []
    duration = float(duration or 0)
    total_chars = sum(len(p) for p in parts) or 1
    cursor = 0.0
    transcript = []
    for index, part in enumerate(parts):
        span = duration * (len(part) / total_chars) if duration else 0.0
        end = duration if index == len(parts) - 1 else cursor + span
        transcript.append({
            'start': cursor,
            'end': end,
            'text': part,
            'speaker': 'SPEAKER_00',
        })
        cursor = end
    return transcript


def _transcribe_file(wav_path, asr_model):
    from tools.api_keys import is_quota_error, mark_api_key_dead

    audio_path, tmp_path = _prepare_audio(wav_path)
    timeout = float(os.getenv('OPENAI_TIMEOUT') or 600)
    result = None
    last_exc = None
    try:
        logger.info(f'OpenAI ASR 辨識: {wav_path} model={asr_model}')
        for _attempt in range(3):
            client, api_key = _get_client()
            try:
                with open(audio_path, 'rb') as audio_file:
                    create_kwargs = {
                        'model': asr_model,
                        'file': audio_file,
                        'timeout': timeout,
                    }
                    if asr_model in DIARIZE_MODELS or asr_model.endswith('-diarize'):
                        create_kwargs['response_format'] = 'diarized_json'
                        create_kwargs['chunking_strategy'] = 'auto'
                    elif asr_model in JSON_ONLY_MODELS:
                        create_kwargs['response_format'] = 'json'
                        create_kwargs['prompt'] = ASR_PROMPT
                    elif asr_model == 'gpt-transcribe':
                        create_kwargs['response_format'] = 'json'
                        create_kwargs['prompt'] = ASR_PROMPT
                    else:
                        create_kwargs['response_format'] = 'verbose_json'
                        create_kwargs['timestamp_granularities'] = ['segment']
                    result = client.audio.transcriptions.create(**create_kwargs)
                from tools.cost_tracker import record
                record('openai', 'asr', asr_model, response=result, seconds=getattr(result, 'duration', None))
                last_exc = None
                break
            except Exception as exc:
                last_exc = exc
                if is_quota_error(exc):
                    mark_api_key_dead(api_key, 'quota')
                    logger.warning('OpenAI ASR key 額度用盡，改用下一把')
                    continue
                raise
        if last_exc is not None:
            raise last_exc
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

    transcript = _segments_from_result(result)
    if not transcript:
        full_text = (getattr(result, 'text', None) or '').strip()
        duration = getattr(result, 'duration', None)
        if duration is None:
            usage = getattr(result, 'usage', None)
            duration = getattr(usage, 'seconds', None)
        transcript = _timed_sentences(full_text, duration)
        if not transcript and full_text:
            transcript.append({
                'start': 0.0,
                'end': float(duration or 0),
                'text': full_text,
                'speaker': 'SPEAKER_00',
            })
    logger.info(f'OpenAI ASR 完成，共 {len(transcript)} 段 model={asr_model}')
    return transcript


def openai_transcribe_audio(wav_path, model_name: str = 'gpt-4o-transcribe-diarize', **kwargs):
    from tools.audio_chunks import cut_audio, keep_chunk_segment, media_duration, plan_chunks

    asr_model = _resolve_asr_model(model_name)
    hop = float(os.getenv('OPENAI_ASR_MAX_SECONDS') or 1400)
    overlap = float(os.getenv('OPENAI_ASR_OVERLAP_SECONDS') or 1.0)
    duration = media_duration(wav_path)
    chunks = plan_chunks(duration, hop, overlap)
    if len(chunks) <= 1:
        return _transcribe_file(wav_path, asr_model)

    logger.info(f'音訊 {duration:.0f}s 超過 OpenAI ASR 約 25 分鐘上限，分成 {len(chunks)} 段')
    tmpdir = tempfile.mkdtemp(prefix='asr_chunks_')
    transcript = []
    try:
        for index, (start, end) in enumerate(chunks):
            piece = os.path.join(tmpdir, f'chunk_{index:03d}.wav')
            cut_audio(wav_path, piece, start, end, sample_rate=16000, channels=1)
            segs = _transcribe_file(piece, asr_model)
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
    transcript.sort(key=lambda item: item.get('start') or 0)
    logger.info(f'OpenAI ASR 分段完成，共 {len(transcript)} 段 model={asr_model}')
    return transcript
