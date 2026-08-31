# -*- coding: utf-8 -*-
import os

from dotenv import load_dotenv
from loguru import logger
from openai import OpenAI

load_dotenv()

OPENAI_VOICES = ['alloy', 'ash', 'coral', 'echo', 'fable', 'nova', 'onyx', 'sage', 'shimmer']


def _get_client():
    from tools.api_keys import next_api_key
    api_key, n_keys, index = next_api_key(
        'OPENAI_API_KEY', error='請先在 .env 設定有效的 OPENAI_API_KEY'
    )
    base_url = os.getenv('OPENAI_API_BASE') or 'https://api.openai.com/v1'
    if n_keys > 1:
        logger.info(f'OpenAI TTS 使用 key {index}/{n_keys}')
    return OpenAI(base_url=base_url, api_key=api_key)


def tts(text, output_path, target_language='中文', voice='alloy', speed=1.0):
    if os.path.exists(output_path):
        logger.info(f'TTS {text} 已存在')
        return
    if voice not in OPENAI_VOICES:
        voice = os.getenv('OPENAI_TTS_VOICE', 'alloy')
    model_name = os.getenv('OPENAI_TTS_MODEL', 'tts-1')
    client = _get_client()
    for retry in range(3):
        try:
            os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
            with client.audio.speech.with_streaming_response.create(
                model=model_name,
                voice=voice,
                input=text,
                response_format='wav',
                speed=float(min(4.0, max(0.25, speed or 1.0))),
            ) as response:
                response.stream_to_file(output_path)
            logger.info(f'OpenAI TTS {text}')
            from tools.cost_tracker import record
            record('openai', 'tts_openai', model_name, characters=len(text or ''))
            return
        except Exception as e:
            logger.warning(f'OpenAI TTS 失敗 ({retry + 1}/3): {e}')
    raise RuntimeError(f'OpenAI TTS 失敗: {text}')
