# -*- coding: utf-8 -*-
"""Dispatch a translation request to OpenAI, with machine-translate fallback."""
from loguru import logger

from tools.translation_openai import openai_response
from tools.translation_machine import translator_response
from tools.target_language import translation_language


def llm_translate(method, messages, reasoning_effort=None, timeout=None, model=None, purpose='translate'):
    if method in (None, '', 'OpenAI', 'LLM', 'Ernie', '阿里云-通义千问', 'Ollama'):
        return openai_response(
            messages, reasoning_effort=reasoning_effort, timeout=timeout, model=model,
            purpose=purpose,
        )
    raise Exception(f'Invalid method: {method}')


def emergency_translate(text, target_language, method, fixed_message):
    from tools.translation_quality import usable_target_text

    lang = translation_language(target_language)
    user = (
        f'Translate the whole line into spoken {lang}. '
        f'Do not leave any Chinese characters. Output only the {lang} line:"{text}"'
    )
    try:
        response = llm_translate(method, list(fixed_message) + [{'role': 'user', 'content': user}])
        usable = usable_target_text(response, target_language)
        if usable:
            logger.warning(f'緊急重翻成功：{usable[:80]}')
            return usable
    except Exception as exc:
        logger.warning(f'緊急重翻失敗：{exc}')
    try:
        raw = translator_response(text, to_language=target_language, translator_server='bing')
        usable = usable_target_text(raw, target_language)
        if usable:
            logger.warning(f'改用機器翻譯以免留下中文：{usable[:80]}')
            return usable
    except Exception as exc:
        logger.warning(f'機器翻譯後援失敗：{exc}')
    return ''
