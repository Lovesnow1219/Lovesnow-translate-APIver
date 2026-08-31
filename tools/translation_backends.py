# -*- coding: utf-8 -*-
"""Dispatch a translation request to the configured backend."""
from loguru import logger

from tools.step031_translation_openai import openai_response
from tools.step032_translation_llm import llm_response
from tools.step033_translation_translator import translator_response
from tools.step034_translation_ernie import ernie_response
from tools.step035_translation_qwen import qwen_response
from tools.step036_translation_ollama import ollama_response
from tools.target_language import translation_language


def llm_translate(method, messages):
    if method == 'LLM':
        return llm_response(messages)
    if method == 'OpenAI':
        return openai_response(messages)
    if method == 'Ernie':
        return ernie_response(messages[1:], system=messages[0]['content'])
    if method == '阿里云-通义千问':
        return qwen_response(messages)
    if method == 'Ollama':
        return ollama_response(messages)
    raise Exception('Invalid method')


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
