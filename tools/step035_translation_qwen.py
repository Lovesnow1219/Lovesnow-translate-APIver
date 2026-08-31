# -*- coding: utf-8 -*-
import os
from openai import OpenAI
from dotenv import load_dotenv
from loguru import logger
from tools.step031_translation_openai import openai_response

load_dotenv()


def _is_quota_error(exc):
    text = str(exc)
    return any(token in text for token in (
        'FreeTierOnly',
        'AllocationQuota',
        'free quota has been exhausted',
        'use free tier only',
    ))


def qwen_response(messages):
    api_key = os.getenv('QWEN_API_KEY') or os.getenv('DASHSCOPE_API_KEY')
    if not api_key:
        raise ValueError('請先在 .env 設定 DASHSCOPE_API_KEY')
    region = (os.getenv('DASHSCOPE_REGION') or '').lower()
    default_base = (
        'https://dashscope-intl.aliyuncs.com/compatible-mode/v1'
        if region in {'intl', 'international', 'singapore', 'sg'}
        else 'https://dashscope.aliyuncs.com/compatible-mode/v1'
    )
    client = OpenAI(
        base_url=os.getenv('QWEN_API_BASE') or default_base,
        api_key=api_key,
    )
    model_name = os.getenv('QWEN_MODEL_ID', 'qwen-plus')
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            timeout=240,
        )
        from tools.cost_tracker import record
        record('qwen', 'translate', model_name, response=response)
        return response.choices[0].message.content
    except Exception as exc:
        if _is_quota_error(exc):
            logger.warning('通義千問免費額度已用完，自動改用 OpenAI GPT-5.6 Luna 翻譯')
            return openai_response(messages)
        raise

if __name__ == '__main__':
    test_message = [{"role": "user", "content": "你好，介绍一下你自己"}]
    response = qwen_response(test_message)
    print(response)
