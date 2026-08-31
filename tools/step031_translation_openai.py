# -*- coding: utf-8 -*-
import os
from openai import OpenAI
from dotenv import load_dotenv
from loguru import logger

load_dotenv()


def _extract_output_text(response):
    text = (getattr(response, 'output_text', None) or '').strip()
    if text:
        return text
    for item in getattr(response, 'output', None) or []:
        for part in getattr(item, 'content', None) or []:
            part_text = getattr(part, 'text', None)
            if part_text:
                return str(part_text).strip()
    return ''


def _call_openai(api_key, n_keys, index, messages):
    from tools.api_keys import is_quota_error

    model_name = os.getenv('MODEL_NAME') or 'gpt-5.6-luna'
    base_url = os.getenv('OPENAI_API_BASE') or 'https://api.openai.com/v1'
    effort = (os.getenv('OPENAI_REASONING_EFFORT') or 'xhigh').strip()
    timeout = float(os.getenv('OPENAI_TIMEOUT') or 600)
    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
    )
    if n_keys > 1:
        logger.info(f'OpenAI 翻譯使用 key {index}/{n_keys}')

    use_responses = os.getenv('OPENAI_USE_RESPONSES', '1').strip() not in ('0', 'false', 'False')
    if use_responses:
        kwargs = {
            'model': model_name,
            'input': messages,
            'timeout': timeout,
        }
        if effort and effort.lower() not in ('none', 'off'):
            kwargs['reasoning'] = {'effort': effort}
        try:
            response = client.responses.create(**kwargs)
            text = _extract_output_text(response)
            if text:
                from tools.cost_tracker import record
                record('openai', 'translate', model_name, response=response)
                return text
            logger.warning('Responses API 沒有回文字，改走 Chat Completions')
        except Exception as exc:
            if is_quota_error(exc):
                raise
            logger.warning(f'Responses API 失敗，改走 Chat Completions: {exc}')

    kwargs = {
        'model': model_name,
        'messages': messages,
        'timeout': timeout,
    }
    extra_body = {}
    if effort and effort.lower() not in ('none', 'off'):
        extra_body['reasoning_effort'] = effort
    if os.getenv('OPENAI_REPETITION_PENALTY'):
        extra_body['repetition_penalty'] = float(os.getenv('OPENAI_REPETITION_PENALTY'))
    if extra_body:
        kwargs['extra_body'] = extra_body
    response = client.chat.completions.create(**kwargs)
    from tools.cost_tracker import record
    record('openai', 'translate', model_name, response=response)
    return response.choices[0].message.content


def openai_response(messages):
    from tools.api_keys import is_quota_error, mark_api_key_dead, next_api_key
    from tools.job_control import check_stop

    last_exc = None
    for _attempt in range(4):
        check_stop()
        api_key, n_keys, index = next_api_key(
            'OPENAI_LUNA_API_KEY',
            'OPENAI_API_KEY',
            error='請先在 .env 或網頁「API 設定」填 OPENAI_API_KEY 或 OPENAI_LUNA_API_KEY',
        )
        try:
            return _call_openai(api_key, n_keys, index, messages)
        except Exception as exc:
            last_exc = exc
            if is_quota_error(exc):
                mark_api_key_dead(api_key, 'quota')
                logger.warning(f'OpenAI key {index} 額度用盡，改用下一把')
                continue
            raise
    raise last_exc


if __name__ == '__main__':
    test_message = [{"role": "user", "content": "你好,介绍一下你自己"}]
    response = openai_response(test_message)
    print(response)
