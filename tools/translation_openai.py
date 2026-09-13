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


def _chat_messages(messages):
    """Preserve image inputs when using the Chat Completions compatibility path."""
    result = []
    for message in messages:
        row = dict(message)
        if isinstance(row.get('content'), list):
            blocks = []
            for part in row['content']:
                if part.get('type') == 'input_text':
                    blocks.append({'type': 'text', 'text': part['text']})
                elif part.get('type') == 'input_image':
                    blocks.append({'type': 'image_url', 'image_url': {
                        'url': part['image_url'], 'detail': part.get('detail', 'auto')}})
                else:
                    blocks.append(part)
            row['content'] = blocks
        result.append(row)
    return result


def _call_openai(api_key, n_keys, index, messages, reasoning_effort=None, timeout=None, model=None, purpose='translate'):
    from tools.api_keys import is_quota_error, openai_key_label
    from tools.cost_tracker import record

    model_name = (model or os.getenv('MODEL_NAME') or 'gpt-5.6-luna').strip()
    base_url = os.getenv('OPENAI_API_BASE') or 'https://api.openai.com/v1'
    effort = (
        reasoning_effort
        if reasoning_effort is not None
        else (os.getenv('OPENAI_REASONING_EFFORT') or 'xhigh')
    ).strip()
    timeout = float(timeout if timeout is not None else (os.getenv('OPENAI_TIMEOUT') or 600))
    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
    )
    kind = 'review' if purpose == 'review' else 'translate'
    if purpose == 'review':
        logger.info('OpenAI 審稿使用主 key')
    elif n_keys > 1:
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
            # Even an empty / reasoning-only response may consume tokens.
            record('openai', kind, model_name, response=response,
                   credential=openai_key_label(api_key))
            text = _extract_output_text(response)
            if text:
                return text
            logger.warning('Responses API 沒有回文字，改走 Chat Completions')
        except Exception as exc:
            # A timeout can occur after generation was billed. Only fall back
            # when the endpoint explicitly reports protocol incompatibility.
            if is_quota_error(exc) or getattr(exc, 'status_code', None) not in (400, 404, 405, 422, 501):
                raise
            logger.warning(f'Responses API 失敗，改走 Chat Completions: {exc}')

    kwargs = {
        'model': model_name,
        'messages': _chat_messages(messages),
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
    record('openai', kind, model_name, response=response,
           credential=openai_key_label(api_key))
    return response.choices[0].message.content


def openai_response(messages, reasoning_effort=None, timeout=None, model=None, purpose='translate'):
    from tools.api_keys import (
        is_quota_error,
        mark_api_key_dead,
        next_api_key,
        openai_review_key_names,
        openai_translate_key_names,
    )
    from tools.job_control import check_stop

    review = purpose == 'review'
    names = openai_review_key_names() if review else openai_translate_key_names()
    error = (
        '審稿固定走主 key，請先在 .env 或網頁「API 設定」填 OPENAI_API_KEY'
        if review
        else '請先在 .env 或網頁「API 設定」填 OPENAI_API_KEY 或 OPENAI_TRANSLATE_API_KEY'
    )
    last_exc = None
    attempts = 1 if review else 4
    for _attempt in range(attempts):
        check_stop()
        api_key, n_keys, index = next_api_key(*names, error=error, expand=False)
        try:
            return _call_openai(
                api_key, n_keys, index, messages,
                reasoning_effort=reasoning_effort, timeout=timeout, model=model,
                purpose=purpose,
            )
        except Exception as exc:
            last_exc = exc
            if review:
                if is_quota_error(exc):
                    logger.warning('主 key 審稿額度用盡，不改打翻譯 key')
                raise
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
