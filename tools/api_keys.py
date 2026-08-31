# -*- coding: utf-8 -*-
"""Read one or more API keys from env. Comma-separated or KEY / KEY_2 / KEY_3."""
import os
import re
import threading

from loguru import logger

_COUNTERS = {}
_DEAD = {}
_LOCK = threading.Lock()

_QUOTA_MARKERS = (
    'insufficient_quota',
    'credit_balance_exhausted',
    'exceeded your current quota',
    'you have no credits',
    'billing_hard_limit_reached',
    'account_deactivated',
    'billing_not_active',
)


def split_api_keys(raw):
    if not raw:
        return []
    keys = []
    for part in re.split(r'[\s,;]+', str(raw).strip()):
        part = part.strip().strip('"').strip("'")
        if not part or part.startswith('sk-***') or part.lower() in {'none', 'null', 'undefined'}:
            continue
        keys.append(part)
    return keys


def collect_api_keys(*env_names):
    names = []
    for name in env_names:
        names.append(name)
        if not name.endswith(('_2', '_3')):
            names.extend((f'{name}_2', f'{name}_3'))
    keys = []
    seen = set()
    for name in names:
        for key in split_api_keys(os.getenv(name)):
            if key in seen:
                continue
            seen.add(key)
            keys.append(key)
    return keys


def require_api_keys(*env_names, error=''):
    keys = collect_api_keys(*env_names)
    if not keys:
        raise ValueError(error or f'請先在 .env 設定 {env_names[0]}')
    return keys


def _key_tag(key):
    text = str(key or '')
    return f'...{text[-4:]}' if len(text) >= 4 else '?'


def is_quota_error(exc):
    code = getattr(exc, 'code', None)
    if isinstance(code, str) and code.lower() in {
        'insufficient_quota',
        'credit_balance_exhausted',
        'billing_hard_limit_reached',
        'account_deactivated',
        'billing_not_active',
    }:
        return True
    body = getattr(exc, 'body', None)
    if isinstance(body, dict):
        nested = str(body.get('code') or body.get('type') or '').lower()
        if nested in {
            'insufficient_quota',
            'credit_balance_exhausted',
            'billing_hard_limit_reached',
        }:
            return True
        error = body.get('error')
        if isinstance(error, dict):
            nested = str(error.get('code') or error.get('type') or '').lower()
            if nested in {
                'insufficient_quota',
                'credit_balance_exhausted',
                'billing_hard_limit_reached',
            }:
                return True
    text = str(exc or '').lower()
    return any(marker in text for marker in _QUOTA_MARKERS)


def mark_api_key_dead(key, reason='quota'):
    if not key:
        return
    with _LOCK:
        if key in _DEAD:
            return
        _DEAD[key] = reason or 'quota'
    logger.warning(f'API key {_key_tag(key)} 已停用（{reason}），之後改用其他 key')


def live_api_keys(*env_names):
    keys = collect_api_keys(*env_names)
    with _LOCK:
        live = [key for key in keys if key not in _DEAD]
    return live


def next_api_key(*env_names, error=''):
    keys = require_api_keys(*env_names, error=error)
    with _LOCK:
        live = [key for key in keys if key not in _DEAD]
        pool = live or keys
        slot = env_names[0]
        index = _COUNTERS.get(slot, 0)
        _COUNTERS[slot] = index + 1
    position = index % len(pool)
    return pool[position], len(pool), position + 1


def workers_for_keys(env_name, *key_envs, default_per_key=4, absolute_max=16):
    live = live_api_keys(*key_envs)
    n_keys = max(1, len(live or collect_api_keys(*key_envs)))
    maximum = min(int(absolute_max), max(8, default_per_key * n_keys))
    raw = (os.getenv(env_name) or '').strip()
    if raw:
        try:
            value = int(raw)
        except (TypeError, ValueError):
            value = default_per_key * n_keys
        if n_keys > 1 and value == default_per_key:
            value = default_per_key * n_keys
    else:
        value = default_per_key * n_keys
    return max(1, min(maximum, value)), n_keys
