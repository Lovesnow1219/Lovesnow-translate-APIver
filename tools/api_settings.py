# -*- coding: utf-8 -*-
"""Read/write API keys in the local .env file for the WebUI settings tab."""
import os
import re
from pathlib import Path

from dotenv import load_dotenv

_MASK_RE = re.compile(r'^(sk-\*+|•+)')

ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT / '.env'

# (env_key, label, secret, placeholder, help)
REQUIRED_FIELDS = [
    ('OPENAI_API_KEY', 'OpenAI API Key', True, 'sk-…', '語音識別，以及翻譯。沒有右邊 Luna key 時，翻譯也用這把。'),
    ('FISH_API_KEY', 'Fish Audio API Key', True, 'sk-…', '配音。須為 Fish 帳號金鑰。'),
    ('REPLICATE_API_TOKEN', 'Replicate API Token', True, 'r8_…', '雲端人聲分離 Demucs。https://replicate.com/account/api-tokens'),
]
EXTRA_FIELDS = [
    ('OPENAI_LUNA_API_KEY', 'OpenAI Luna Key', True, 'sk-…', '第二把 OpenAI，只給 Luna 翻譯加速，不拿去 ASR。'),
    ('FISH_API_KEY_2', 'Fish API Key 2', True, 'sk-…', '第二個 Fish 帳號，用來加總配音併發。'),
]
GROUPS = [
    ('必要金鑰', REQUIRED_FIELDS),
    ('額外加速（可選）', EXTRA_FIELDS),
]
REQUIRED_KEYS = [key for key, *_ in REQUIRED_FIELDS]
EXTRA_KEYS = [key for key, *_ in EXTRA_FIELDS]
ALL_KEYS = REQUIRED_KEYS + EXTRA_KEYS


def env_path():
    return ENV_PATH


def _parse_env_file(path):
    data = {}
    if not path.exists():
        return data
    for raw in path.read_text(encoding='utf-8').splitlines():
        line = raw.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, _, value = line.partition('=')
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
            value = value[1:-1]
        data[key] = value
    return data


def _format_env_value(value):
    text = str(value)
    if any(ch in text for ch in ' \t#\'"'):
        return '"' + text.replace('\\', '\\\\').replace('"', '\\"') + '"'
    return text


def stored_secrets():
    stored = _parse_env_file(ENV_PATH)
    load_dotenv(ENV_PATH, override=True)
    values = []
    for key in ALL_KEYS:
        values.append((os.getenv(key) or stored.get(key) or '').strip())
    return values


def field_values():
    return stored_secrets()


def key_status():
    rows = []
    for key, value in zip(ALL_KEYS, stored_secrets()):
        rows.append((key, bool(value)))
    return rows


def _status_table(keys, present_map):
    lines = [
        '| 變數 | 狀態 |',
        '| --- | --- |',
    ]
    for key in keys:
        lines.append(f'| `{key}` | {"已設定" if present_map.get(key) else "未設定"} |')
    return '\n'.join(lines)


def status_markdown():
    present_map = dict(key_status())
    return (
        '金鑰只存在本機 `.env`，**不會上傳、也不會進 git**。密碼框留空 = 保留現有值。\n\n'
        + _status_table(REQUIRED_KEYS, present_map)
    )


def extra_status_markdown():
    present_map = dict(key_status())
    return (
        '第二組帳號，用來**額外加速**翻譯或配音；不填不影響主流程。\n\n'
        + _status_table(EXTRA_KEYS, present_map)
    )


def upsert_env(updates):
    updates = {k: v.strip() for k, v in updates.items() if v is not None and str(v).strip()}
    if not updates:
        return 0
    aliases = {
        'REPLICATE_API_TOKEN': 'REPLICATE_API_KEY',
    }
    for src, dest in aliases.items():
        if src in updates and dest not in updates:
            updates[dest] = updates[src]
    original = ENV_PATH.read_text(encoding='utf-8').splitlines() if ENV_PATH.exists() else []
    seen = set()
    new_lines = []
    for line in original:
        stripped = line.strip()
        if not stripped or stripped.startswith('#') or '=' not in stripped:
            new_lines.append(line)
            continue
        key = stripped.split('=', 1)[0].strip()
        if key in updates:
            new_lines.append(f'{key}={_format_env_value(updates[key])}')
            seen.add(key)
        else:
            new_lines.append(line)
    missing = [key for key in updates if key not in seen]
    if missing:
        if new_lines and new_lines[-1].strip():
            new_lines.append('')
        for key in missing:
            new_lines.append(f'{key}={_format_env_value(updates[key])}')
    ENV_PATH.write_text('\n'.join(new_lines) + '\n', encoding='utf-8')
    for key, value in updates.items():
        os.environ[key] = value
    load_dotenv(ENV_PATH, override=True)
    return len(updates)


def _keep_existing(text):
    text = (text or '').strip()
    return (not text) or bool(_MASK_RE.match(text))


def save_from_ui(*values):
    current = stored_secrets()
    updates = {}
    for key, value, old in zip(ALL_KEYS, values, current):
        if value is None or _keep_existing(str(value)):
            continue
        text = str(value).strip()
        if text and text != old:
            updates[key] = text
    count = upsert_env(updates)
    fields = field_values()
    if count == 0:
        return (
            '沒有新內容可寫入。密碼框留空會保留現有金鑰。',
            status_markdown(),
            extra_status_markdown(),
            *fields,
        )
    return (
        f'已寫入 {count} 項到 {ENV_PATH.name}，立即生效。',
        status_markdown(),
        extra_status_markdown(),
        *fields,
    )


def load_from_ui():
    return status_markdown(), extra_status_markdown(), *field_values()
