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
    ('OPENAI_API_KEY', 'OpenAI API Key', True, 'sk-…', '語音識別、審稿（固定這把），以及翻譯。沒有右邊翻譯 key 時，翻譯也用這把。'),
    ('FISH_API_KEY', 'Fish Audio API Key', True, 'sk-…', '配音。須為 Fish 帳號金鑰。'),
    ('REPLICATE_API_TOKEN', 'Replicate API Token', True, 'r8_…', '雲端人聲分離 Demucs。https://replicate.com/account/api-tokens'),
]
EXTRA_FIELDS = [
    ('OPENAI_TRANSLATE_API_KEY', 'OpenAI API Key 2', True, 'sk-…', '第二把 OpenAI，只給翻譯加速，不拿去 ASR。有這把時翻譯併發 4+3=7。'),
    ('OPENAI_TRANSLATE_API_KEY_2', 'OpenAI API Key 3', True, 'sk-…', '第三把 OpenAI，只給翻譯加速。三把滿是 4+3+3=10 併發。'),
    ('FISH_API_KEY_2', 'Fish API Key 2', True, 'sk-…', '第二個 Fish 帳號，用來加總配音併發。'),
]
TRANSLATION_MODELS = ['gpt-5.6-luna', 'gpt-5.6-terra', 'gpt-5.6-sol']
REASONING_EFFORTS = ['none', 'low', 'medium', 'high', 'xhigh', 'max']
MODEL_DEFAULTS = {
    'MODEL_NAME': 'gpt-5.6-luna',
    'OPENAI_REASONING_EFFORT': 'xhigh',
    'REVIEW_MODEL_NAME': 'gpt-5.6-sol',
    'REVIEW_REASONING_EFFORT': 'high',
}
MODEL_FIELD_CHOICES = {
    'MODEL_NAME': TRANSLATION_MODELS,
    'OPENAI_REASONING_EFFORT': REASONING_EFFORTS,
    'REVIEW_MODEL_NAME': TRANSLATION_MODELS,
    'REVIEW_REASONING_EFFORT': REASONING_EFFORTS,
}
MODEL_FIELDS = [
    ('MODEL_NAME', '翻譯模型', False, 'gpt-5.6-luna', '配音翻譯。量大用 luna；要穩用 terra；最嚴用 sol。'),
    ('OPENAI_REASONING_EFFORT', '翻譯推理', False, 'xhigh', '翻譯建議 xhigh。sol 開 max 會更慢更貴。'),
    ('REVIEW_MODEL_NAME', '審稿模型', False, 'gpt-5.6-sol', '旗艦審稿用 sol。固定走主 key。'),
    ('REVIEW_REASONING_EFFORT', '審稿推理', False, 'high', 'sol 用 high。整表審稿；不要為了省時降到 medium。'),
]
GROUPS = [
    ('必要金鑰', REQUIRED_FIELDS),
    ('額外加速（可選）', EXTRA_FIELDS),
    ('模型（翻譯／審稿）', MODEL_FIELDS),
]
REQUIRED_KEYS = [key for key, *_ in REQUIRED_FIELDS]
EXTRA_KEYS = [key for key, *_ in EXTRA_FIELDS]
MODEL_KEYS = [key for key, *_ in MODEL_FIELDS]
ALL_KEYS = REQUIRED_KEYS + EXTRA_KEYS + MODEL_KEYS


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


_LEGACY_ENV = {
    'OPENAI_TRANSLATE_API_KEY': 'OPENAI_LUNA_API_KEY',
}


def migrate_legacy_env_names():
    if not ENV_PATH.exists():
        return
    text = ENV_PATH.read_text(encoding='utf-8')
    if 'OPENAI_TRANSLATE_API_KEY=' in text or 'OPENAI_LUNA_API_KEY=' not in text:
        return
    ENV_PATH.write_text(
        text.replace('OPENAI_LUNA_API_KEY=', 'OPENAI_TRANSLATE_API_KEY='),
        encoding='utf-8',
    )
    load_dotenv(ENV_PATH, override=True)


def stored_secrets():
    migrate_legacy_env_names()
    stored = _parse_env_file(ENV_PATH)
    load_dotenv(ENV_PATH, override=True)
    values = []
    for key in ALL_KEYS:
        value = (os.getenv(key) or stored.get(key) or '').strip()
        legacy = _LEGACY_ENV.get(key)
        if not value and legacy:
            value = (os.getenv(legacy) or stored.get(legacy) or '').strip()
        values.append(value)
    return values


def coerce_choice(value, choices, default):
    text = (value or '').strip()
    return text if text in choices else default


def current_model_value(key):
    choices = MODEL_FIELD_CHOICES[key]
    default = MODEL_DEFAULTS[key]
    return coerce_choice(os.getenv(key) or '', choices, default)


def apply_model_choices(translation_model=None, translation_effort=None, review_model=None, review_effort=None):
    updates = {}
    if translation_model:
        updates['MODEL_NAME'] = coerce_choice(translation_model, TRANSLATION_MODELS, MODEL_DEFAULTS['MODEL_NAME'])
    if translation_effort:
        updates['OPENAI_REASONING_EFFORT'] = coerce_choice(
            translation_effort, REASONING_EFFORTS, MODEL_DEFAULTS['OPENAI_REASONING_EFFORT']
        )
    if review_model:
        updates['REVIEW_MODEL_NAME'] = coerce_choice(review_model, TRANSLATION_MODELS, MODEL_DEFAULTS['REVIEW_MODEL_NAME'])
    if review_effort:
        updates['REVIEW_REASONING_EFFORT'] = coerce_choice(
            review_effort, REASONING_EFFORTS, MODEL_DEFAULTS['REVIEW_REASONING_EFFORT']
        )
    updates = {
        key: value
        for key, value in updates.items()
        if value != (os.getenv(key) or '').strip()
    }
    if updates:
        upsert_env(updates)
    return updates


def field_values():
    values = list(stored_secrets())
    index = {key: i for i, key in enumerate(ALL_KEYS)}
    for key, choices in MODEL_FIELD_CHOICES.items():
        values[index[key]] = coerce_choice(values[index[key]], choices, MODEL_DEFAULTS[key])
    return values


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
        '第二組帳號，用來**額外加速**翻譯或配音；不填不影響主流程。'
        '翻譯併發依 OpenAI 金鑰數：1 把 4、2 把 7、3 把 10。\n\n'
        + _status_table(EXTRA_KEYS, present_map)
    )


def model_status_markdown():
    labels = {key: label for key, label, *_ in MODEL_FIELDS}
    lines = [
        '從清單選，不能手打。審稿建議用**另一檔**當第二雙眼睛。'
        '現役：`gpt-5.6-luna`（量大）／`gpt-5.6-terra`（穩）／`gpt-5.6-sol`（最嚴）。',
        '',
        '| 項目 | 目前 |',
        '| --- | --- |',
    ]
    for key in MODEL_KEYS:
        lines.append(f'| {labels[key]} | `{current_model_value(key)}` |')
    return '\n'.join(lines)


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
            model_status_markdown(),
            *fields,
        )
    return (
        f'已寫入 {count} 項到 {ENV_PATH.name}，立即生效。',
        status_markdown(),
        extra_status_markdown(),
        model_status_markdown(),
        *fields,
    )


def load_from_ui():
    return status_markdown(), extra_status_markdown(), model_status_markdown(), *field_values()
