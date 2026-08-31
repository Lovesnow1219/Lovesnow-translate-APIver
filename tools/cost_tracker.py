# -*- coding: utf-8 -*-
"""Estimate per-video API cost. Actual invoices may differ."""
import json
import os
import subprocess
import threading
import time
from contextvars import ContextVar
from datetime import datetime, timezone

from loguru import logger

_SESSION = ContextVar('linly_cost_session', default=None)
_ACTIVE_LOCK = threading.Lock()
_ACTIVE_SESSION = None
_LAST_MARKDOWNS = []

PIPELINE_STEPS = [
    ('prepare', '準備影片'),
    ('demucs', '人聲分離'),
    ('asr', '語音識別'),
    ('translate', '字幕翻譯'),
    ('tts', '語音合成'),
    ('mux', '影片合成'),
]

KIND_TO_STEP = {
    'demucs_replicate': 'demucs',
    'asr': 'asr',
    'asr_qwen': 'asr',
    'translate': 'translate',
    'llm': 'translate',
    'chat': 'translate',
    'tts_fish': 'tts',
    'tts_openai': 'tts',
}

_STAGE_HINTS = (
    ('prepare', ('準備處理', '準備', '初始化', '下載影片', '下載', '本地', '取得影片')),
    ('demucs', ('人聲分離',)),
    ('asr', ('語音識別', '智慧語音', 'ASR')),
    ('translate', ('字幕翻譯', '翻譯')),
    ('tts', ('語音合成', 'TTS')),
    ('mux', ('影片合成',)),
    ('done', ('處理完成', '處理成功')),
)

# USD per 1M tokens unless noted. Official rates checked 2026-08-27:
# https://developers.openai.com/api/docs/pricing
# gpt-4o-transcribe-diarize: same $2.50 / $10.00 as gpt-4o-transcribe
#   https://developers.openai.com/api/docs/models/gpt-4o-transcribe-diarize
# gpt-5.6-sol $4 / $20 is OpenAI's listed Standard short-context (promo at least through 2026-11-21)
_MODEL_TOKEN_RATES = {
    'gpt-5.6-luna': (0.20, 1.20),
    'gpt-5.6-terra': (2.00, 12.00),
    'gpt-5.6-sol': (4.00, 20.00),
    'gpt-5.6': (4.00, 20.00),
    'gpt-4o-mini': (0.15, 0.60),
    'gpt-4.1': (2.00, 8.00),
    'gpt-4o-transcribe-diarize': (2.50, 10.00),
    'gpt-4o-transcribe': (2.50, 10.00),
    'gpt-4o-mini-transcribe': (1.25, 5.00),
    'gpt-4o': (2.50, 10.00),
    'qwen-plus': (0.40, 1.20),
    'qwen-turbo': (0.15, 0.50),
}


def _env_float(name, default):
    try:
        return float(os.getenv(name) or default)
    except (TypeError, ValueError):
        return default


def usd_twd_rate():
    return _env_float('USD_TWD_RATE', 32.0)


def token_rates(model):
    name = (model or '').strip().lower()
    default_in = _env_float('OPENAI_INPUT_USD_PER_1M', 2.50)
    default_out = _env_float('OPENAI_OUTPUT_USD_PER_1M', 10.00)
    for key, pair in sorted(_MODEL_TOKEN_RATES.items(), key=lambda item: -len(item[0])):
        if name == key or name.startswith(key):
            return pair
    return default_in, default_out


def media_seconds(path):
    if not path or not os.path.exists(path):
        return 0.0
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'json', path],
            capture_output=True, text=True, check=False,
        )
        data = json.loads(result.stdout or '{}')
        return max(0.0, float((data.get('format') or {}).get('duration') or 0))
    except Exception:
        return 0.0


def _usage_dict(response):
    usage = getattr(response, 'usage', None)
    if usage is None:
        return {}
    if hasattr(usage, 'model_dump'):
        try:
            data = usage.model_dump() or {}
        except Exception:
            data = {}
    elif isinstance(usage, dict):
        data = usage
    else:
        data = {
            'input_tokens': getattr(usage, 'input_tokens', None) or getattr(usage, 'prompt_tokens', None),
            'output_tokens': getattr(usage, 'output_tokens', None) or getattr(usage, 'completion_tokens', None),
            'total_tokens': getattr(usage, 'total_tokens', None),
            'seconds': getattr(usage, 'seconds', None),
        }
    details = data.get('input_token_details') or data.get('input_tokens_details') or {}
    audio_tokens = 0
    if isinstance(details, dict):
        audio_tokens = int(details.get('audio_tokens') or 0)
    return {
        'input_tokens': int(data.get('input_tokens') or data.get('prompt_tokens') or 0),
        'output_tokens': int(data.get('output_tokens') or data.get('completion_tokens') or 0),
        'audio_tokens': audio_tokens,
        'seconds': data.get('seconds'),
    }


def match_stage_id(name):
    text = name or ''
    for stage_id, hints in _STAGE_HINTS:
        if any(h in text for h in hints):
            return stage_id
    return None


def group_entries(entries):
    grouped = {}
    for entry in entries or []:
        key = (entry.get('provider') or '-', entry.get('kind') or '-', entry.get('model') or '-')
        item = grouped.setdefault(key, {'usd': 0.0, 'count': 0})
        item['usd'] += float(entry.get('usd') or 0)
        item['count'] += 1
    return grouped


def format_duration(seconds):
    try:
        total = max(0, int(round(float(seconds or 0))))
    except (TypeError, ValueError):
        total = 0
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f'{hours}小時{minutes:02d}分{secs:02d}秒'
    if minutes:
        return f'{minutes}分{secs:02d}秒'
    return f'{secs}秒'


class CostSession:
    def __init__(self, folder=None):
        self.folder = folder
        self.entries = []
        self.percent = 0
        self.current_label = '準備中'
        self.stage_index = -1
        self.progress_cb = None
        self._last_notify = 0.0
        self._lock = threading.RLock()
        self.started_at = time.monotonic()
        self.step_status = {key: 'pending' for key, _ in PIPELINE_STEPS}
        self.step_started = {key: None for key, _ in PIPELINE_STEPS}
        self.step_seconds = {key: 0.0 for key, _ in PIPELINE_STEPS}
        self._tick_stop = threading.Event()
        self._ticker = threading.Thread(target=self._run_ticker, daemon=True, name='cost-ticker')
        self._ticker.start()

    def _run_ticker(self):
        while not self._tick_stop.wait(1.0):
            if any(status == 'running' for status in self.step_status.values()):
                self.notify(force=True)

    def stop_ticker(self):
        self._tick_stop.set()

    def elapsed_seconds(self):
        return max(0.0, time.monotonic() - self.started_at)

    def step_elapsed(self, step_id):
        with self._lock:
            total = float(self.step_seconds.get(step_id) or 0)
            started = self.step_started.get(step_id)
        if started is not None:
            total += max(0.0, time.monotonic() - started)
        return total

    def _open_step(self, key):
        if self.step_status.get(key) == 'running' and self.step_started.get(key) is not None:
            return
        self.step_started[key] = time.monotonic()

    def _close_step(self, key):
        started = self.step_started.get(key)
        if started is None:
            return
        self.step_seconds[key] = float(self.step_seconds.get(key) or 0) + max(0.0, time.monotonic() - started)
        self.step_started[key] = None

    def close_running_steps(self):
        with self._lock:
            for key in list(self.step_status):
                if self.step_status.get(key) == 'running':
                    self._close_step(key)
                    self.step_status[key] = 'done'

    def add(self, provider, kind, model='', **metrics):
        entry = {'provider': provider, 'kind': kind, 'model': model or ''}
        entry.update({k: v for k, v in metrics.items() if v is not None})
        entry['usd'] = round(_price_entry(entry), 6)
        with self._lock:
            self.entries.append(entry)
        logger.info(f"成本 {provider}/{kind} {model}: ${entry['usd']:.4f}")
        self.notify(force=False)

    def total_usd(self):
        with self._lock:
            return round(sum(float(e.get('usd') or 0) for e in self.entries), 6)

    def step_usd(self, step_id):
        total = 0.0
        with self._lock:
            for entry in self.entries:
                if KIND_TO_STEP.get(entry.get('kind')) == step_id:
                    total += float(entry.get('usd') or 0)
        return round(total, 6)

    def apply_stage(self, name, percent=None):
        if name:
            self.current_label = name
        if percent is not None:
            try:
                self.percent = max(0, min(100, float(percent)))
            except (TypeError, ValueError):
                pass
        stage_id = match_stage_id(name)
        keys = [key for key, _ in PIPELINE_STEPS]
        with self._lock:
            if stage_id == 'done':
                for key in self.step_status:
                    if self.step_status[key] == 'running':
                        self._close_step(key)
                    if self.step_status[key] != 'done':
                        self.step_status[key] = 'done'
                self.percent = 100
            elif stage_id is None:
                pass
            else:
                idx = keys.index(stage_id)
                self.stage_index = idx
                for i, key in enumerate(keys):
                    if i < idx:
                        if self.step_status[key] == 'running':
                            self._close_step(key)
                        self.step_status[key] = 'done'
                    elif i == idx:
                        self._open_step(key)
                        self.step_status[key] = 'running'
                    elif self.step_status.get(key) == 'running':
                        self._close_step(key)
                        self.step_status[key] = 'pending'
        self.notify(force=True)

    def notify(self, force=False):
        if self.progress_cb is None:
            return
        now = time.monotonic()
        if not force and (now - self._last_notify) < 0.4:
            return
        self._last_notify = now
        try:
            self.progress_cb(self.percent, live_status_text(self.percent, self.current_label, session=self))
        except Exception:
            pass

    def snapshot(self):
        return {
            'folder': self.folder,
            'at': datetime.now(timezone.utc).isoformat(),
            'currency': 'USD',
            'total_usd': self.total_usd(),
            'entries': self.entries,
            'total_seconds': round(self.elapsed_seconds(), 3),
            'steps': [
                {
                    'id': key,
                    'name': label,
                    'usd': self.step_usd(key),
                    'status': self.step_status.get(key),
                    'seconds': round(self.step_elapsed(key), 3),
                }
                for key, label in PIPELINE_STEPS
            ],
        }


def begin_session(folder=None, progress_cb=None):
    existing = current_session()
    if existing is not None:
        if folder and not existing.folder:
            existing.folder = folder
        if progress_cb:
            existing.progress_cb = progress_cb
        _bind_session(existing)
        return existing
    session = CostSession(folder)
    session.progress_cb = progress_cb
    _bind_session(session)
    return session


def _bind_session(session):
    global _ACTIVE_SESSION
    _SESSION.set(session)
    with _ACTIVE_LOCK:
        _ACTIVE_SESSION = session


def current_session():
    return _SESSION.get() or _ACTIVE_SESSION


def mark_stage(name, percent=None):
    session = current_session()
    if session is None:
        return
    session.apply_stage(name, percent)


def record(provider, kind, model='', response=None, **metrics):
    session = current_session()
    if session is None:
        return
    if response is not None:
        metrics = {**_usage_dict(response), **metrics}
        model = model or getattr(response, 'model', None) or metrics.get('model') or ''
    session.add(provider, kind, model, **metrics)


def finish_session(folder=None):
    global _ACTIVE_SESSION
    session = current_session()
    if session is not None:
        session.close_running_steps()
        session.notify(force=True)
        session.stop_ticker()
    _SESSION.set(None)
    with _ACTIVE_LOCK:
        if _ACTIVE_SESSION is session:
            _ACTIVE_SESSION = None
    if session is None:
        return None
    if folder:
        session.folder = folder
    snap = session.snapshot()
    if session.folder:
        _merge_save(session.folder, snap)
    history = load_cost(session.folder) if session.folder else {'last_run': snap, 'total_usd': snap['total_usd']}
    md = format_snapshot(session.folder, history)
    _LAST_MARKDOWNS.append(md)
    return md


def last_cost_markdown():
    if not _LAST_MARKDOWNS:
        return '尚無成本資料。跑完一鍵配音後會顯示本片 API 估計費用。'
    return _LAST_MARKDOWNS[-1]


def clear_last_markdowns():
    _LAST_MARKDOWNS.clear()


def live_status_text(percent=None, stage=None, session=None):
    session = session or current_session()
    pct = session.percent if session and percent is None else (percent or 0)
    label = stage or (session.current_label if session else '準備中')
    total = session.total_usd() if session else 0.0
    elapsed = session.elapsed_seconds() if session else 0.0
    lines = [
        f'目前：{label}  {int(pct)}%',
        f'已過時間：{format_duration(elapsed)}',
        f'本片目前累計：{format_money(total)}',
        '',
        '各步驟成本與耗時',
    ]
    for key, name in PIPELINE_STEPS:
        status = session.step_status.get(key, 'pending') if session else 'pending'
        usd = session.step_usd(key) if session else 0.0
        mark = {'pending': '○', 'running': '▶', 'done': '✓'}.get(status, '○')
        extra = ''
        if status == 'running':
            extra = f'  {format_money(usd)}  {format_duration(session.step_elapsed(key))}  ← 進行中'
            if key == 'tts':
                n = sum(1 for e in (session.entries if session else []) if KIND_TO_STEP.get(e.get('kind')) == 'tts')
                if n:
                    extra += f'（已合成 {n} 段）'
        elif status == 'done':
            extra = f'  {format_money(usd)}  {format_duration(session.step_elapsed(key))}'
            if key in ('mux', 'prepare') and not usd:
                extra = f'  本機，無 API 費用  {format_duration(session.step_elapsed(key))}'
        elif usd:
            extra = f'  {format_money(usd)}'
        elif key == 'mux':
            extra = '  本機合成，無 API 費用'
        elif key == 'prepare':
            extra = '  本機／下載，無 API 費用'
        lines.append(f'{mark} {name}{extra}')
    lines.append('')
    lines.append(f'總耗時：{format_duration(elapsed)}')
    lines.append('')
    lines.append('本趟明細')
    grouped = group_entries(session.entries if session else [])
    if grouped:
        for (provider, kind, model), item in grouped.items():
            count = f' ×{item["count"]}' if item['count'] > 1 else ''
            lines.append(f'- {provider} / {kind} / {model}{count}：{format_money(item["usd"])}')
    else:
        lines.append('（尚無 API 花費，開始打 API 後會即時更新）')
    lines.append('')
    lines.append('實際以各平台帳單為準。')
    return '\n'.join(lines)


def live_cost_markdown():
    session = current_session()
    if session is None:
        return last_cost_markdown()
    lines = [
        '### 本片 API 成本（即時估計）',
        f'**目前累計：{format_money(session.total_usd())}**',
        f'已過時間：{format_duration(session.elapsed_seconds())}',
        '',
    ]
    for key, name in PIPELINE_STEPS:
        status = session.step_status.get(key, 'pending')
        usd = session.step_usd(key)
        mark = {'pending': '○', 'running': '▶', 'done': '✓'}.get(status, '○')
        timed = ''
        if status in ('running', 'done'):
            timed = f' · {format_duration(session.step_elapsed(key))}'
        lines.append(f'- {mark} {name}：{format_money(usd)}{timed}')
    grouped = group_entries(session.entries)
    if grouped:
        lines.append('')
        lines.append('明細：')
        for (provider, kind, model), item in grouped.items():
            count = f' ×{item["count"]}' if item['count'] > 1 else ''
            lines.append(f'- {provider} / {kind} / {model}{count}：{format_money(item["usd"])}')
    lines.append('')
    lines.append('實際以各平台帳單為準。')
    return '\n'.join(lines)


def _price_entry(entry):
    kind = entry.get('kind')
    model = (entry.get('model') or '').lower()
    if kind in ('translate', 'llm', 'chat'):
        inn, out = token_rates(model)
        return (entry.get('input_tokens') or 0) / 1_000_000 * inn + (entry.get('output_tokens') or 0) / 1_000_000 * out
    if kind == 'asr':
        inn, out = token_rates(model or 'gpt-4o-transcribe-diarize')
        token_cost = (entry.get('input_tokens') or 0) / 1_000_000 * inn + (entry.get('output_tokens') or 0) / 1_000_000 * out
        seconds = float(entry.get('seconds') or 0)
        per_min = _env_float(
            'OPENAI_ASR_USD_PER_MINUTE',
            0.0045 if 'gpt-transcribe' == (model or '').lower() else 0.006,
        )
        minute_cost = seconds / 60.0 * per_min
        return token_cost if token_cost > 0 else minute_cost
    if kind == 'tts_fish':
        if 'free' in model:
            return 0.0
        nbytes = int(entry.get('utf8_bytes') or 0)
        return nbytes / 1_000_000 * _env_float('FISH_USD_PER_1M_BYTES', 15.0)
    if kind == 'tts_openai':
        chars = int(entry.get('characters') or 0)
        return chars / 1_000_000 * _env_float('OPENAI_TTS_USD_PER_1M_CHARS', 15.0)
    if kind == 'demucs_replicate':
        seconds = float(entry.get('seconds') or 0)
        per_min = os.getenv('REPLICATE_DEMUCS_USD_PER_MINUTE')
        if per_min:
            try:
                return seconds / 60.0 * float(per_min)
            except (TypeError, ValueError):
                pass
        return seconds * _env_float('REPLICATE_T4_USD_PER_SECOND', 0.000225)
    if kind == 'asr_qwen':
        minutes = float(entry.get('seconds') or 0) / 60.0
        return minutes * _env_float('QWEN_ASR_USD_PER_MINUTE', 0.002)
    return 0.0


def _merge_save(folder, snap):
    path = os.path.join(folder, 'cost.json')
    history = {'folder': folder, 'currency': 'USD', 'runs': [], 'total_usd': 0.0}
    if os.path.exists(path):
        try:
            history = json.load(open(path, encoding='utf-8'))
        except Exception:
            pass
    runs = list(history.get('runs') or [])
    runs.append(snap)
    history['runs'] = runs
    history['last_run'] = snap
    history['total_usd'] = round(sum(float(r.get('total_usd') or 0) for r in runs), 6)
    history['updated_at'] = datetime.now(timezone.utc).isoformat()
    os.makedirs(folder, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    with open(os.path.join(folder, 'cost.txt'), 'w', encoding='utf-8') as f:
        f.write(format_snapshot(folder, history))


def load_cost(folder):
    path = os.path.join(folder or '', 'cost.json')
    if not os.path.exists(path):
        return None
    try:
        return json.load(open(path, encoding='utf-8'))
    except Exception:
        return None


def format_money(usd):
    usd = float(usd or 0)
    twd = usd * usd_twd_rate()
    return f'US${usd:.4f}（NT${twd:.1f}）'


def _short_folder(folder):
    text = str(folder or '').replace('\\', '/').strip('/')
    if not text:
        return ''
    parts = [p for p in text.split('/') if p]
    if len(parts) <= 2:
        return text
    return '/'.join(parts[-2:])


def format_snapshot(folder, history):
    if not history:
        return f'{folder or ""}：尚無成本紀錄'
    last = history.get('last_run') or {}
    folder_label = _short_folder(folder or history.get('folder') or '')
    lines = [
        f'### 本片 API 成本（估計）',
        f'資料夾：`{folder_label}`' if folder_label else '資料夾：-',
        f'**本片累計：{format_money(history.get("total_usd"))}**（{len(history.get("runs") or [])} 次處理）',
        f'最近一次：{format_money(last.get("total_usd"))}',
    ]
    if last.get('total_seconds') is not None:
        lines.append(f'最近一次總耗時：{format_duration(last.get("total_seconds"))}')
    lines.extend(['', '各步驟（最近一次）'])
    last_entries = last.get('entries') or []
    last_steps = last.get('steps')
    if last_steps:
        for step in last_steps:
            timed = f' · {format_duration(step.get("seconds"))}' if step.get('seconds') else ''
            lines.append(f'- {step.get("name")}：{format_money(step.get("usd"))}{timed}')
    else:
        by_step = {key: 0.0 for key, _ in PIPELINE_STEPS}
        for entry in last_entries:
            step_id = KIND_TO_STEP.get(entry.get('kind'))
            if step_id:
                by_step[step_id] += float(entry.get('usd') or 0)
        for key, name in PIPELINE_STEPS:
            lines.append(f'- {name}：{format_money(by_step.get(key))}')
    grouped = group_entries(last_entries)
    if grouped:
        lines.append('')
        lines.append('明細：')
        for (provider, kind, model), item in grouped.items():
            count = f' ×{item["count"]}' if item['count'] > 1 else ''
            lines.append(f'- {provider} / {kind} / {model}{count}：{format_money(item["usd"])}')
    lines.append('')
    lines.append(
        '費率（官方，2026-08-27）：ASR gpt-4o-transcribe-diarize $2.50 / $10 每百萬 token；'
        'Luna $0.20 / $1.20；Fish s2.1-pro $15 / 百萬 UTF-8 bytes（free $0）；'
        'Demucs 依 Replicate T4 GPU $0.000225 / 秒。實際以帳單為準。'
    )
    return '\n'.join(lines)


def costs_under_folder(root='videos'):
    parts = []
    root = root or 'videos'
    if not os.path.isdir(root):
        return '找不到資料夾。'
    for dirpath, _dirs, files in os.walk(root):
        if 'cost.json' in files:
            parts.append(format_snapshot(dirpath, load_cost(dirpath)))
    return '\n\n---\n\n'.join(parts) if parts else '這個資料夾還沒有成本紀錄。跑完一鍵配音後會寫入 `cost.json`。'
