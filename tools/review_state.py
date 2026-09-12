"""Keep failed review calls visible instead of treating them as zero findings."""
import os

from tools.dubbing_settings import read_json, write_json_atomic
from tools.target_language import translation_language, load_dub_meta

NAME = 'review_failures.json'


def failures(folder, language=None):
    if not folder:
        return {}
    language = translation_language(language or load_dub_meta(folder).get('translation') or 'English')
    result = dict(read_json(os.path.join(folder, NAME), {}).get(language, {}))
    from tools.source_subtitles import SETTINGS, REPORT
    if read_json(os.path.join(folder, SETTINGS), {}).get('enabled'):
        report = read_json(os.path.join(folder, REPORT), {})
        if report.get('errors'):
            result['原片字幕對照'] = 'Incomplete'
    return result


def begin_stage(folder, language, stage):
    if not folder:
        return
    path = os.path.join(folder, NAME)
    data = read_json(path, {})
    key = translation_language(language)
    data.setdefault(key, {}).pop(stage, None)
    if os.path.exists(path):
        write_json_atomic(path, data)


def failed(folder, language, stage, exc):
    from tools.job_control import JobStopped
    if isinstance(exc, JobStopped):
        raise exc
    if not folder:
        return
    path = os.path.join(folder, NAME)
    data = read_json(path, {})
    # Exception messages may contain provider URLs or request data.
    data.setdefault(translation_language(language), {})[stage] = type(exc).__name__
    write_json_atomic(path, data)


def warning(folder, language=None):
    missing = failures(folder, language)
    return ('審核未完成：' + '、'.join(missing) + '。成片可預覽，請重新執行未完成階段。') if missing else ''
