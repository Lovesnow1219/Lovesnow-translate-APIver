# -*- coding: utf-8 -*-
"""One episode folder can keep several dubbed languages without overwriting."""
import json
import os
import shutil

from loguru import logger

from tools.target_language import load_dub_meta, save_dub_meta, translation_language

VERSION_DIR = 'translations'


def lang_key(language):
    return translation_language(language or 'English')


def versions_dir(folder):
    return os.path.join(folder, VERSION_DIR)


def version_path(folder, language):
    return os.path.join(versions_dir(folder), f'{lang_key(language)}.json')


def review_version_path(folder, language):
    return os.path.join(versions_dir(folder), f'{lang_key(language)}.review.json')


def _read_list(path):
    if not os.path.isfile(path):
        return None
    with open(path, 'r', encoding='utf-8') as handle:
        data = json.load(handle)
    return data if isinstance(data, list) else None


def _write_json(path, data, indent=2):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(data, handle, indent=indent, ensure_ascii=False)


def _same_lang(a, b):
    if not a or not b:
        return False
    return lang_key(a) == lang_key(b)


def has_any_version(folder):
    vdir = versions_dir(folder)
    if os.path.isdir(vdir):
        for name in os.listdir(vdir):
            if name.endswith('.json') and not name.endswith('.review.json'):
                return True
    return os.path.isfile(os.path.join(folder, 'translation.json'))


def has_version(folder, language):
    lang = lang_key(language)
    if os.path.isfile(version_path(folder, lang)):
        return True
    meta = load_dub_meta(folder)
    return (
        os.path.isfile(os.path.join(folder, 'translation.json'))
        and _same_lang(meta.get('translation'), lang)
    )


def list_saved_languages(folder):
    found = []
    vdir = versions_dir(folder)
    if os.path.isdir(vdir):
        for name in sorted(os.listdir(vdir)):
            if name.endswith('.json') and not name.endswith('.review.json'):
                found.append(name[:-5])
    meta = load_dub_meta(folder)
    active = meta.get('translation')
    if active and os.path.isfile(os.path.join(folder, 'translation.json')) and active not in found:
        found.append(active)
    return found


def snapshot_active(folder):
    """Copy the working translation.json into translations/{current}.json."""
    path = os.path.join(folder, 'translation.json')
    if not os.path.isfile(path):
        return None
    meta = load_dub_meta(folder)
    lang = meta.get('translation')
    if not lang:
        return None
    dest = version_path(folder, lang)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    created = not os.path.isfile(dest)
    shutil.copy2(path, dest)
    review_src = os.path.join(folder, 'translation_review.json')
    if os.path.isfile(review_src):
        try:
            with open(review_src, 'r', encoding='utf-8') as handle:
                report = json.load(handle)
        except Exception:
            report = {}
        if isinstance(report, dict) and (
            not report.get('language') or _same_lang(report.get('language'), lang)
        ):
            shutil.copy2(review_src, review_version_path(folder, lang))
    versions = dict(meta.get('versions') or {})
    if lang not in versions:
        try:
            versions[lang] = int(meta.get('translation_version') or 0)
        except (TypeError, ValueError):
            versions[lang] = 0
        save_dub_meta(folder, translation=lang, translation_version=versions[lang])
    if created:
        logger.info(f'已備份 {lang} 譯文：{dest}')
    return dest


def activate_language(folder, language):
    """Make this language the working translation.json. Returns lines or None."""
    lang = lang_key(language)
    src = version_path(folder, lang)
    dest = os.path.join(folder, 'translation.json')
    meta = load_dub_meta(folder)
    if not os.path.isfile(src):
        if os.path.isfile(dest) and _same_lang(meta.get('translation'), lang):
            return _read_list(dest)
        return None
    snapshot_active(folder)
    shutil.copy2(src, dest)
    review_src = review_version_path(folder, lang)
    review_dest = os.path.join(folder, 'translation_review.json')
    if os.path.isfile(review_src):
        shutil.copy2(review_src, review_dest)
    elif os.path.isfile(review_dest):
        try:
            with open(review_dest, 'r', encoding='utf-8') as handle:
                report = json.load(handle)
        except Exception:
            report = {}
        if isinstance(report, dict) and report.get('language') and not _same_lang(report.get('language'), lang):
            os.remove(review_dest)
    versions = dict(load_dub_meta(folder).get('versions') or {})
    try:
        ver = int(versions.get(lang, 0) or 0)
    except (TypeError, ValueError):
        ver = 0
    if ver <= 0 and _same_lang(meta.get('translation'), lang):
        try:
            ver = int(meta.get('translation_version') or 0)
        except (TypeError, ValueError):
            ver = 0
    save_dub_meta(folder, translation=lang, translation_version=ver)
    logger.info(f'已切到 {lang}：{folder}')
    return _read_list(dest)


def write_translation(folder, transcript, language, translation_version=None):
    lang = lang_key(language)
    snapshot_active(folder)
    _write_json(version_path(folder, lang), transcript)
    _write_json(os.path.join(folder, 'translation.json'), transcript)
    kwargs = {'translation': lang}
    if translation_version is None:
        versions = dict(load_dub_meta(folder).get('versions') or {})
        if lang not in versions:
            from tools.target_language import required_translation_version
            translation_version = required_translation_version(lang)
    if translation_version is not None:
        kwargs['translation_version'] = int(translation_version)
    save_dub_meta(folder, **kwargs)
    return lang


def write_review_report(folder, language, report):
    lang = lang_key(language)
    report = dict(report or {})
    report['language'] = lang
    _write_json(review_version_path(folder, lang), report)
    _write_json(os.path.join(folder, 'translation_review.json'), report)
    return report


def load_lines_for_language(folder, language):
    """Saved version, or active file if it matches, or source cards with empty dub."""
    lang = lang_key(language)
    data = _read_list(version_path(folder, lang))
    if data is not None:
        return data
    active = os.path.join(folder, 'translation.json')
    meta = load_dub_meta(folder)
    if os.path.isfile(active) and _same_lang(meta.get('translation'), lang):
        data = _read_list(active)
        if data is not None:
            return data
    for name in ('transcript.json', 'translation.json'):
        path = os.path.join(folder, name)
        data = _read_list(path)
        if data is None:
            continue
        if name == 'translation.json':
            out = []
            for line in data:
                row = dict(line)
                row['translation'] = ''
                out.append(row)
            return out
        return data
    return []
