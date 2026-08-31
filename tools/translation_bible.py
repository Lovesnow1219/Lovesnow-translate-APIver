# -*- coding: utf-8 -*-
"""Episode outline / glossary used while translating."""
import json
import os
import re
import time

import traceback

from loguru import logger

from tools.step033_translation_translator import translator_response
from tools.target_language import needs_translation_refresh, translation_language
from tools.translation_backends import llm_translate
from tools.translation_quality import HAN_RE

def get_necessary_info(info: dict):
    return {
        'title': info['title'],
        'uploader': info['uploader'],
        'description': info['description'],
        'upload_date': info['upload_date'],
        # 'categories': info['categories'],
        'tags': info['tags'],
    }

def ensure_transcript_length(transcript, max_length=4000):
    mid = len(transcript)//2
    before, after = transcript[:mid], transcript[mid:]
    length = max_length//2
    return before[:length] + after[-length:]
def as_bible_text(value):
    if value is None:
        return ''
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        return '; '.join(f'{key}={item}' for key, item in value.items() if item)
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                src = item.get('src') or item.get('zh') or item.get('source') or ''
                dst = item.get('en') or item.get('name') or item.get('to') or item.get('dst') or ''
                parts.append(f'{src}={dst}' if src and dst else str(item))
            else:
                parts.append(str(item))
        return '; '.join(part for part in parts if part)
    return str(value).strip()


_GENERIC_GLOSSARY_ZH = {
    '篝火', '火堆', '火', '金币', '金幣', '钱', '錢', '林场', '林場',
    '森林', '木头', '木頭', '木材', '营地', '營地', '领营', '領營',
    '血量', '生命', '经验', '經驗', '背包', '药水', '藥水', '金币',
}
_GENERIC_GLOSSARY_EN = {
    'bonfire', 'campfire', 'fire', 'gold', 'coin', 'coins', 'money',
    'forest', 'woods', 'wood', 'camp', 'hp', 'mp', 'exp', 'potion',
}


def _split_glossary_pair(part):
    for sep in ('=', '：', ':'):
        if sep in part:
            src, dst = part.split(sep, 1)
            return src.strip(), dst.strip()
    return part.strip(), ''


def _filter_glossary_names(glossary):
    if not glossary:
        return glossary
    kept = []
    for part in re.split(r'[;\n]+', glossary):
        part = part.strip()
        if not part:
            continue
        src, dst = _split_glossary_pair(part)
        src_n = src.replace(' ', '')
        dst_n = (dst or src).strip().lower()
        if src_n in _GENERIC_GLOSSARY_ZH or dst_n in _GENERIC_GLOSSARY_EN:
            continue
        kept.append(f'{src}={dst}' if src and dst else part)
    return '; '.join(kept)


def bible_context(summary):
    summary = summary or {}
    parts = []
    title = (summary.get('title') or '').strip()
    plot = (summary.get('summary') or '').strip()
    if title:
        parts.append(f'Title: {title}')
    if plot:
        parts.append(f'Plot: {plot}')
    outline = as_bible_text(summary.get('outline'))
    if outline:
        parts.append(f'Outline:\n{outline}')
    glossary = _filter_glossary_names(as_bible_text(summary.get('glossary')))
    if glossary:
        parts.append(f'Names (use these spellings): {glossary}')
    voices = as_bible_text(summary.get('voices'))
    if voices:
        parts.append(f'Voices / narration / inner monologue: {voices}')
    return '\n'.join(parts)


def _extract_json_object(text):
    text = (text or '').strip()
    if text.startswith('```'):
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
    start = text.find('{')
    end = text.rfind('}')
    if start < 0 or end <= start:
        raise ValueError('no json object')
    return json.loads(text[start:end + 1])


OUTLINE_VERSION = 3


def _summary_has_bible(summary, target_language):
    if not isinstance(summary, dict):
        return False
    if summary.get('outline_locked') and (summary.get('outline') or summary.get('glossary')):
        return True
    try:
        version_ok = int(summary.get('outline_version') or 0) >= OUTLINE_VERSION
    except (TypeError, ValueError):
        version_ok = False
    lang_ok = translation_language(summary.get('language') or '') == translation_language(target_language)
    return version_ok and lang_ok and bool(summary.get('outline') or summary.get('glossary'))


def _load_folder_info(folder):
    info_path = os.path.join(folder, 'download.info.json')
    if os.path.isfile(info_path):
        with open(info_path, 'r', encoding='utf-8') as handle:
            return get_necessary_info(json.load(handle))
    return {
        'title': os.path.basename(folder),
        'uploader': 'Unknown',
        'description': 'Unknown',
        'upload_date': 'Unknown',
        'tags': [],
    }


def _source_transcript(folder):
    for name in ('transcript.json', 'translation.json'):
        path = os.path.join(folder, name)
        if not os.path.isfile(path):
            continue
        with open(path, 'r', encoding='utf-8') as handle:
            data = json.load(handle)
        if isinstance(data, list) and data:
            return data
    return []


def _transcript_for_bible(transcript, max_chars=8000):
    lines = []
    seen = set()
    for i, line in enumerate(transcript or []):
        text = (line.get('text') or '').strip()
        if not text or text in seen:
            continue
        seen.add(text)
        speaker = line.get('speaker') or ''
        prefix = f'{speaker}: ' if speaker else ''
        lines.append(f'{i}. {prefix}{text}')
    blob = '\n'.join(lines)
    if len(blob) <= max_chars:
        return blob
    head = blob[: max_chars // 2]
    tail = blob[-max_chars // 2:]
    return f'{head}\n...\n{tail}'


def _glossary_example(lang):
    if lang == 'Japanese':
        return (
            'Chinese source=spoken Japanese name, e.g. '
            '威拉/薇拉/维拉=ウィラ; 现实世界=リアルワールド; 战争游戏=ウォーゲーム. '
            'Do not put English or Chinese on the right side.\n'
        )
    if lang == 'Vietnamese':
        return (
            'Chinese source=spoken Vietnamese or a fixed Latin name, e.g. '
            '威拉/薇拉/维拉=Willa. No Chinese on the right side.\n'
        )
    return (
        'Chinese source=spoken English name, e.g. '
        '良心小贩=Honest Peddler; 灵合=Linghe; 祈天柱=Kitenchu; 薇拉=Willa. '
    )


def build_dubbing_bible(info, transcript, target_language, method='OpenAI'):
    lang = translation_language(target_language)
    body = _transcript_for_bible(transcript)
    if len(re.sub(r'\s+', '', body)) < 80:
        logger.warning('對白太少，不寫配音大綱')
        return None
    user = (
        f'Title: "{info.get("title")}" Author: "{info.get("uploader")}".\n'
        f'Dialogue (the only source of truth):\n{body}\n\n'
        f'Write a dubbing bible in {lang} as JSON only:\n'
        '{"title":"", "summary":"", "outline":"", "glossary":"", "voices":""}\n'
        'Use ONLY events, items, and names that appear in the dialogue. '
        'Do not invent a different plot from the title.\n'
        'summary: 2-5 sentences of what actually happens.\n'
        'outline: 6-12 short beats in dialogue order.\n'
        'glossary: ONLY people, places, invented skills, and unique named items. '
        'Do NOT list generic objects such as 篝火/金幣/林場/fire/gold/camp. '
        f'{_glossary_example(lang)}'
        'Keep those spellings forever. No Chinese on the right side.\n'
        'voices: narrator, system UI, inner monologue/self-talk, and main speakers, plus tone. '
        'Inner monologue and narration may run a bit long.'
    )
    messages = [
        {'role': 'system', 'content': (
            f'You prepare a {lang} dubbing bible for an anime/game episode. '
            'Keep names consistent and speakable. Output JSON only.'
        )},
        {'role': 'user', 'content': user},
    ]
    last_error = ''
    for _retry in range(2):
        try:
            response = llm_translate(method, messages)
            data = _extract_json_object(response)
            title = as_bible_text(data.get('title')) or (info.get('title') or '')
            plot = as_bible_text(data.get('summary'))
            outline = as_bible_text(data.get('outline'))
            raw_glossary = as_bible_text(data.get('glossary'))
            glossary = _filter_glossary_names(raw_glossary)
            voices = as_bible_text(data.get('voices'))
            if not plot or not outline:
                raise ValueError('bible missing outline')
            if lang != 'Japanese' and glossary and not HAN_RE.search(glossary):
                logger.warning('詞彙表沒有中文原文，仍沿用大綱繼續翻譯')
            logger.info(f'已生成配音大綱：{title[:60]}')
            return {
                'title': title,
                'author': info.get('uploader') or 'Unknown',
                'summary': plot,
                'outline': outline,
                'glossary': glossary,
                'voices': voices,
                'tags': info.get('tags') or [],
                'language': lang,
                'outline_version': OUTLINE_VERSION,
            }
        except Exception as exc:
            last_error = str(exc)
            logger.warning(f'配音大綱失敗：{exc}')
            time.sleep(1)
    logger.error(f'配音大綱多次失敗：{last_error}')
    return None


def ensure_episode_bible(folder, info, transcript, target_language, method='OpenAI'):
    summary_path = os.path.join(folder, 'summary.json')
    summary = {}
    if os.path.isfile(summary_path):
        try:
            with open(summary_path, 'r', encoding='utf-8') as handle:
                loaded = json.load(handle)
            if isinstance(loaded, dict):
                summary = loaded
        except Exception:
            summary = {}
    if _summary_has_bible(summary, target_language):
        return summary
    try:
        already_tried = int(summary.get('outline_attempted') or 0) >= OUTLINE_VERSION
    except (TypeError, ValueError):
        already_tried = False
    if already_tried and (summary.get('outline') or summary.get('summary')):
        logger.info(f'配音大綱已試過，沿用現有摘要：{folder}')
        return summary
    if needs_translation_refresh(target_language) and method not in ('Google Translate', 'Bing Translate'):
        bible = build_dubbing_bible(info, transcript, target_language, method)
        if bible:
            summary.update(bible)
            with open(summary_path, 'w', encoding='utf-8') as handle:
                json.dump(summary, handle, indent=2, ensure_ascii=False)
            return summary
        summary['outline_attempted'] = OUTLINE_VERSION
        os.makedirs(folder, exist_ok=True)
        with open(summary_path, 'w', encoding='utf-8') as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)
    if summary.get('title') and summary.get('summary'):
        return summary
    built = summarize(info, transcript, target_language, method)
    if built:
        with open(summary_path, 'w', encoding='utf-8') as handle:
            json.dump(built, handle, indent=2, ensure_ascii=False)
    return built



def summarize(info, transcript, target_language='简体中文', method='LLM'):
    transcript = ' '.join(line['text'] for line in transcript)
    transcript = ensure_transcript_length(transcript, max_length=2000)
    info_message = f'Title: "{info["title"]}" Author: "{info["uploader"]}". ' 
    
    if method in ['Google Translate', 'Bing Translate']:
        full_description = f'{info_message}\n{transcript}\n{info_message}\n'
        translation = translator_response(full_description, target_language)
        return {
                'title': translator_response(info['title'], target_language),
                'author': info['uploader'],
                'summary': translation,
                'language': target_language
            }

    full_description = f'The following is the full content of the video:\n{info_message}\n{transcript}\n{info_message}\nAccording to the above content, detailedly Summarize the video in JSON format:\n```json\n{{"title": "", "summary": ""}}\n```'
    
    messages = [
        {'role': 'system',
            'content': f'You are a expert in the field of this video. Please detailedly summarize the video in JSON format.\n```json\n{{"title": "the title of the video", "summary", "the summary of the video"}}\n```'},
        {'role': 'user', 'content': full_description},
    ]
    retry_message=''
    success = False
    for retry in range(9):
        try:
            messages = [
                {'role': 'system', 'content': f'You are a expert in the field of this video. Please summarize the video in JSON format.\n```json\n{{"title": "the title of the video", "summary", "the summary of the video"}}\n```'},
                {'role': 'user', 'content': full_description+retry_message},
            ]
            response = llm_translate(method, messages)
            summary = response.replace('\n', '')
            if '视频标题' in summary:
                raise Exception("包含“视频标题”")
            logger.info(summary)
            summary = re.findall(r'\{.*?\}', summary)[0]
            summary = json.loads(summary)
            summary = {
                'title': summary['title'].replace('title:', '').strip(),
                'summary': summary['summary'].replace('summary:', '').strip()
            }
            if summary['title'] == '' or summary['summary'] == '':
                raise Exception('Invalid summary')
            
            if 'title' in summary['title']:
                raise Exception('Invalid summary')
            success = True
            break
        except Exception as e:
            traceback.print_exc()
            err = str(e)
            if any(token in err for token in ('FreeTierOnly', 'AllocationQuota', 'free quota')):
                logger.error('通義千問額度用盡，停止重試')
                break
            retry_message += '\nSummarize the video in JSON format:\n```json\n{"title": "", "summary": ""}\n```'
            logger.warning(f'总结失败\n{e}')
            time.sleep(1)
            
    if not success:
        raise Exception(f'总结失败')
            
    messages = [
        {'role': 'system',
            'content': f'You are a native speaker of {target_language}. Please translate the title and summary into {target_language} in JSON format. ```json\n{{"title": "the {target_language} title of the video", "summary", "the {target_language} summary of the video", "tags": [list of tags in {target_language}]}}\n```.'},
        {'role': 'user',
            'content': f'The title of the video is "{summary["title"]}". The summary of the video is "{summary["summary"]}". Tags: {info["tags"]}.\nPlease translate the above title and summary and tags into {target_language} in JSON format. ```json\n{{"title": "", "summary", ""， "tags": []}}\n```. Remember to tranlate the title and the summary and tags into {target_language} in JSON.'},
    ]
    while True:
        try: 
            logger.info(summary)
            if target_language in summary['title'] or target_language in summary['summary']:
                raise Exception('Invalid translation')
            title = summary['title'].strip()
            if (title.startswith('"') and title.endswith('"')) or (title.startswith('“') and title.endswith('”')) or (title.startswith('‘') and title.endswith('’')) or (title.startswith("'") and title.endswith("'")) or (title.startswith('《') and title.endswith('》')):
                title = title[1:-1]
            result = {
                'title': title,
                'author': info['uploader'],
                'summary': summary['summary'],
                'tags': info['tags'],
                'language': target_language
            }
            return result
        except Exception as e:
            logger.warning(f'总结翻译失败\n{e}')
            time.sleep(1)

