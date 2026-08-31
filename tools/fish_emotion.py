# -*- coding: utf-8 -*-
"""Pick Fish Audio emotion tags from a dubbed line."""
import re

_TAG_RE = re.compile(r'\[([^\]]+)\]')
_LAUGH = ('哈哈', '呵呵', '嘿嘿', '嘻嘻', 'haha', 'hehe', 'lol')
_WHISPER = ('轻轻', '小声', '低声', '心想', '耳语', '悄悄')
_SIGH = ('唉', '啧')
_ANGRY = ('混蛋', '滚开', '快滚', '该死', '白痴', '走狗', '臭鱼', 'idiot', 'damn', 'lackey', 'shut up')
_SAD = ('损失惨重', 'miss you')
_SURPRISE = ('居然', '竟然', '不对', '哦对', '什么', 'wait', 'what?')
_EXCITED = ('终于', '复活', '哈哈')
_NERVOUS = ('擦了把', '额头的汗', '不敢当')
_RELIEF = ('松了口气', '还活得挺好')
_ANGRY_JA = ('クソ', '馬鹿', 'てめえ', 'お前', 'ふざける', '何する', '死ね')
_SURPRISE_JA = ('えっ', 'まじ', 'まさか', 'なに', '何だ', 'うそ', 'うそだ')
_LAUGH_JA = ('へへ', 'ふふ', 'わはは', 'あはは', 'うふふ')
_EXCITED_JA = ('やった', 'よっしゃ', 'いくぞ')
_WHISPER_JA = ('こっそり', 'ひそひそ', '小声で')
_SIGH_JA = ('はぁ', 'やれやれ', 'ちっ')
_ANGRY_VI = ('đồ khốn', 'chết tiệt', 'cút đi', 'đồ ngu', 'khốn nạn')
_SURPRISE_VI = ('trời ơi', 'cái gì', 'sao vậy')
_LAUGH_VI = ('ha ha', 'hi hi')
_EXCITED_VI = ('cuối cùng', 'được rồi')


def infer_emotion_tags(english, source='', source_is_shared=False):
    """Pick up to two Fish S2 bracket tags from this dubbed line."""
    src = source or ''
    en = (english or '').strip()
    zh = '' if source_is_shared else src
    blob = f'{zh} {en}'.lower()
    tags = []

    def add(tag):
        if tag not in tags:
            tags.append(tag)

    if zh.strip() in {'哼', '哼哼'} or en.lower().rstrip('.! ') in {'hmph', 'humph'}:
        return []
    if (
        any(m in en.lower() for m in ('haha', 'hehe', 'lol'))
        or any(m in zh for m in ('哈哈', '呵呵', '嘿嘿', '嘻嘻'))
        or any(m in en for m in _LAUGH_JA)
        or any(m in en.lower() for m in _LAUGH_VI)
    ):
        add('laughing')
    if any(m in zh for m in _WHISPER) or 'whisper' in en.lower() or any(m in en for m in _WHISPER_JA):
        add('whispering')
    if any(m in zh for m in _SIGH) or 'sigh' in en.lower() or any(m in en for m in _SIGH_JA):
        add('sighing')
    if any(m in blob for m in _ANGRY) or any(m in en for m in _ANGRY_JA) or any(m in en.lower() for m in _ANGRY_VI):
        add('angry')
    if any(m in zh for m in _RELIEF) or 'relieved' in en.lower():
        add('relieved')
    if any(m in zh for m in _NERVOUS) or 'wiped his brow' in en.lower() or 'wiped her brow' in en.lower():
        add('nervous')
    if 'laughing' not in tags and (
        any(m in zh for m in _SURPRISE)
        or any(m in en for m in _SURPRISE_JA)
        or any(m in en.lower() for m in _SURPRISE_VI)
    ):
        add('surprised')
    if 'laughing' not in tags and 'angry' not in tags and (
        any(m in zh for m in _EXCITED)
        or any(m in en for m in _EXCITED_JA)
        or any(m in en.lower() for m in _EXCITED_VI)
        or en.endswith('！') or en.endswith('!')
    ):
        add('excited')
    if 'angry' not in tags and any(m in zh for m in _SAD):
        add('sad')
    return tags[:2]


def apply_emotion_tags(english, source='', source_is_shared=False, target_language=''):
    line = (english or '').strip()
    if not line:
        return line
    existing = [m.group(1).strip() for m in _TAG_RE.finditer(line)]
    spoken = _TAG_RE.sub('', line).strip()
    tags = existing or infer_emotion_tags(spoken, source, source_is_shared=source_is_shared)
    from tools.target_language import tts_language
    if tts_language(target_language) == 'Japanese':
        tags = [tag for tag in tags if tag.lower() != 'in japanese']
        tags = ['in Japanese'] + tags
    if not tags:
        return spoken
    prefix = ''.join(f'[{tag}]' for tag in tags[:3])
    return f'{prefix} {spoken}'


def spoken_text_for_timing(text):
    return re.sub(r'\s+', ' ', _TAG_RE.sub('', text or '')).strip()

