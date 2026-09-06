# -*- coding: utf-8 -*-
"""Pick Fish Audio emotion tags from a dubbed line."""
import re

_TAG_RE = re.compile(r'\[([^\]]+)\]')
_WORD_RE = re.compile(r"[a-zA-Z']+")
_BANG_RE = re.compile(r'[!！]+')
_Q_RE = re.compile(r'[?？]+')
_UI_RE = re.compile(r'[【\[]\s*(系統|系统|提示|任務|任务)')
_NARR_RE = re.compile(r'narr|旁白|_SYS\b|\bSYSTEM\b', re.I)

# Core Fish S2 tags. Keep to this set; free-form piles make delivery mushy.
_LAUGH = (
    '哈哈', '呵呵', '嘿嘿', '嘻嘻', 'haha', 'hehe', 'lol',
    'へへ', 'ふふ', 'わはは', 'あはは', 'うふふ',
    'ha ha', 'hi hi', 'he he',
)
_WHISPER = (
    '轻轻', '小聲', '小声', '低声', '低聲', '心想', '耳语', '耳語', '悄悄',
    'whisper', 'こっそり', 'ひそひそ', '小声で',
)
_SIGH = ('唉', '啧', '嘖', 'sigh', 'はぁ', 'やれやれ', 'ちっ')
_ANGRY = (
    '混蛋', '滚开', '滾開', '快滚', '快滾', '该死', '該死', '白痴', '闭嘴', '閉嘴',
    '找死', '去死', '可恶', '可惡', '放肆', '废物', '廢物', '恶心', '噁心',
    '妈的', '媽的', '你敢', '休想', '杀了你', '殺了你',
    'idiot', 'damn', 'bastard', 'hell', 'shut up', 'get out', 'kill you',
    'クソ', '馬鹿', 'てめえ', 'ふざける', '死ね',
    'đồ khốn', 'chết tiệt', 'cút đi', 'đồ ngu', 'khốn nạn',
)
_SAD = (
    '伤心', '傷心', '难过', '難過', '哭', '眼泪', '眼淚', '对不起', '對不起',
    '抱歉', '没了', '沒了',
    'sorry', 'sad',
)
_SURPRISE = (
    '居然', '竟然', '不会吧', '不會吧', '怎么可能', '怎麼可能', '不对吧', '不對吧',
    'no way', 'what?', 'wait what',
    'えっ', 'まじ', 'まさか', 'うそだ',
    'trời ơi', 'sao vậy',
)
_EXCITED = (
    '终于', '終於', '太好了', '冲啊', '衝啊', '加油',
    'yes!', 'finally',
    'やった', 'よっしゃ', 'いくぞ',
    'cuối cùng',
)
_NERVOUS = (
    '紧张', '緊張', '害怕', '好怕', '不敢', '救命', '完了', '糟了',
    '怎么办', '怎麼辦', '别杀', '別殺', '求你',
    'scared', 'nervous', 'please don',
)
_RELIEF = ('松了口气', '鬆了口氣', '还好', '還好', 'relieved', 'thank god')
_GASP = ('倒抽', '吸了口气', '吸了口氣')
_COLD = ('可笑', '幼稚', '无聊', '無聊', '省省吧', '得了吧')


def _norm(text):
    return (text or '').strip()


def _has(blob, needles):
    if not blob:
        return False
    low = blob.lower()
    for item in needles:
        if not item:
            continue
        token = item.lower()
        if _WORD_RE.fullmatch(token):
            if re.search(rf"\b{re.escape(token)}\b", low):
                return True
        elif token in low or item in blob:
            return True
    return False


def _is_narration(speaker):
    return bool(_NARR_RE.search(str(speaker or '')))


def infer_emotion_tags(english, source='', source_is_shared=False, speaker=''):
    """Pick up to two Fish S2 bracket tags from this dubbed line."""
    src = source or ''
    en = _norm(english)
    zh = '' if source_is_shared else src
    blob = f'{zh} {en}'
    tags = []

    def add(tag):
        if tag and tag not in tags:
            tags.append(tag)

    compact_zh = re.sub(r'[\s,，。！？!?、…]+', '', zh)
    compact_en = re.sub(r'[\s.,!?]+', '', en).lower()
    if compact_zh in {'哼', '哼哼'} or compact_en in {'hmph', 'humph'}:
        return []
    if _UI_RE.search(src):
        return []

    laugh = _has(blob, _LAUGH)
    if laugh:
        add('laughing')
    if _has(blob, _WHISPER):
        add('whispering')
    if _has(blob, _SIGH) or compact_zh == '唉':
        add('sighing')
    if _has(blob, _ANGRY):
        add('angry')
    if _has(blob, _RELIEF):
        add('relieved')
    if _has(blob, _NERVOUS):
        add('nervous')
    if _has(blob, _GASP) or (compact_zh in {'啊', '呀'} and _BANG_RE.search(zh)):
        add('gasp')
    if 'laughing' not in tags and (
        _has(blob, _SURPRISE)
        or re.search(r'什么[！？?!]|什麼[！？?!]', zh)
        or (compact_zh in {'啊', '诶', '欸'} and _Q_RE.search(zh + en))
    ):
        add('surprised')
    if 'angry' not in tags and _has(blob, _SAD):
        add('sad')
    if 'laughing' not in tags and 'angry' not in tags and _has(blob, _EXCITED):
        add('excited')
    if 'angry' not in tags and 'laughing' not in tags and _has(blob, _COLD):
        add('cold')

    bang = bool(_BANG_RE.search(zh) or en.endswith('!') or en.endswith('！'))
    if bang and not _is_narration(speaker):
        if 'angry' in tags or 'nervous' in tags or 'sad' in tags:
            pass
        elif 'excited' not in tags and 'laughing' not in tags and _has(blob, _EXCITED):
            add('excited')
        elif 'excited' not in tags and 'angry' not in tags and len(en.split()) <= 6:
            add('emphasis')
        elif 'excited' not in tags and 'laughing' not in tags and not tags:
            add('excited')

    if _is_narration(speaker):
        tags = [tag for tag in tags if tag in {'sad', 'whispering', 'sighing', 'cold'}]
    return tags[:2]


def apply_emotion_tags(english, source='', source_is_shared=False, target_language='', speaker=''):
    line = _norm(english)
    if not line:
        return line
    existing = [m.group(1).strip() for m in _TAG_RE.finditer(line)]
    spoken = _TAG_RE.sub('', line).strip()
    tags = existing or infer_emotion_tags(
        spoken, source, source_is_shared=source_is_shared, speaker=speaker,
    )
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


def delivery_temperature(text, japanese=False):
    blob = (text or '').lower()
    if any(tag in blob for tag in ('[angry]', '[excited]', '[surprised]', '[emphasis]')):
        return 0.92
    if any(tag in blob for tag in ('[whispering]', '[sad]', '[sighing]', '[cold]')):
        return 0.78
    return 0.88 if japanese else 0.84


if __name__ == '__main__':
    import os
    import sys
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)
    assert infer_emotion_tags('Get out!', '滚开！') == ['angry']
    assert 'nervous' in infer_emotion_tags('You scared me to death.', '你吓死我了。')
    assert 'laughing' in infer_emotion_tags('Hehe.', '嘿嘿')
    assert infer_emotion_tags('Hmph.', '哼') == []
    assert infer_emotion_tags("It's finally over.", '终于结束了。')[:1] == ['excited']
    assert 'excited' not in infer_emotion_tags('He finally saw it.', '他终于看见了。', speaker='SPEAKER_NARR')
    assert apply_emotion_tags('Get out!', '滚开！').startswith('[angry]')
    print('fish_emotion: ok')
