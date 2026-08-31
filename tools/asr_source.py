# -*- coding: utf-8 -*-
"""Split, glue, and clean ASR source cards before translation."""
import re

from tools.target_language import is_asr_junk

def split_text_into_sentences(para):
    holders = {}

    def _hold(match):
        key = f'ABBREV{len(holders)}'
        holders[key] = match.group(0)
        return key

    para = re.sub(r'\bP\.S\.', _hold, para, flags=re.I)
    para = re.sub(r'\b(?:Mr|Mrs|Ms|Dr|Prof|vs|etc)\.', _hold, para)
    para = re.sub('([。！？!?\?])([^，。！？!?\?”’》])', r"\1\n\2", para)
    para = re.sub(r'(?<!\d)\.(?!\d)([^，。！？!?\?”’》])', r'.\n\1', para)
    para = re.sub('(\.{6})([^，。！？!?\?”’》])', r"\1\n\2", para)
    para = re.sub('(\…{2})([^，。！？!?\?”’》])', r"\1\n\2", para)
    para = re.sub('([。！？!?\?][”’])([^，。！？!?\?”’》])', r'\1\n\2', para)
    para = para.rstrip()
    lines = [line.strip() for line in para.split("\n") if line.strip()]
    restored = []
    for line in lines:
        for key, value in holders.items():
            line = line.replace(key, value)
        if line in {'.', '..', '...'}:
            continue
        restored.append(line)
    return restored


def _source_chars(text):
    return len(re.sub(r'\s+', '', text or ''))

def split_long_segments(transcript, max_seconds=10.0):
    """Break mega ASR blobs so each dubbing line has a realistic speaking window."""
    packed = []
    for item in transcript:
        start = float(item.get('start') or 0)
        end = float(item.get('end') or start)
        text = (item.get('text') or '').strip()
        speaker = item.get('speaker', 'SPEAKER_00')
        duration = max(0.0, end - start)
        if is_asr_junk(text) or duration <= max_seconds or len(text) < 24:
            packed.append(item)
            continue
        parts = []
        for sentence in split_text_into_sentences(text) or [text]:
            bits = [bit.strip() for bit in re.split(r'[\s,，、]+', sentence) if bit.strip()]
            parts.extend(bits or [sentence])
        if len(parts) <= 1:
            n = max(2, int((duration / max_seconds) + 0.999))
            chunk = max(8, (len(text) + n - 1) // n)
            parts = [text[i:i + chunk].strip() for i in range(0, len(text), chunk) if text[i:i + chunk].strip()]
        groups = []
        buf = ''
        total_chars = sum(len(p) for p in parts) or 1
        for part in parts:
            tentative = buf + (' ' if buf else '') + part
            tentative_dur = duration * (len(tentative) / total_chars)
            if buf and tentative_dur > max_seconds:
                groups.append(buf)
                buf = part
            else:
                buf = tentative
        if buf:
            groups.append(buf)
        group_chars = sum(len(g) for g in groups) or 1
        cursor = start
        for i, group in enumerate(groups):
            span = duration * (len(group) / group_chars)
            group_end = end if i == len(groups) - 1 else cursor + span
            packed.append({
                'start': round(cursor, 3),
                'end': round(group_end, 3),
                'text': group,
                'speaker': speaker,
            })
            cursor = group_end
    return split_packed_utterances(packed)


_PACKED_PUNCT = re.compile(r'(?<=[。！？；;!?])\s*')


def _packed_parts(text):
    punct = [part.strip() for part in _PACKED_PUNCT.split(text) if part.strip()]
    if len(punct) >= 2:
        return punct, 'punct'
    spaces = [part.strip() for part in re.split(r'[ \t]+', text) if part.strip()]
    return spaces, 'space'


def split_packed_utterances(transcript, min_duration=1.0):
    """ASR often glues two subtitle cards, e.g. '币是 继续说下去' or '你好。继续。'."""
    packed = []
    for item in transcript:
        text = (item.get('text') or '').strip()
        start = float(item.get('start') or 0)
        end = float(item.get('end') or start)
        duration = max(0.0, end - start)
        parts, kind = _packed_parts(text)
        keep = duration < min_duration or len(parts) < 2
        if not keep and kind == 'punct':
            keep = any(len(part) < 1 for part in parts)
        if not keep and kind == 'space':
            keep = not (2 <= len(parts) <= 6 and all(2 <= len(part) <= 24 for part in parts))
            if not keep and all(re.fullmatch(r"[A-Za-z0-9'.,!?-]+", part or '') for part in parts):
                keep = len(parts) > 3
        if keep:
            packed.append(item)
            continue
        total = sum(len(part) for part in parts) or 1
        cursor = start
        for i, part in enumerate(parts):
            span = max(0.35, duration * (len(part) / total))
            part_end = end if i == len(parts) - 1 else min(end, cursor + span)
            extra = {
                key: value for key, value in item.items()
                if key not in {'start', 'end', 'text', 'translation', 'orig_start', 'orig_end'}
            }
            packed.append({
                **extra,
                'start': round(cursor, 3),
                'end': round(part_end, 3),
                'orig_start': round(cursor, 3),
                'orig_end': round(part_end, 3),
                'text': part,
            })
            cursor = part_end
    return packed


_ASR_SOURCE_FIXES = (
    ('需积', '囤积'),
    ('需積', '囤積'),
    ('巨罕', '巨款'),
    ('本书女', '本小姐'),
    ('粉树女', '本小姐'),
    ('本树女', '本小姐'),
)
_SENTENCE_END = ('。', '！', '？', '!', '?')
_CLAUSE_END = ('，', ',', '、', '：', ':')
_LEAD_INS = {
    '看来', '那麼', '那么', '也就是说', '也就是說', '而且', '可是', '但是',
    '所以', '于是', '於是', '就是', '不过', '不過', '还有', '還有',
}
_PARTICLE_ONLY = {'嗯', '恩', '呵', '哼', '嘿', '嘿嘿', '哈哈', '嘻嘻', '啊', '哦', '噢', '呃', '唉', '欸', '唔'}
_STUMP_PHRASES = {
    'congratulations',
    'good heavens',
    'you see',
    'see',
    'look',
    "that's it",
    'looks like it',
}


def cleanup_asr_source(text):
    cleaned = text or ''
    for src, dst in _ASR_SOURCE_FIXES:
        cleaned = cleaned.replace(src, dst)
    cleaned = re.sub(r'(^|[\s,，、])恩(?=[\s,，。！？!?、…]|$)', r'\1嗯', cleaned)
    compact = re.sub(r'\s+', ' ', cleaned).strip()
    if re.fullmatch(r'看[，,\s]*买[，,\s]*好[！!。.]?', compact):
        return '好看，买！'
    cleaned = re.sub(r'(^|[\s,，])看\s+好看', r'\1好看', cleaned)
    cleaned = re.sub(r'好看\s+好看', '好看，好看', cleaned)
    return cleaned


def _is_particle_only(text):
    compact = re.sub(r'[\s,，。！？!?、…]+', '', text or '')
    return compact in _PARTICLE_ONLY


def _glue_source(left, right):
    left = (left or '').strip()
    right = (right or '').strip()
    if not left:
        return right
    if not right or right == left or left.endswith(right):
        return left
    if right.endswith(left) or left in right:
        return right
    if '\u4e00' <= left[-1] <= '\u9fff' and '\u4e00' <= right[0] <= '\u9fff':
        return left + right
    if left[-1].isascii() and right[0].isascii():
        return left + ' ' + right
    return left + right


def _can_merge_utterance(prev, nxt):
    if str(prev.get('speaker') or '') != str(nxt.get('speaker') or ''):
        return False
    if is_asr_junk(prev.get('text')) or is_asr_junk(nxt.get('text')):
        return False
    left = (prev.get('text') or '').strip()
    right = (nxt.get('text') or '').strip()
    if not left or not right:
        return False
    if _is_particle_only(left) or _is_particle_only(right):
        return False
    gap = float(nxt.get('start') or 0) - float(prev.get('end') or 0)
    if gap > 0.16:
        return False
    glued = _glue_source(left, right)
    if _source_chars(glued) > 96:
        return False
    span = float(nxt.get('end') or 0) - float(prev.get('start') or 0)
    if span > 14:
        return False
    if left == right:
        return True
    if left.rstrip().endswith(_SENTENCE_END):
        return False
    right_lead = right.rstrip().rstrip('，,')
    if right_lead in _LEAD_INS and _source_chars(right) <= 8:
        return False
    if left.rstrip().endswith(_CLAUSE_END):
        return True
    lead = left.rstrip().rstrip('，,')
    if lead in _LEAD_INS:
        return True
    if left.rstrip().endswith(tuple('啊呀吧呢吗嗎了嘛哦噢哇哈')):
        return False
    # Long unfinished blob cut mid-line by ASR (e.g. 获得奖 + 励现实).
    return (
        gap <= 0.12
        and _source_chars(left) >= 16
        and not left.rstrip().endswith(_SENTENCE_END)
    )


def merge_utterance_cards(transcript):
    """Join same-speaker fragments that were one spoken line before translating."""
    if not transcript:
        return transcript
    merged = []
    i = 0
    while i < len(transcript):
        cur = dict(transcript[i])
        glued = 0
        while i + 1 < len(transcript):
            nxt = transcript[i + 1]
            if not _can_merge_utterance(cur, nxt):
                break
            same = (cur.get('text') or '').strip() == (nxt.get('text') or '').strip()
            comma = (cur.get('text') or '').rstrip().endswith(_CLAUSE_END)
            if glued >= 1 and not same and not comma:
                break
            cur['text'] = _glue_source(cur.get('text'), nxt.get('text'))
            cur['end'] = nxt.get('end')
            if nxt.get('orig_end') is not None:
                cur['orig_end'] = nxt.get('orig_end')
            elif cur.get('orig_start') is not None:
                cur['orig_end'] = nxt.get('end')
            glued += 1
            i += 1
        merged.append(cur)
        i += 1
    return merged


