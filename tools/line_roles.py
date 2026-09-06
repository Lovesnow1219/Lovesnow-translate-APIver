# -*- coding: utf-8 -*-
"""Speaker tags the rest of the pipeline already understands.

Episode knowledge lives in the outline + ASR repair, not in word lists here.
"""
import re
from collections import Counter

from tools.asr_source import _is_particle_only
from tools.target_language import is_asr_junk

# "How can YOU this X say that" / "you wouldn't ask?" — the labeled speaker
# cannot be the person just spoken to, or the person who just delivered the pitch.
_YOU_THIS_RE = re.compile(
    r'你这个|你這個|从你这|從你這|怎么从你|怎麼從你|从你.{0,10}嘴|從你.{0,10}嘴'
)
_INTERRUPT_YOU_RE = re.compile(r'^不是[，,]\s*你不|^不是你不')

SPEAKER_SYS = 'SPEAKER_SYS'
SPEAKER_CLERK = 'SPEAKER_CLERK'
SPEAKER_BGM = 'SPEAKER_BGM'
SPEAKER_NARR = 'SPEAKER_NARR'
FUNCTIONAL_SPEAKERS = {SPEAKER_SYS, SPEAKER_CLERK, SPEAKER_BGM, SPEAKER_NARR, 'SYSTEM', 'CLERK', 'NARRATOR'}


def is_functional_speaker(speaker):
    name = str(speaker or '')
    return name in FUNCTIONAL_SPEAKERS or name.endswith('_SYS')


def is_system_line(text):
    return False


def is_clerk_line(text):
    return False


def line_role(text):
    return ''


def looks_like_speech(text):
    return bool((text or '').strip())


def looks_like_dialogue(text):
    return looks_like_speech(text)


def is_asr_stump(line):
    item = line if isinstance(line, dict) else {'text': line or ''}
    src = (item.get('text') or '').strip()
    if not src or _is_particle_only(src) or is_asr_junk(src):
        return False
    compact = re.sub(r'[\s,，。！？!?、…]+', '', src)
    if compact and all('\u4e00' <= ch <= '\u9fff' for ch in compact) and len(compact) >= 2:
        return False
    if not isinstance(line, dict):
        return False
    han = sum(1 for ch in src if '\u4e00' <= ch <= '\u9fff')
    try:
        duration = max(0.0, float(item.get('end') or 0) - float(item.get('start') or 0))
    except (TypeError, ValueError):
        duration = 0.0
    if duration and duration <= 0.35 and han <= 3:
        return True
    return han <= 2 and not src.endswith(('。', '！', '？', '!', '?')) and duration <= 0.45


def _spoken_han(text):
    return sum(1 for ch in (text or '') if '\u4e00' <= ch <= '\u9fff')


def promote_bgm_speech(transcript):
    """Speech sitting in the music bed is still dialogue. Do not leave it as BGM skip."""
    n = 0
    for line in transcript or []:
        text = (line.get('text') or '').strip()
        if _spoken_han(text) < 8:
            continue
        speaker = str(line.get('speaker') or '')
        skipped = bool(line.get('skip_tts'))
        if speaker == SPEAKER_BGM or (skipped and speaker in {SPEAKER_BGM, SPEAKER_NARR, 'NARRATOR', ''}):
            line['speaker'] = SPEAKER_NARR
            line.pop('skip_tts', None)
            n += 1
    return n


def _is_talk_card(line):
    text = ((line or {}).get('text') or '').strip()
    return bool(text) and not _is_particle_only(text) and not is_asr_junk(text)


def _neighbor_talk(transcript, index, step):
    i = index + step
    while 0 <= i < len(transcript):
        if _is_talk_card(transcript[i]):
            return transcript[i]
        i += step
    return None


def _lead_ids(transcript):
    counts = Counter()
    for line in transcript or []:
        speaker = str(line.get('speaker') or '')
        if is_functional_speaker(speaker) or not _is_talk_card(line):
            continue
        if _spoken_han(line.get('text')) < 2:
            continue
        counts[speaker] += 1
    return [speaker for speaker, n in counts.most_common() if n >= 2]


def _other_lead(leads, exclude):
    blocked = {name for name in exclude if name}
    for speaker in leads:
        if speaker not in blocked:
            return speaker
    return ''


def _addressee_speaker(transcript, index):
    """The person 'you' most likely points at: the longest recent talker."""
    best_spk = ''
    best_han = 0
    looked = 0
    i = index - 1
    while i >= 0 and looked < 6:
        line = transcript[i]
        i -= 1
        if not _is_talk_card(line):
            continue
        looked += 1
        speaker = str(line.get('speaker') or '')
        if not speaker or is_functional_speaker(speaker):
            continue
        han = _spoken_han(line.get('text'))
        if han > best_han:
            best_han = han
            best_spk = speaker
    if best_spk:
        return best_spk
    prev_line = _neighbor_talk(transcript, index, -1)
    return str((prev_line or {}).get('speaker') or '')


def reassign_addressed_you_lines(transcript):
    """Move cards that talk *to* someone off that someone.

    Diarization glues similar voices. 「你这个…」 / 「不是，你不问…」
    is almost never said by the person just holding forth.
    """
    leads = _lead_ids(transcript)
    moved = 0
    for index, line in enumerate(transcript or []):
        src = (line.get('text') or '').strip()
        speaker = str(line.get('speaker') or '')
        if not speaker or is_functional_speaker(speaker):
            continue
        next_line = _neighbor_talk(transcript, index, 1)
        next_spk = str((next_line or {}).get('speaker') or '')
        dest = ''
        if _YOU_THIS_RE.search(src):
            addressed = _addressee_speaker(transcript, index)
            if addressed and speaker == addressed:
                dest = _other_lead(leads, {speaker, addressed})
        elif _INTERRUPT_YOU_RE.search(src):
            addressed = _addressee_speaker(transcript, index)
            prev_line = _neighbor_talk(transcript, index, -1)
            prev_spk = str((prev_line or {}).get('speaker') or '')
            if addressed and speaker == addressed:
                dest = _other_lead(leads, {speaker, addressed, prev_spk, next_spk})
        if dest and dest != speaker:
            line['speaker'] = dest
            moved += 1
    return moved


def should_skip_dub(line):
    if not line:
        return True
    item = line if isinstance(line, dict) else {'text': line}
    speaker = str(item.get('speaker') or '')
    if is_asr_junk(item.get('text')):
        return True
    if speaker in {SPEAKER_SYS, SPEAKER_NARR, 'NARRATOR', 'SYSTEM'}:
        return False
    if _spoken_han(item.get('text')) >= 8:
        return False
    if item.get('skip_tts') or speaker == SPEAKER_BGM:
        return True
    return is_asr_stump(item)


def normalize_line_roles(transcript):
    """Kept for callers. Roles come from ASR repair / the outline, not regex."""
    return list(transcript or [])
