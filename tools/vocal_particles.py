# -*- coding: utf-8 -*-
"""Recover short 呵 / 嗯 / 哼 vocalizations that ASR often drops."""
import os
import re

import numpy as np
from loguru import logger


_PARTICLE_TEXTS = {
    '嗯', '恩', '呵', '哼', '嘿', '嘿嘿', '哈哈', '嘻嘻',
    '啊', '哦', '噢', '呃', '唉', '欸', '唔',
}
_HENG_MARKS = {'哼', '哼哼'}
_LAUGH_MARKS = {'嘿', '嘿嘿', '哈哈', '嘻嘻'}
_OVERLAP_EPS = 0.04


def _bursts(samples, sample_rate, hop=0.02, win=0.04, floor=0.025):
    hop_i = max(1, int(hop * sample_rate))
    win_i = max(hop_i, int(win * sample_rate))
    found = []
    on = False
    burst_start = 0.0
    peak = 0.0
    for i in range(0, max(0, len(samples) - win_i), hop_i):
        value = float(np.sqrt(np.mean(np.square(samples[i:i + win_i]))))
        t = i / float(sample_rate)
        if value >= floor:
            if not on:
                on = True
                burst_start = t
                peak = value
            else:
                peak = max(peak, value)
        elif on:
            found.append((burst_start, t, peak))
            on = False
    if on:
        found.append((burst_start, len(samples) / float(sample_rate), peak))
    return found


def _overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def card_start(line):
    if line.get('orig_start') is not None:
        return float(line['orig_start'])
    return float(line.get('start') or 0)


def card_end(line):
    if line.get('orig_end') is not None:
        return float(line['orig_end'])
    return float(line.get('end') or card_start(line))


def _nearest_speaker(transcript, t):
    best = 'SPEAKER_00'
    best_dist = 1e9
    for line in transcript or []:
        start = card_start(line)
        end = card_end(line)
        if start <= t <= end:
            return line.get('speaker') or best
        dist = min(abs(t - start), abs(t - end))
        if dist < best_dist:
            best_dist = dist
            best = line.get('speaker') or best
    return best


def _is_particle_text(text):
    compact = ''.join(ch for ch in (text or '') if ch.strip() and ch not in '，,。！？!?、…')
    return compact in _PARTICLE_TEXTS


def is_particle_card(text):
    return _is_particle_text(text)


def _particle_glyph(text):
    compact = ''.join(ch for ch in (text or '') if ch.strip() and ch not in '，,。！？!?、…')
    if compact in {'嗯', '恩', '唔'}:
        return '嗯'
    if compact in {'呵', '呵呵'}:
        return '呵'
    if compact in _HENG_MARKS:
        return '哼'
    if compact in _LAUGH_MARKS:
        return '嘿嘿' if compact in {'嘿', '嘿嘿'} else compact
    if compact in _PARTICLE_TEXTS:
        return compact
    return ''


def _host_starts_with_particle(text):
    body = (text or '').lstrip()
    return body.startswith(('嗯', '恩', '呵', '哼', '嘿', '哈哈', '嘻嘻'))


_LEAD_ALIASES = {
    '嗯': {
        'latin': ('Hmm', 'Mm', 'Mhm', 'Hum'),
        'vi': ('Ừ',),
        'ja': ('ん',),
        'ko': ('음',),
        'es': ('Mm',),
        'fr': ('Hum',),
    },
}


def _source_lead_glyph(text):
    body = (text or '').lstrip()
    if not body or _is_particle_text(body):
        return ''
    for mark in ('嘿嘿', '哈哈', '嘻嘻', '嗯', '恩', '呵', '哼', '嘿'):
        if body.startswith(mark):
            rest = body[len(mark):].lstrip('，, ')
            if not rest:
                return ''
            if mark in {'嗯', '恩'}:
                return '嗯'
            if mark == '哼':
                return '哼'
            if mark in _LAUGH_MARKS:
                return '嘿嘿' if mark in {'嘿', '嘿嘿'} else mark
            return '呵'
    return ''


def particle_lead_stems(glyph, target_language='English'):
    mapped = particle_translation(glyph, target_language)
    stem = (mapped or '').rstrip('。. ')
    kind = _particle_kind(target_language)
    aliases = _LEAD_ALIASES.get(glyph, {}).get(kind, ())
    out = []
    for item in (stem,) + tuple(aliases):
        if item and item not in out:
            out.append(item)
    return out


def translation_has_particle_lead(trans, glyph, target_language='English'):
    body = (trans or '').lstrip()
    if not body or not glyph:
        return False
    low = body.lower()
    return any(low.startswith(stem.lower()) for stem in particle_lead_stems(glyph, target_language))


def collapse_double_particle_lead(trans, target_language='English'):
    text = trans or ''
    kind = _particle_kind(target_language)
    if kind in {'latin', 'es', 'fr'}:
        return re.sub(
            r'^(Hmm|Mm|Mhm|Hum)\s*,\s*(Hmm|Mm|Mhm|Hum)\b',
            r'\2',
            text,
            count=1,
            flags=re.I,
        )
    return text


def particle_lead_in(text, target_language='English'):
    """Comma lead for a host line that already starts with 嗯/呵/哼. Empty for particle-only."""
    glyph = _source_lead_glyph(text)
    if not glyph:
        return ''
    mapped = particle_translation(glyph, target_language)
    if not mapped:
        return ''
    stem = mapped.rstrip('。. ')
    lang = (target_language or '').lower()
    if 'japan' in lang:
        return stem + '、'
    return stem + ', '


def ensure_particle_translation_lead(line, target_language='English'):
    trans = collapse_double_particle_lead((line.get('translation') or '').lstrip(), target_language)
    glyph = _source_lead_glyph(line.get('text'))
    lead = particle_lead_in(line.get('text'), target_language)
    if trans and glyph and translation_has_particle_lead(trans, glyph, target_language):
        line['translation'] = trans
        return line
    if not lead or not trans:
        if trans:
            line['translation'] = trans
        return line
    line['translation'] = collapse_double_particle_lead(lead + trans, target_language)
    return line


def prepend_particle_to_host(host, particle_text, target_language=None):
    """Prefix 嗯，/呵，/哼， onto a host. Skip if the host already starts with a particle."""
    glyph = _particle_glyph(particle_text) or '嗯'
    body = (host.get('text') or '').lstrip()
    if body and not _host_starts_with_particle(body):
        host['text'] = f'{glyph}，' + body
    if host.get('translation'):
        ensure_particle_translation_lead(host, target_language or '')
    return host


def _quiet_after(samples, sample_rate, t, span=0.18, floor=0.02):
    start_i = max(0, int(t * sample_rate))
    end_i = min(len(samples), int((t + span) * sample_rate))
    if end_i <= start_i:
        return False
    return float(np.sqrt(np.mean(np.square(samples[start_i:end_i])))) < floor


def repair_overlapping_particle_cards(transcript, target_language=None):
    """Merge a particle-only card into the longer host if their orig windows overlap."""
    if not transcript:
        return list(transcript or [])
    lines = [dict(item) for item in transcript]
    lines.sort(key=lambda item: (card_start(item), -(card_end(item) - card_start(item))))
    keep = [True] * len(lines)
    merged = 0
    for i, particle in enumerate(lines):
        if not keep[i] or not _is_particle_text(particle.get('text')):
            continue
        ps, pe = card_start(particle), card_end(particle)
        host_idx = None
        host_score = None
        for j, other in enumerate(lines):
            if i == j or not keep[j]:
                continue
            os, oe = card_start(other), card_end(other)
            if _overlap(ps, pe, os, oe) <= _OVERLAP_EPS:
                continue
            is_host = not _is_particle_text(other.get('text'))
            score = (1 if is_host else 0, oe - os, len((other.get('text') or '')))
            if host_score is None or score > host_score:
                host_score = score
                host_idx = j
        if host_idx is None:
            continue
        host = lines[host_idx]
        if _is_particle_text(host.get('text')):
            keep[i] = False
            merged += 1
            continue
        prepend_particle_to_host(host, particle.get('text'), target_language)
        keep[i] = False
        merged += 1
    out = [line for i, line in enumerate(lines) if keep[i]]
    out.sort(key=lambda item: card_start(item))
    if merged:
        logger.info(f'合併 {merged} 張重疊語氣詞卡片到主句')
    return out


def next_nonoverlapping_start(transcript, index):
    """Start of the next card that does not sit inside this line's orig window."""
    orig_end = card_end(transcript[index])
    for j in range(index + 1, len(transcript)):
        nxt_start = card_start(transcript[j])
        if nxt_start < orig_end - _OVERLAP_EPS:
            continue
        return nxt_start
    return None


def available_tts_seconds(transcript, index, start=None):
    """Room for this line: until the next non-overlapping card, never an inner particle."""
    if start is None:
        start = card_start(transcript[index])
    nxt = next_nonoverlapping_start(transcript, index)
    if nxt is None:
        return None
    return max(0.08, nxt - start - 0.04)


def is_inner_overlapping_particle(transcript, index):
    line = transcript[index]
    if not _is_particle_text(line.get('text')):
        return False
    ps, pe = card_start(line), card_end(line)
    for j, other in enumerate(transcript):
        if j == index or _is_particle_text(other.get('text')):
            continue
        if _overlap(ps, pe, card_start(other), card_end(other)) > _OVERLAP_EPS:
            return True
    return False


def _burst_features(samples, sample_rate, start, end):
    clip = np.asarray(samples[int(start * sample_rate):int(end * sample_rate)], dtype=np.float32)
    if clip.size < 64:
        return None
    window = np.hanning(clip.size)
    spec = np.abs(np.fft.rfft(clip * window)) + 1e-12
    freqs = np.fft.rfftfreq(clip.size, 1.0 / float(sample_rate))
    centroid = float(np.sum(freqs * spec) / np.sum(spec))
    low = float(spec[(freqs >= 200) & (freqs <= 800)].sum())
    high = float(spec[(freqs >= 1500) & (freqs <= 4000)].sum())
    zcr = float(np.mean(np.abs(np.diff(np.sign(clip)))) / 2.0)
    return {
        'centroid': centroid,
        'nasal': low / (high + 1e-9),
        'zcr': zcr,
        'peak': float(np.max(np.abs(clip))),
        'duration': float(end - start),
    }


def classify_vocal_burst(samples, sample_rate, start, end):
    """呵 = breathy heh; 哼 = nasal hmph. Empty if the clip is too short to tell."""
    feat = _burst_features(samples, sample_rate, start, end)
    if not feat:
        return ''
    if feat['nasal'] >= 4.0 and feat['centroid'] < 1600 and feat['zcr'] < 0.16:
        return '哼'
    return '呵'


def reclassify_particles_from_vocals(folder, transcript, target_language=None):
    """Turn a saved 呵/嗯 card into 哼 when the vocal clip is a nasal hmph."""
    if not transcript or not folder:
        return list(transcript or [])
    wav_path = os.path.join(folder, 'audio_vocals.wav')
    if not os.path.isfile(wav_path):
        return [dict(item) for item in transcript]
    try:
        import librosa
        samples, sample_rate = librosa.load(wav_path, sr=24000, mono=True)
    except Exception as exc:
        logger.warning(f'無法依人聲重標語氣詞：{exc}')
        return [dict(item) for item in transcript]
    out = [dict(item) for item in transcript]
    flipped = 0
    for line in out:
        if not _is_particle_text(line.get('text')):
            continue
        current = _particle_glyph(line.get('text'))
        guess = classify_vocal_burst(samples, sample_rate, card_start(line), card_end(line))
        if guess != '哼' or current == '哼':
            continue
        line['text'] = '哼'
        mapped = particle_translation('哼', target_language or '')
        if mapped:
            line['translation'] = mapped
        flipped += 1
    if flipped:
        logger.info(f'依人聲把 {flipped} 個語氣詞改成哼')
    return out


def recover_vocal_particles(folder, transcript, isolated_glyphs=None):
    """Insert 呵/嗯/哼 cards for short voice bursts that have no ASR text.

    isolated_glyphs: if set, only insert those isolated particles (used on redub
    so we can add missed 哼 without dumping extra 呵).
    """
    if not transcript:
        return transcript
    wav_path = os.path.join(folder, 'audio_vocals.wav')
    if not os.path.isfile(wav_path):
        return repair_overlapping_particle_cards([dict(item) for item in transcript])
    try:
        import librosa
        samples, sample_rate = librosa.load(wav_path, sr=24000, mono=True)
    except Exception as exc:
        logger.warning(f'無法掃描語氣詞空隙：{exc}')
        return repair_overlapping_particle_cards([dict(item) for item in transcript])

    lines = [dict(item) for item in transcript]
    added = []

    def already(start):
        return any(
            abs(card_start(item) - start) < 0.1 and _is_particle_text(item.get('text'))
            for item in lines + added
        )

    def live_windows():
        return [(card_start(line), card_end(line), line) for line in lines + added]

    for start, end, peak in _bursts(samples, sample_rate):
        duration = end - start
        if duration < 0.14 or duration > 0.85 or peak < 0.18:
            continue
        mid = (start + end) / 2
        host = None
        covered = 0.0
        prev_end = -1.0
        next_start = 1e9
        for line_start, line_end, line in live_windows():
            covered += _overlap(start, end, line_start, line_end)
            if line_end <= start + 0.02:
                prev_end = max(prev_end, line_end)
            if line_start >= end - 0.02:
                next_start = min(next_start, line_start)
            if line_start <= mid <= line_end:
                host = (line_start, line_end, line)
        gap_before = start - prev_end
        hole = next_start - prev_end
        isolated = (
            covered / max(duration, 0.01) <= 0.3
            and hole >= 1.5
            and gap_before >= 0.25
            and next_start - end >= 0.20
            and peak >= 0.35
        )
        onset = False
        if host and not isolated and duration <= 0.40:
            line_start, line_end, _line = host
            pause = line_start - prev_end
            onset = (
                pause >= 1.5
                and start <= line_start + 0.40
                and peak >= 0.18
                and _quiet_after(samples, sample_rate, end, span=0.16)
                and line_end - end >= 0.8
            )
        if not isolated and not onset:
            continue
        if already(start):
            continue
        glyph = classify_vocal_burst(samples, sample_rate, start, end) or '呵'
        if isolated_glyphs and glyph not in isolated_glyphs:
            continue
        if onset and host:
            prepend_particle_to_host(host[2], '哼' if glyph == '哼' else '嗯')
            continue
        overlapping = []
        for line_start, line_end, line in live_windows():
            if _overlap(start, end, line_start, line_end) > _OVERLAP_EPS:
                overlapping.append((line_start, line_end, line))
        if overlapping:
            line_start, line_end, line = max(
                overlapping,
                key=lambda item: (item[1] - item[0], len((item[2].get('text') or ''))),
            )
            if not _is_particle_text(line.get('text')) and start <= line_start + 0.40:
                prepend_particle_to_host(line, '哼' if glyph == '哼' else '嗯')
            continue
        if prev_end >= 0:
            start = max(start, prev_end + _OVERLAP_EPS)
        if next_start < 1e8:
            end = min(end, next_start - 0.06)
        if end - start < 0.14:
            continue
        added.append({
            'start': round(start, 3),
            'end': round(max(end, start + 0.22), 3),
            'orig_start': round(start, 3),
            'orig_end': round(end, 3),
            'text': glyph,
            'speaker': _nearest_speaker(lines, mid),
        })
    pulses = [
        (start, end, peak)
        for start, end, peak in _bursts(samples, sample_rate)
        if 0.12 <= (end - start) <= 0.50 and peak >= 0.11
    ]
    used = set()
    for i, (s1, e1, p1) in enumerate(pulses):
        if i in used or i + 1 >= len(pulses):
            continue
        s2, e2, p2 = pulses[i + 1]
        gap = s2 - e1
        if gap < 0.05 or gap > 0.40 or max(p1, p2) < 0.13:
            continue
        start, end = s1, e2
        mid = (start + end) / 2
        if any(_overlap(start, end, card_start(item), card_end(item)) > 0.08 for item in lines + added):
            continue
        if already(start):
            continue
        prev_end = max(
            (card_end(item) for item in lines + added if card_end(item) <= start + 0.02),
            default=-1.0,
        )
        next_start = min(
            (card_start(item) for item in lines + added if card_start(item) >= end - 0.02),
            default=1e9,
        )
        if next_start - prev_end < 0.80:
            continue
        used.add(i)
        used.add(i + 1)
        nxt = None
        for item in lines + added:
            if card_start(item) >= end - 0.02:
                nxt = item
                break
        speaker = (nxt.get('speaker') if nxt and card_start(nxt) - end < 2.2 else None) or _nearest_speaker(lines, mid)
        added.append({
            'start': round(start, 3),
            'end': round(max(end, start + 0.28), 3),
            'orig_start': round(start, 3),
            'orig_end': round(end, 3),
            'text': '嘿嘿',
            'speaker': speaker,
        })
    if added:
        lines.extend(added)
        lines.sort(key=lambda item: card_start(item))
        logger.info(
            f'補上 {len(added)} 個被辨識漏掉的語氣詞：'
            + '、'.join(f"{item['text']}@{item['start']:.1f}s" for item in added)
        )
    lines = reclassify_particles_from_vocals(folder, lines)
    return repair_overlapping_particle_cards(lines)


def _particle_kind(target_language):
    try:
        from tools.target_language import translation_language
        lang = translation_language(target_language)
    except Exception:
        lang = str(target_language or 'English')
    by_name = {
        'Japanese': 'ja',
        'Vietnamese': 'vi',
        'Korean': 'ko',
        'Thai': 'th',
        'Spanish': 'es',
        'French': 'fr',
        '简体中文': 'zh',
        '繁体中文': 'zh',
        '中文': 'zh',
        'Cantonese': 'zh',
    }
    if lang in by_name:
        return by_name[lang]
    low = lang.lower()
    if 'japan' in low:
        return 'ja'
    if 'viet' in low or '越' in lang:
        return 'vi'
    if 'korea' in low:
        return 'ko'
    if 'thai' in low or '泰' in lang:
        return 'th'
    if 'span' in low or '西' in lang:
        return 'es'
    if 'french' in low or '法' in lang:
        return 'fr'
    if any(mark in lang for mark in ('简体', '繁体', '中文', '粤', '粵', 'canton')):
        return 'zh'
    return 'latin'


def particle_translation(text, target_language='English'):
    compact = ''.join(ch for ch in (text or '') if ch.strip() and ch not in '，,。！？!?、…')
    kind = _particle_kind(target_language)
    if compact in {'呵', '呵呵'}:
        return {'ja': 'ふふ。', 'vi': 'Hề.', 'ko': '헤.', 'th': 'เฮะ', 'es': 'Je.', 'fr': 'Hé.', 'zh': '呵。'}.get(kind, 'Heh.')
    if compact in {'嗯', '恩', '唔'}:
        return {'ja': 'ん。', 'vi': 'Ừ.', 'ko': '음.', 'th': 'อืม', 'es': 'Mm.', 'fr': 'Hum.', 'zh': '嗯。'}.get(kind, 'Hmm.')
    if compact in _HENG_MARKS:
        return {'ja': 'ふん。', 'vi': 'Hừ.', 'ko': '흥.', 'th': 'หึ', 'es': 'Hum.', 'fr': 'Hmpf.', 'zh': '哼。'}.get(kind, 'Hmph.')
    if compact in _LAUGH_MARKS:
        if compact in {'哈哈', '嘻嘻'}:
            return {'ja': 'はは。', 'vi': 'Ha ha.', 'ko': '하하.', 'th': 'ฮ่าฮ่า', 'es': 'Jaja.', 'fr': 'Haha.', 'zh': '哈哈。'}.get(kind, 'Haha.')
        return {'ja': 'へへ。', 'vi': 'He he.', 'ko': '헤헤.', 'th': 'ฮี่ฮี่', 'es': 'Jeje.', 'fr': 'Héhé.', 'zh': '嘿嘿。'}.get(kind, 'Hehe.')
    return ''


def _smoke():
    ship = {'start': 1.46, 'end': 4.31, 'text': '我们的新船终于竣工了'}
    heh = {
        'start': 5.86, 'end': 6.46, 'orig_start': 5.86, 'orig_end': 6.46, 'text': '呵',
    }
    host = {
        'start': 6.56, 'end': 9.61, 'orig_start': 6.56, 'orig_end': 9.61,
        'text': '这下终于可以向着更远的海域进发了',
        'translation': 'now we can finally set sail for farther waters!',
    }
    inner = {
        'start': 6.72, 'end': 7.00, 'orig_start': 6.72, 'orig_end': 7.00, 'text': '嗯',
    }
    heh2 = {
        'start': 10.64, 'end': 10.86, 'orig_start': 10.64, 'orig_end': 10.86, 'text': '呵',
    }
    wardrobe = {
        'start': 302.926, 'end': 303.301, 'orig_start': 302.926, 'orig_end': 303.301,
        'text': '嗯,',
    }
    after = {
        'start': 303.301, 'end': 305.926, 'orig_start': 303.301, 'orig_end': 305.926,
        'text': '这套是挺不错的,就先买这些吧',
    }
    out = repair_overlapping_particle_cards(
        [ship, heh, host, inner, heh2, wardrobe, after],
        target_language='English',
    )
    assert len(out) == 6, len(out)
    sail = next(item for item in out if '海域' in (item.get('text') or ''))
    assert sail['text'].startswith('嗯，'), sail['text']
    assert sail['orig_start'] == 6.56
    assert sail['translation'].startswith('Hmm,')
    assert not any(abs(card_start(item) - 6.72) < 0.05 for item in out)
    again = repair_overlapping_particle_cards(out, target_language='English')
    assert again[2]['text'].count('嗯') == 1
    assert any(_is_particle_text(item.get('text')) and abs(card_start(item) - 302.926) < 0.01 for item in again)
    crush = available_tts_seconds(out, 2)
    assert crush is None or crush > 2.0, crush
    dirty = [ship, heh, dict(host), dict(inner), heh2]
    assert available_tts_seconds(dirty, 2) > 2.0
    assert is_inner_overlapping_particle(dirty, 3)
    assert particle_translation('哼', 'English') == 'Hmph.'
    assert particle_translation('嘿嘿', 'English') == 'Hehe.'
    assert particle_translation('呵', 'Japanese') == 'ふふ。'
    assert particle_translation('哼', 'Japanese') == 'ふん。'
    assert particle_translation('嘿嘿', 'Japanese') == 'へへ。'
    assert particle_translation('呵', '越南文') == 'Hề.'
    assert particle_translation('哼', '越南文') == 'Hừ.'
    assert particle_translation('嘿嘿', 'Vietnamese') == 'He he.'
    assert particle_translation('呵', 'Korean（部分支援）') == '헤.'
    assert _is_particle_text('嘿嘿')
    assert _is_particle_text('哼')
    heng_host = {
        'start': 20, 'end': 24, 'text': '这有什么好垄断的',
        'translation': "What's the point of a monopoly?",
    }
    prepend_particle_to_host(heng_host, '哼', 'English')
    assert heng_host['text'].startswith('哼，'), heng_host['text']
    assert heng_host['translation'].startswith('Hmph,'), heng_host['translation']
    clerk = {
        'text': '嗯，先生，这款手机还有碎裂风险',
        'translation': 'Mm, sir, this phone can still crack.',
    }
    ensure_particle_translation_lead(clerk, 'English')
    assert clerk['translation'].startswith('Mm,'), clerk['translation']
    assert not clerk['translation'].lower().startswith('hmm, mm'), clerk['translation']
    assert collapse_double_particle_lead('Hmm, Mm, sir, hello.', 'English') == 'Mm, sir, hello.'
    print('vocal_particles overlap repair: ok')


if __name__ == '__main__':
    _smoke()
