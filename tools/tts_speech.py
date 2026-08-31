# -*- coding: utf-8 -*-
"""Expand numbers and symbols so TTS can read them."""
import re

from .cn_tx import TextNorm
from tools.target_language import tts_language

normalizer = TextNorm()

_FULLWIDTH_DIGITS = str.maketrans('０１２３４５６７８９', '0123456789')
_JA_ONES = ('ゼロ', 'いち', 'に', 'さん', 'よん', 'ご', 'ろく', 'なな', 'はち', 'きゅう')
_JA_HUNDREDS = {
    1: 'ひゃく', 2: 'にひゃく', 3: 'さんびゃく', 4: 'よんひゃく',
    5: 'ごひゃく', 6: 'ろっぴゃく', 7: 'ななひゃく', 8: 'はっぴゃく', 9: 'きゅうひゃく',
}
_JA_THOUSANDS = {
    1: 'せん', 2: 'にせん', 3: 'さんぜん', 4: 'よんせん',
    5: 'ごせん', 6: 'ろくせん', 7: 'ななせん', 8: 'はっせん', 9: 'きゅうせん',
}
# Kanji that Fish often skips when the clone is Chinese (続=续, 話=话).
_JA_KANA_WORDS = (
    ('続けて', 'つづけて'),
    ('続ける', 'つづける'),
    ('続き', 'つづき'),
    ('続け', 'つづけ'),
    ('話して', 'はなして'),
    ('話した', 'はなした'),
    ('話せる', 'はなせる'),
    ('話せ', 'はなせ'),
    ('話す', 'はなす'),
    ('話し', 'はなし'),
)


def _int_to_ja(value):
    number = int(value)
    if number < 0:
        return 'マイナス' + _int_to_ja(-number)
    if number < 10:
        return _JA_ONES[number]
    if number == 10:
        return 'じゅう'
    if number < 20:
        return 'じゅう' + _JA_ONES[number - 10]
    if number < 100:
        tens, ones = divmod(number, 10)
        head = 'じゅう' if tens == 1 else _JA_ONES[tens] + 'じゅう'
        return head + (_JA_ONES[ones] if ones else '')
    if number < 1000:
        hundreds, rest = divmod(number, 100)
        return _JA_HUNDREDS[hundreds] + (_int_to_ja(rest) if rest else '')
    if number < 10000:
        thousands, rest = divmod(number, 1000)
        return _JA_THOUSANDS[thousands] + (_int_to_ja(rest) if rest else '')
    if number < 100000000:
        man, rest = divmod(number, 10000)
        return _int_to_ja(man) + 'まん' + (_int_to_ja(rest) if rest else '')
    return ''.join(_JA_ONES[int(ch)] if ch.isdigit() else ch for ch in str(number))


def _number_token_to_ja(token):
    if '.' in token:
        left, right = token.split('.', 1)
        return _int_to_ja(int(left or '0')) + 'てん' + ''.join(
            _JA_ONES[int(ch)] for ch in right if ch.isdigit()
        )
    return _int_to_ja(token)


def expand_japanese_speech(text):
    """Fish skips Arabic digits, slashes, and % when Japanese normalize is off."""
    line = (text or '').translate(_FULLWIDTH_DIGITS)
    for src, spoken in _JA_KANA_WORDS:
        line = line.replace(src, spoken)
    line = re.sub(r'[\u4e00-\u9fff]+[（(]([ァ-ヶー]+)[）)]', r'\1', line)
    line = line.replace('％', '%').replace('＋', '+')
    line = re.sub(r'([0-9]+(?:\.[0-9]+)?)%', r'\1パーセント', line)
    line = re.sub(r'\+([0-9]+(?:\.[0-9]+)?)', r'プラス\1', line)
    line = re.sub(r'パーセント/', 'パーセント、', line)
    line = re.sub(r'([0-9]+(?:\.[0-9]+)?)/([0-9]+(?:\.[0-9]+)?)', r'\1の\2', line)
    line = re.sub(r'[0-9]+(?:\.[0-9]+)?', lambda match: _number_token_to_ja(match.group(0)), line)
    line = line.replace('…', '、').replace('...', '、')
    line = re.sub(r'[―—–]+', '、', line)
    return re.sub(r'\s+', ' ', line).strip()


_VI_ONES = ('không', 'một', 'hai', 'ba', 'bốn', 'năm', 'sáu', 'bảy', 'tám', 'chín')


def _int_to_vi(value):
    number = int(value)
    if number < 0:
        return 'âm ' + _int_to_vi(-number)
    if number < 10:
        return _VI_ONES[number]
    if number == 10:
        return 'mười'
    if number < 20:
        rest = number - 10
        if rest == 5:
            return 'mười lăm'
        return 'mười ' + ('mốt' if rest == 1 else _VI_ONES[rest])
    if number < 100:
        tens, ones = divmod(number, 10)
        head = _VI_ONES[tens] + ' mươi'
        if ones == 0:
            return head
        if ones == 1:
            return head + ' mốt'
        if ones == 5:
            return head + ' lăm'
        return head + ' ' + _VI_ONES[ones]
    if number < 1000:
        hundreds, rest = divmod(number, 100)
        head = _VI_ONES[hundreds] + ' trăm'
        return head if not rest else head + ' ' + _int_to_vi(rest)
    if number < 10000:
        thousands, rest = divmod(number, 1000)
        head = _VI_ONES[thousands] + ' nghìn'
        return head if not rest else head + ' ' + _int_to_vi(rest)
    if number < 1000000:
        thousand, rest = divmod(number, 1000)
        head = _int_to_vi(thousand) + ' nghìn'
        return head if not rest else head + ' ' + _int_to_vi(rest)
    return ' '.join(_VI_ONES[int(ch)] for ch in str(number) if ch.isdigit())


def _number_token_to_vi(token):
    if '.' in token:
        left, right = token.split('.', 1)
        return _int_to_vi(int(left or '0')) + ' phẩy ' + ' '.join(
            _VI_ONES[int(ch)] for ch in right if ch.isdigit()
        )
    return _int_to_vi(token)


def expand_english_speech(text):
    line = (text or '').translate(_FULLWIDTH_DIGITS)
    line = line.replace('％', '%').replace('＋', '+')
    line = re.sub(r'([0-9]+(?:\.[0-9]+)?)%', r'\1 percent', line)
    line = re.sub(r'\+([0-9]+(?:\.[0-9]+)?)', r'plus \1', line)
    line = re.sub(r'percent/', 'percent, ', line)
    line = re.sub(r'([0-9]+(?:\.[0-9]+)?)/([0-9]+(?:\.[0-9]+)?)', r'\1 of \2', line)
    line = line.replace('…', ', ').replace('...', ', ')
    line = re.sub(r'[―—–]+', ', ', line)
    return re.sub(r'\s+', ' ', line).strip()


def expand_vietnamese_speech(text):
    line = (text or '').translate(_FULLWIDTH_DIGITS)
    line = line.replace('％', '%').replace('＋', '+')
    line = re.sub(r'([0-9]+(?:\.[0-9]+)?)%', r'\1 phần trăm', line)
    line = re.sub(r'\+([0-9]+(?:\.[0-9]+)?)', r'cộng \1', line)
    line = re.sub(r'phần trăm/', 'phần trăm, ', line)
    line = re.sub(r'([0-9]+(?:\.[0-9]+)?)/([0-9]+(?:\.[0-9]+)?)', r'\1 trên \2', line)
    line = re.sub(r'[0-9]+(?:\.[0-9]+)?', lambda match: _number_token_to_vi(match.group(0)), line)
    line = line.replace('…', ', ').replace('...', ', ')
    line = re.sub(r'[―—–]+', ', ', line)
    return re.sub(r'\s+', ' ', line).strip()


def preprocess_text(text, target_language='中文'):
    text = text or ''
    if str(target_language) in ('中文', '粤语', 'Cantonese'):
        text = text.replace('AI', '人工智能')
        text = re.sub(r'(?<!^)([A-Z])', r' \1', text)
        text = normalizer(text)
        text = re.sub(r'(?<=[a-zA-Z])(?=\d)|(?<=\d)(?=[a-zA-Z])', ' ', text)
        return text
    lang = tts_language(target_language)
    if lang == 'Japanese':
        return expand_japanese_speech(text)
    if lang == 'English':
        return expand_english_speech(text)
    if lang == 'Vietnamese':
        return expand_vietnamese_speech(text)
    return re.sub(r'\s+', ' ', text).strip()

