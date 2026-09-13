"""Keep a pure laugh in its original timing instead of synthesizing it anew."""
import os
import re

import numpy as np

from tools.dubbing_settings import file_identity, read_json, write_json_atomic
from tools.vocal_particles import card_start, card_end


def is_laughter_card(text):
    compact = re.sub(r'[\s,，。.!！?？、…–—-]', '', str(text or '')).casefold()
    return bool(re.fullmatch(r'[哈呵嘻嘿]{2,}|(?:ha){2,}|(?:he){2,}|(?:heh){2,}|(?:ho){2,}', compact))


def preserve_laughter(folder, line, output_path, vocals, sample_rate=24000):
    if not is_laughter_card(line.get('text')) or vocals is None:
        return False
    start, end = card_start(line), card_end(line)
    if not 0 <= start < end <= len(vocals) / sample_rate:
        return False
    clip = np.asarray(vocals[round(start * sample_rate):round(end * sample_rate)], dtype=np.float32)
    if not clip.size or not np.isfinite(clip).all() or np.max(np.abs(clip)) <= .0005:
        return False
    meta_path = output_path + '.source.json'
    request = {'version': 1, 'source': file_identity(os.path.join(folder, 'audio_vocals.wav')),
               'start': start, 'end': end, 'sample_rate': sample_rate}
    cached = read_json(meta_path, {})
    if cached.get('request') != request or cached.get('output') != file_identity(output_path):
        from tools.utils import save_wav
        save_wav(clip, output_path)
        write_json_atomic(meta_path, {'request': request, 'output': file_identity(output_path)})
    line['audio_source'] = 'original_laughter'
    return True
