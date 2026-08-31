from collections import Counter
import json
import os
import re

import librosa
from loguru import logger
import numpy as np

from .utils import save_wav
from .step044_tts_edge_tts import tts as edge_tts
from .step045_tts_openai import tts as openai_tts
from .step046_tts_fish import (
    tts as fish_tts,
    apply_emotion_tags,
    assign_speaker_voices,
    fish_clone_pending,
    spoken_text_for_timing,
    split_mixed_pitch_speakers,
    voice_for_method,
)
from tools.target_language import is_asr_junk, tts_language
from tools.tts_mix import (
    assemble_dub_timeline,
    remix_combined_audio,
    restitch_tts_timeline,
    write_combined_mix,
    _load_vocals,
)
from tools.tts_speech import preprocess_text

tts_support_languages = {
    # XTTS-v2 supports 17 languages: English (en), Spanish (es), French (fr), German (de), Italian (it), Portuguese (pt), Polish (pl), Turkish (tr), Russian (ru), Dutch (nl), Czech (cs), Arabic (ar), Chinese (zh-cn), Japanese (ja), Hungarian (hu), Korean (ko) Hindi (hi).
    'xtts': ['中文', 'English', 'Japanese', 'Korean', 'French', 'Polish', 'Spanish'],
    'bytedance': [],
    'GPTSoVits': [],
    'EdgeTTS': ['中文', 'English', 'Japanese', 'Korean', 'French', 'Polish', 'Spanish', '粤语', 'Vietnamese', 'Thai', 'Indonesian', 'Malay', 'Filipino'],
    'OpenAI': ['中文', 'English', 'Japanese', 'Korean', 'French', 'Polish', 'Spanish', '粤语', 'Vietnamese', 'Thai', 'Indonesian', 'Malay', 'Filipino'],
    'Fish': ['中文', '粤语', 'English', 'Japanese', 'Korean', 'French', 'Polish', 'Spanish', 'Vietnamese', 'Thai', 'Indonesian', 'Malay', 'Filipino'],
    # zero_shot usage, <|zh|><|en|><|jp|><|yue|><|ko|> for Chinese/English/Japanese/Cantonese/Korean
    'cosyvoice': ['中文', '粤语', 'English', 'Japanese', 'Korean', 'French'],
}

def generate_wavs(method, folder, target_language='中文', voice = 'zh-CN-XiaoxiaoNeural'):
    assert method in ['xtts', 'bytedance', 'cosyvoice', 'EdgeTTS', 'OpenAI', 'Fish']
    transcript_path = os.path.join(folder, 'translation.json')
    output_folder = os.path.join(folder, 'wavs')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    with open(transcript_path, 'r', encoding='utf-8') as f:
        transcript = json.load(f)
    from tools.vocal_particles import (
        ensure_particle_translation_lead,
        is_particle_card,
        particle_translation,
        reclassify_particles_from_vocals,
        recover_vocal_particles,
        repair_overlapping_particle_cards,
    )
    transcript = recover_vocal_particles(folder, transcript, isolated_glyphs={'哼'})
    transcript = reclassify_particles_from_vocals(folder, transcript, target_language)
    transcript = repair_overlapping_particle_cards(transcript, target_language)
    for line in transcript:
        if is_particle_card(line.get('text')) and not (line.get('translation') or '').strip():
            line['translation'] = particle_translation(line.get('text'), target_language)
        ensure_particle_translation_lead(line, target_language)
    from tools.step030_translation import repair_shared_orig_windows
    transcript = repair_shared_orig_windows(transcript)
    from tools.target_language import speakers_are_locked, spoken_char_pace, spoken_word_pace, uses_char_budget
    split_now = False
    if not speakers_are_locked(folder):
        transcript, split_now = split_mixed_pitch_speakers(folder, transcript)
        if split_now:
            from .step020_asr import generate_speaker_audio
            generate_speaker_audio(folder, transcript)
            with open(transcript_path, 'w', encoding='utf-8') as handle:
                json.dump(transcript, handle, indent=2, ensure_ascii=False)
            logger.info('已依音高拆開混雜說話人，稍後會分開克隆')
    source_counts = Counter((line.get('text') or '') for line in transcript)
    speakers = set()
    
    for line in transcript:
        speakers.add(line['speaker'])
    num_speakers = len(speakers)
    logger.info(f'Found {num_speakers} speakers')

    if target_language not in tts_support_languages[method]:
        logger.error(f'{method} does not support {target_language}')
        return f'{method} does not support {target_language}'

    voices, cloned_now = assign_speaker_voices(
        folder, transcript, target_language, clone=(method == 'Fish'),
    )
    if cloned_now or split_now:
        from tools.target_language import clear_tts_cache
        logger.info('已為角色建立 Fish 克隆，重新合成配音')
        clear_tts_cache(folder)
        os.makedirs(output_folder, exist_ok=True)
    jobs = []
    for i, line in enumerate(transcript):
        speaker = line['speaker']
        text = preprocess_text(line.get('translation') or '', target_language)
        if not text.strip() and tts_language(target_language) in ('中文', '粤语'):
            text = preprocess_text(line.get('text') or '', target_language)
        source_text = line.get('text') or ''
        output_path = os.path.join(output_folder, f'{str(i).zfill(4)}.wav')
        if is_asr_junk(source_text) or not (text or '').strip():
            silence = np.zeros((max(8, int(0.08 * 24000)),), dtype=np.float32)
            save_wav(silence, output_path)
            logger.info(f'跳過無詞配音：#{i}')
            continue
        tts_text = apply_emotion_tags(
            text,
            source_text,
            source_is_shared=source_counts[source_text] > 1,
            target_language=target_language,
        ) if method == 'Fish' else text
        emotion_shown = tts_text.replace('[in Japanese]', '').strip()
        if emotion_shown != text:
            logger.info(f'情绪配音: {emotion_shown}')
        speaker_wav = os.path.join(folder, 'SPEAKER', f'{speaker}.wav')
        speaker_voice = voice_for_method(voices, speaker, method, ui_voice=voice)
        slot = float(line.get('orig_end') or line.get('end') or 0) - float(line.get('orig_start') or line.get('start') or 0)
        spoken = spoken_text_for_timing(tts_text)
        tts_speed = 1.0
        if uses_char_budget(target_language):
            units = max(1, len(re.sub(r'\s+', '', spoken)))
            pace = max(3.0, spoken_char_pace(target_language))
        else:
            units = max(1, len(spoken.split()))
            pace = max(1.2, spoken_word_pace(target_language))
        if slot > 0.25 and units / pace > slot * 1.08:
            tts_speed = float(min(1.12, (units / pace) / slot))
        jobs.append({
            'i': i,
            'line': line,
            'text': text,
            'tts_text': tts_text,
            'output_path': output_path,
            'speaker_wav': speaker_wav,
            'tts_speed': tts_speed,
            'speaker_voice': speaker_voice,
        })

    def _synth(job):
        output_path = job['output_path']
        text = job['text']
        tts_text = job['tts_text']
        speaker_wav = job['speaker_wav']
        tts_speed = job['tts_speed']
        speaker_voice = job['speaker_voice']
        if method == 'bytedance':
            from .step041_tts_bytedance import tts as bytedance_tts
            bytedance_tts(text, output_path, speaker_wav, target_language=target_language)
        elif method == 'xtts':
            from .step042_tts_xtts import tts as xtts_tts
            xtts_tts(text, output_path, speaker_wav, target_language=target_language)
        elif method == 'cosyvoice':
            from .step043_tts_cosyvoice import tts as cosyvoice_tts
            cosyvoice_tts(text, output_path, speaker_wav, target_language=target_language)
        elif method == 'EdgeTTS':
            edge_tts(text, output_path, target_language=target_language, voice=speaker_voice, speed=tts_speed)
        elif method == 'OpenAI':
            openai_tts(text, output_path, target_language=target_language, voice=speaker_voice, speed=tts_speed)
        elif method == 'Fish':
            fish_tts(tts_text, output_path, target_language=target_language, voice=speaker_voice, speed=tts_speed)
        return job['i']

    if method in ('Fish', 'OpenAI', 'EdgeTTS'):
        from tools.api_keys import workers_for_keys
        from tools.parallel import env_workers, map_parallel
        if method == 'Fish':
            workers, n_keys = workers_for_keys('FISH_TTS_WORKERS', 'FISH_API_KEY')
        elif method == 'OpenAI':
            workers, n_keys = workers_for_keys('TTS_WORKERS', 'OPENAI_API_KEY', default_per_key=2)
        else:
            workers, n_keys = env_workers('TTS_WORKERS', 2), 1
        extra = f'、{n_keys} 把 API key' if n_keys > 1 else ''
        logger.info(f'{method} TTS 併發 {workers}{extra}')
        map_parallel(_synth, jobs, workers)
    else:
        for job in jobs:
            _synth(job)

    wav_paths = [
        os.path.join(output_folder, f'{str(i).zfill(4)}.wav')
        for i in range(len(transcript))
    ]
    full_wav = assemble_dub_timeline(transcript, wav_paths, vocals=_load_vocals(folder))
    save_wav(full_wav, os.path.join(folder, 'audio_tts.wav'))
    with open(transcript_path, 'w', encoding='utf-8') as f:
        json.dump(transcript, f, indent=2, ensure_ascii=False)

    instruments_wav, sr = librosa.load(os.path.join(folder, 'audio_instruments.wav'), sr=24000)
    write_combined_mix(folder, full_wav, instruments_wav, target_language)
    logger.info(f'Generated {os.path.join(folder, "audio_combined.wav")}')
    return os.path.join(folder, 'audio_combined.wav'), os.path.join(folder, 'audio.wav')

def generate_all_wavs_under_folder(root_folder, method, target_language='中文', voice = 'zh-CN-XiaoxiaoNeural'):
    from tools.target_language import clear_tts_cache, layout_cache_ok, mix_cache_ok, tts_cache_ok
    wav_combined, wav_ori = None, None
    for root, dirs, files in os.walk(root_folder):
        if 'translation.json' not in files:
            continue
        clone_pending = False
        if method == 'Fish':
            with open(os.path.join(root, 'translation.json'), 'r', encoding='utf-8') as handle:
                transcript = json.load(handle)
            clone_pending = fish_clone_pending(root, transcript)
        if clone_pending:
            logger.info(f'尚未克隆角色聲線，將重新合成 {root}')
            clear_tts_cache(root)
        elif not tts_cache_ok(root, target_language):
            if 'audio_combined.wav' in files:
                logger.info(f'目標語言改為 {target_language}，清除舊配音後重合成')
                clear_tts_cache(root)
        if tts_cache_ok(root, target_language) and not clone_pending:
            if layout_cache_ok(root) and mix_cache_ok(root):
                wav_combined, wav_ori = os.path.join(root, 'audio_combined.wav'), os.path.join(root, 'audio.wav')
                logger.info(f'Wavs already generated in {root}')
            elif layout_cache_ok(root) and remix_combined_audio(root, target_language):
                wav_combined, wav_ori = os.path.join(root, 'audio_combined.wav'), os.path.join(root, 'audio.wav')
            elif restitch_tts_timeline(root, target_language):
                wav_combined, wav_ori = os.path.join(root, 'audio_combined.wav'), os.path.join(root, 'audio.wav')
            else:
                wav_combined, wav_ori = generate_wavs(method, root, target_language, voice)
        else:
            wav_combined, wav_ori = generate_wavs(method, root, target_language, voice)
    return f'Generated all wavs under {root_folder}', wav_combined, wav_ori

if __name__ == '__main__':
    folder = r'videos/村长台钓加拿大/20240805 英文无字幕 阿里这小子在水城威尼斯发来问候'
    generate_wavs('xtts', folder)
