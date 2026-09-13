from collections import Counter
import json
import os

import librosa
from loguru import logger
import numpy as np

from .utils import save_wav
from .tts_fish import (
    tts as fish_tts,
    apply_emotion_tags,
    assign_speaker_voices,
    fish_clone_pending,
    reassign_crossed_pitch_speakers,
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
    'Fish': ['中文', '粤语', 'English', 'Japanese', 'Korean', 'French', 'Polish', 'Spanish', 'Vietnamese', 'Thai', 'Indonesian', 'Malay', 'Filipino'],
}

def generate_wavs(method, folder, target_language='中文', voice=None):
    from tools.dubbing_settings import load_delivery_settings, stamp_render
    from tools.dubbing_timing import write_timing_report
    delivery = load_delivery_settings(folder)
    method = method or 'Fish'
    if method != 'Fish':
        raise ValueError('目前只支援 Fish 配音')
    from tools.translation_versions import activate_language, has_version, snapshot_active
    snapshot_active(folder)
    if not has_version(folder, target_language):
        raise FileNotFoundError(f'沒有 {target_language} 譯文：{folder}')
    activate_language(folder, target_language)
    transcript_path = os.path.join(folder, 'translation.json')
    output_folder = os.path.join(folder, 'wavs')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    with open(transcript_path, 'r', encoding='utf-8') as f:
        transcript = json.load(f)
    from tools.asr import ensure_speaker_audio
    ensure_speaker_audio(folder, transcript)
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
    from tools.line_roles import normalize_line_roles
    transcript = normalize_line_roles(transcript)
    from tools.line_roles import promote_bgm_speech
    promote_bgm_speech(transcript)
    for line in transcript:
        if is_particle_card(line.get('text')) and not (line.get('translation') or '').strip():
            line['translation'] = particle_translation(line.get('text'), target_language)
        ensure_particle_translation_lead(line, target_language)
    from tools.translation import repair_shared_orig_windows
    transcript = repair_shared_orig_windows(transcript)
    from tools.target_language import speakers_are_locked
    split_now = False
    from tools.asr_repair import already_repaired
    if not speakers_are_locked(folder) and not already_repaired(folder):
        transcript, moved = reassign_crossed_pitch_speakers(folder, transcript)
        if moved:
            split_now = True
            logger.info('已依音高把錯標的句子還給另一個主角')
        else:
            transcript, split_now = split_mixed_pitch_speakers(folder, transcript)
            if split_now:
                logger.info('已依音高拆開混雜說話人，稍後會分開克隆')
        if split_now:
            from .asr import generate_speaker_audio
            generate_speaker_audio(folder, transcript)
            from tools.translation_versions import write_translation
            write_translation(folder, transcript, target_language)
    source_counts = Counter((line.get('text') or '') for line in transcript)
    speakers = set()
    
    for line in transcript:
        speakers.add(line['speaker'])
    num_speakers = len(speakers)
    logger.info(f'Found {num_speakers} speakers')

    if tts_language(target_language) not in tts_support_languages[method]:
        logger.error(f'{method} does not support {target_language}')
        raise ValueError(f'{method} does not support {target_language}')

    voices, cloned_now = assign_speaker_voices(
        folder, transcript, target_language, clone=True,
    )
    if cloned_now or split_now:
        from tools.target_language import clear_tts_cache
        logger.info('已更新角色聲線，重新合成配音')
        clear_tts_cache(folder)
        os.makedirs(output_folder, exist_ok=True)
    from tools.line_roles import should_skip_dub
    from tools.source_vocalizations import preserve_laughter
    source_vocals = _load_vocals(folder)

    def _line_job(i, line):
        speaker = line['speaker']
        text = preprocess_text(line.get('translation') or '', target_language)
        if not text.strip() and tts_language(target_language) in ('中文', '粤语'):
            text = preprocess_text(line.get('text') or '', target_language)
        source_text = line.get('text') or ''
        output_path = os.path.join(output_folder, f'{str(i).zfill(4)}.wav')
        if not should_skip_dub(line) and preserve_laughter(folder, line, output_path, source_vocals):
            logger.info(f'保留原片純笑聲與時長：#{i}')
            return None
        line.pop('audio_source', None)
        if is_asr_junk(source_text) or should_skip_dub(line) or not (text or '').strip():
            silence = np.zeros((max(8, int(0.08 * 24000)),), dtype=np.float32)
            save_wav(silence, output_path)
            logger.info(f'跳過無詞配音：#{i}')
            return None
        tts_text = apply_emotion_tags(
            text,
            source_text,
            source_is_shared=source_counts[source_text] > 1,
            target_language=target_language,
            speaker=speaker,
        )
        emotion_shown = tts_text.replace('[in Japanese]', '').strip()
        if emotion_shown != text:
            logger.info(f'情绪配音: {emotion_shown}')
        speaker_wav = os.path.join(folder, 'SPEAKER', f'{speaker}.wav')
        speaker_voice = voice_for_method(voices, speaker, method, ui_voice=voice)
        # One deliberate speed for the whole episode; never accelerate each
        # sentence from a word-count estimate and then delay it again.
        tts_speed = delivery['tts_speed']
        return {
            'i': i,
            'line': line,
            'text': text,
            'tts_text': tts_text,
            'output_path': output_path,
            'speaker_wav': speaker_wav,
            'tts_speed': tts_speed,
            'speaker_voice': speaker_voice,
        }

    jobs = []
    for i, line in enumerate(transcript):
        job = _line_job(i, line)
        if job:
            jobs.append(job)

    def _synth(job):
        output_path = job['output_path']
        tts_text = job['tts_text']
        tts_speed = job['tts_speed']
        speaker_voice = job['speaker_voice']
        fish_tts(tts_text, output_path, target_language=target_language, voice=speaker_voice, speed=tts_speed)
        return job['i']

    from tools.api_keys import workers_for_keys
    from tools.parallel import map_parallel
    workers, n_keys = workers_for_keys('FISH_TTS_WORKERS', 'FISH_API_KEY')
    extra = f'、{n_keys} 把 API key' if n_keys > 1 else ''
    logger.info(f'Fish TTS 併發 {workers}{extra}')
    map_parallel(_synth, jobs, workers)

    from tools.job_control import check_stop
    from tools.translation_versions import write_translation
    write_translation(folder, transcript, target_language)
    check_stop()
    # Measure approved dialogue as spoken. Timing repairs go through the main
    # reviewer below, never an unreviewed word-budget rewrite after approval.

    wav_paths = [
        os.path.join(output_folder, f'{str(i).zfill(4)}.wav')
        for i in range(len(transcript))
    ]
    full_wav = assemble_dub_timeline(transcript, wav_paths, vocals=_load_vocals(folder))
    write_timing_report(folder, transcript)
    save_wav(full_wav, os.path.join(folder, 'audio_tts.wav'))
    write_translation(folder, transcript, target_language)

    instruments_wav, sr = librosa.load(os.path.join(folder, 'audio_instruments.wav'), sr=24000)
    write_combined_mix(folder, full_wav, instruments_wav, target_language)
    logger.info(f'Generated {os.path.join(folder, "audio_combined.wav")}')

    from tools.target_language import is_chinese_target
    if not is_chinese_target(target_language):
        from tools.translation_review import review_and_redub_after_tts, stamp_after_dub
        check_stop()
        if review_and_redub_after_tts(folder, target_language):
            with open(transcript_path, 'r', encoding='utf-8') as handle:
                transcript = json.load(handle)
            redo_after = []
            for i, line in enumerate(transcript):
                output_path = os.path.join(output_folder, f'{str(i).zfill(4)}.wav')
                if os.path.isfile(output_path):
                    continue
                job = _line_job(i, line)
                if job:
                    redo_after.append(job)
            if redo_after:
                logger.info(f'配音後審稿已改譯，重合成 {len(redo_after)} 句')
                map_parallel(_synth, redo_after, workers)
            wav_paths = [
                os.path.join(output_folder, f'{str(i).zfill(4)}.wav')
                for i in range(len(transcript))
            ]
            full_wav = assemble_dub_timeline(transcript, wav_paths, vocals=_load_vocals(folder))
            write_timing_report(folder, transcript)
            save_wav(full_wav, os.path.join(folder, 'audio_tts.wav'))
            write_translation(folder, transcript, target_language)
            instruments_wav, sr = librosa.load(os.path.join(folder, 'audio_instruments.wav'), sr=24000)
            write_combined_mix(folder, full_wav, instruments_wav, target_language)
            logger.info(f'配音後審稿已重混: {os.path.join(folder, "audio_combined.wav")}')
            stamp_after_dub(folder, target_language)

    stamp_render(folder, target_language, voice)
    return os.path.join(folder, 'audio_combined.wav'), os.path.join(folder, 'audio.wav')

def generate_all_wavs_under_folder(root_folder, method='Fish', target_language='中文', voice=None, tts_speed=None):
    from tools.dubbing_settings import save_delivery_settings, stamp_render
    from tools.target_language import clear_tts_cache, layout_cache_ok, mix_cache_ok, tts_cache_ok
    wav_combined, wav_ori = None, None
    from tools.translation_versions import activate_language, has_any_version, has_version, snapshot_active
    for root, dirs, files in os.walk(root_folder):
        dirs[:] = [name for name in dirs if name not in {'translations', 'wavs', 'SPEAKER', '__pycache__'}]
        if 'translation.json' not in files and not has_any_version(root):
            continue
        snapshot_active(root)
        if not has_version(root, target_language):
            logger.info(f'沒有 {target_language} 譯文，略過配音：{root}')
            continue
        activate_language(root, target_language)
        if tts_speed is not None:
            save_delivery_settings(root, tts_speed)
        files = os.listdir(root)
        clone_pending = False
        if method == 'Fish':
            with open(os.path.join(root, 'translation.json'), 'r', encoding='utf-8') as handle:
                transcript = json.load(handle)
            clone_pending = fish_clone_pending(root, transcript)
        if clone_pending:
            logger.info(f'尚未鎖定角色聲線，將重新合成 {root}')
            clear_tts_cache(root)
        # A changed input invalidates the mix, but matching per-line WAVs can
        # still be reused. Fish validates each line's request fingerprint.
        if tts_cache_ok(root, target_language, voice) and not clone_pending:
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
        if wav_combined:
            stamp_render(root, target_language, voice)
    return f'Generated all wavs under {root_folder}', wav_combined, wav_ori

if __name__ == '__main__':
    generate_wavs('Fish', r'videos')
