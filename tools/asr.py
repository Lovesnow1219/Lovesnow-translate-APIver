
import os
import numpy as np
from dotenv import load_dotenv
from .utils import save_wav
from tools.target_language import drop_asr_junk_lines, is_asr_junk
import json
import librosa
from loguru import logger
load_dotenv()

def _none_if_auto(value):
    if value in (None, 'auto', 'None', 'none', ''):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None

def merge_segments(transcript, ending='!"\').:;?]}~！“”’）。：；？】'):
    merged_transcription = []
    buffer_segment = None

    for segment in transcript:
        if not (segment.get('text') or '').strip() or is_asr_junk(segment.get('text')):
            continue
        if buffer_segment is None:
            buffer_segment = dict(segment)
            continue
        same_speaker = str(buffer_segment.get('speaker')) == str(segment.get('speaker'))
        gap = float(segment.get('start') or 0) - float(buffer_segment.get('end') or 0)
        combined_duration = float(segment.get('end') or 0) - float(buffer_segment.get('start') or 0)
        too_long = combined_duration > 8.0 or len(buffer_segment.get('text') or '') > 80
        if (
            (not same_speaker)
            or gap > 0.2
            or (not buffer_segment['text'] or buffer_segment['text'][-1] in ending)
            or too_long
        ):
            merged_transcription.append(buffer_segment)
            buffer_segment = dict(segment)
        else:
            buffer_segment['text'] += ' ' + segment['text']
            buffer_segment['end'] = segment['end']

    if buffer_segment is not None:
        merged_transcription.append(buffer_segment)

    return merged_transcription

def generate_speaker_audio(folder, transcript):
    wav_path = os.path.join(folder, 'audio_vocals.wav')
    audio_data, samplerate = librosa.load(wav_path, sr=24000)
    speaker_dict = dict()
    length = len(audio_data)
    delay = 0.02
    for segment in transcript:
        start = max(0, int((segment['start'] - delay) * samplerate))
        end = min(int((segment['end']+delay) * samplerate), length)
        speaker_segment_audio = audio_data[start:end]
        speaker_dict[segment['speaker']] = np.concatenate((speaker_dict.get(
            segment['speaker'], np.zeros((0, ))), speaker_segment_audio))

    speaker_folder = os.path.join(folder, 'SPEAKER')
    if not os.path.exists(speaker_folder):
        os.makedirs(speaker_folder)
    
    for speaker, audio in speaker_dict.items():
        speaker_file_path = os.path.join(
            speaker_folder, f"{speaker}.wav")
        save_wav(audio, speaker_file_path)


def transcribe_audio(method, folder, model_name: str = 'large', download_root='models/ASR/whisper', device='auto', batch_size=32, diarization=True,min_speakers=None, max_speakers=None, language=None):
    from tools.target_language import asr_language_code, clear_asr_downstream, load_dub_meta, save_dub_meta

    lang = asr_language_code(language)
    transcript_path = os.path.join(folder, 'transcript.json')
    recorded = (load_dub_meta(folder) or {}).get('asr_language')
    recorded = asr_language_code(recorded) if recorded else ('zh' if os.path.exists(transcript_path) else None)
    if os.path.exists(transcript_path) and recorded and recorded != lang:
        logger.info(f'原片語言 {recorded}→{lang}，重跑識別：{folder}')
        clear_asr_downstream(folder, keep_bible=True)
    if os.path.exists(transcript_path):
        logger.info(f'Transcript already exists in {folder}')
        return True
    
    wav_path = os.path.join(folder, 'audio_vocals.wav')
    if not os.path.exists(wav_path):
        return False
    
    logger.info(f'Transcribing {wav_path} language={lang}')
    diarization = bool(diarization)
    min_speakers = _none_if_auto(min_speakers)
    max_speakers = _none_if_auto(max_speakers)
    
    if method != 'OpenAI':
        logger.warning(f'本機 ASR（{method}）已移除，改用 OpenAI')
    from .asr_openai import openai_transcribe_audio
    transcript = openai_transcribe_audio(wav_path, model_name, language=lang)

    transcript = merge_segments(transcript)
    from tools.asr_source import split_long_segments
    transcript = split_long_segments(transcript)
    transcript, junk_n = drop_asr_junk_lines(transcript)
    if junk_n:
        logger.info(f'已丟掉 {junk_n} 條辨識幻聽（重複單字噪音）')
    from tools.vocal_particles import recover_vocal_particles
    transcript = recover_vocal_particles(folder, transcript)
    from tools.asr_gaps import recover_missing_speech
    transcript = recover_missing_speech(folder, transcript, language=lang)
    with open(os.path.join(folder, 'transcript.json'), 'w', encoding='utf-8') as f:
        json.dump(transcript, f, indent=4, ensure_ascii=False)
    logger.info(f'Transcribed {wav_path} successfully, and saved to {os.path.join(folder, "transcript.json")}')
    try:
        from tools.asr_review import polish_asr_segmentation
        transcript = polish_asr_segmentation(folder, transcript=transcript)
    except Exception as exc:
        from tools.job_control import JobStopped
        if isinstance(exc, JobStopped):
            raise
        logger.warning(f'辨識後斷句略過：{exc}')
    generate_speaker_audio(folder, transcript)
    save_dub_meta(folder, asr_language=lang)
    return transcript

def transcribe_all_audio_under_folder(folder, asr_method, whisper_model_name: str = 'large', diarization=False, batch_size=32, device='auto', min_speakers=None, max_speakers=None, language=None):
    from tools.target_language import asr_language_code, load_dub_meta

    transcribe_json = None
    wanted = asr_language_code(language)
    for root, dirs, files in os.walk(folder):
        dirs[:] = [name for name in dirs if name not in {'translations', 'wavs', 'SPEAKER', '__pycache__', '_demucs_chunks'}]
        recorded = (load_dub_meta(root) or {}).get('asr_language')
        if not recorded and 'transcript.json' in files:
            recorded = 'zh'
        lang_changed = bool(recorded and recorded != wanted)
        if 'audio_vocals.wav' in files and ('transcript.json' not in files or lang_changed):
            transcribe_json = transcribe_audio(asr_method, root, whisper_model_name, 'models/ASR/whisper', device, batch_size, diarization, min_speakers, max_speakers, language=language)
        elif 'transcript.json' in files:
            transcribe_json = json.load(open(os.path.join(root, 'transcript.json'), 'r', encoding='utf-8'))
            from tools.asr_repair import already_repaired
            if already_repaired(root):
                continue
            before = len(transcribe_json)
            from tools.vocal_particles import recover_vocal_particles
            from tools.asr_gaps import recover_missing_speech
            from tools.asr_source import cleanup_asr_source
            filled = recover_missing_speech(root, recover_vocal_particles(root, transcribe_json), language=wanted)
            for line in filled:
                line['text'] = cleanup_asr_source(line.get('text'))
            if len(filled) != before or any(
                (a.get('text') or '') != (b.get('text') or '')
                for a, b in zip(transcribe_json, filled)
            ):
                with open(os.path.join(root, 'transcript.json'), 'w', encoding='utf-8') as handle:
                    json.dump(filled, handle, indent=4, ensure_ascii=False)
                logger.info(f'已補漏句並寫回 {root}')
            transcribe_json = filled
            try:
                from tools.asr_review import polish_asr_segmentation
                transcribe_json = polish_asr_segmentation(root, transcript=transcribe_json)
            except Exception as exc:
                from tools.job_control import JobStopped
                if isinstance(exc, JobStopped):
                    raise
                logger.warning(f'辨識後斷句略過：{exc}')
    return f'Transcribed all audio under {folder}', transcribe_json

if __name__ == '__main__':
    _, transcribe_json = transcribe_all_audio_under_folder('videos', 'OpenAI')
    print(transcribe_json)