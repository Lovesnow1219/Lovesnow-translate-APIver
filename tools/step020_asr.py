
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


def transcribe_audio(method, folder, model_name: str = 'large', download_root='models/ASR/whisper', device='auto', batch_size=32, diarization=True,min_speakers=None, max_speakers=None):
    if os.path.exists(os.path.join(folder, 'transcript.json')):
        logger.info(f'Transcript already exists in {folder}')
        return True
    
    wav_path = os.path.join(folder, 'audio_vocals.wav')
    if not os.path.exists(wav_path):
        return False
    
    logger.info(f'Transcribing {wav_path}')
    diarization = bool(diarization)
    min_speakers = _none_if_auto(min_speakers)
    max_speakers = _none_if_auto(max_speakers)
    
    if method == 'OpenAI':
        from .step023_asr_openai import openai_transcribe_audio
        transcript = openai_transcribe_audio(wav_path, model_name)
    elif method in ('通义千问', 'Qwen', '阿里云-通义千问'):
        from .step024_asr_qwen import qwen_transcribe_audio
        transcript = qwen_transcribe_audio(wav_path, diarization=diarization)
    elif method == 'WhisperX':
        from .step021_asr_whisperx import whisperx_transcribe_audio
        transcript = whisperx_transcribe_audio(wav_path, model_name, download_root, device, batch_size, diarization, min_speakers, max_speakers)
    elif method == 'FunASR':
        from .step022_asr_funasr import funasr_transcribe_audio
        transcript = funasr_transcribe_audio(wav_path, device, batch_size, diarization)
    else:
        logger.error('Invalid ASR method')
        raise ValueError('Invalid ASR method')

    transcript = merge_segments(transcript)
    from tools.asr_source import split_long_segments
    transcript = split_long_segments(transcript)
    transcript, junk_n = drop_asr_junk_lines(transcript)
    if junk_n:
        logger.info(f'已丟掉 {junk_n} 條辨識幻聽（重複單字噪音）')
    from tools.vocal_particles import recover_vocal_particles
    transcript = recover_vocal_particles(folder, transcript)
    from tools.asr_gaps import recover_missing_speech
    transcript = recover_missing_speech(folder, transcript)
    with open(os.path.join(folder, 'transcript.json'), 'w', encoding='utf-8') as f:
        json.dump(transcript, f, indent=4, ensure_ascii=False)
    logger.info(f'Transcribed {wav_path} successfully, and saved to {os.path.join(folder, "transcript.json")}')
    generate_speaker_audio(folder, transcript)
    return transcript

def transcribe_all_audio_under_folder(folder, asr_method, whisper_model_name: str = 'large', diarization=False, batch_size=32, device='auto', min_speakers=None, max_speakers=None):
    transcribe_json = None
    for root, dirs, files in os.walk(folder):
        if 'audio_vocals.wav' in files and 'transcript.json' not in files:
            transcribe_json = transcribe_audio(asr_method, root, whisper_model_name, 'models/ASR/whisper', device, batch_size, diarization, min_speakers, max_speakers)
        elif 'transcript.json' in files:
            transcribe_json = json.load(open(os.path.join(root, 'transcript.json'), 'r', encoding='utf-8'))
            before = len(transcribe_json)
            from tools.vocal_particles import recover_vocal_particles
            from tools.asr_gaps import recover_missing_speech
            from tools.asr_source import cleanup_asr_source
            filled = recover_missing_speech(root, recover_vocal_particles(root, transcribe_json))
            for line in filled:
                line['text'] = cleanup_asr_source(line.get('text'))
            if len(filled) != before or any(
                (a.get('text') or '') != (b.get('text') or '')
                for a, b in zip(transcribe_json, filled)
            ):
                with open(os.path.join(root, 'transcript.json'), 'w', encoding='utf-8') as handle:
                    json.dump(filled, handle, indent=4, ensure_ascii=False)
                transcribe_json = filled
                logger.info(f'已補漏句並寫回 {root}')
    return f'Transcribed all audio under {folder}', transcribe_json

if __name__ == '__main__':
    _, transcribe_json = transcribe_all_audio_under_folder('videos', 'WhisperX')
    print(transcribe_json)
    # _, transcribe_json = transcribe_all_audio_under_folder('videos', 'FunASR')    
    # print(transcribe_json)