import os
from loguru import logger
import asyncio
import edge_tts


language_map = {
    '中文': 'zh-CN-XiaoxiaoNeural',
    'English': 'en-US-MichelleNeural',
    'Japanese': 'ja-JP-NanamiNeural',
    '粤语': 'zh-HK-HiuMaanNeural',
    'Korean': 'ko-KR-SunHiNeural',
    'Spanish': 'es-ES-ElviraNeural',
    'French': 'fr-FR-DeniseNeural',
    'Polish': 'pl-PL-ZofiaNeural',
    'Vietnamese': 'vi-VN-HoaiMyNeural',
    'Thai': 'th-TH-PremwadeeNeural',
    'Indonesian': 'id-ID-GadisNeural',
    'Malay': 'ms-MY-YasminNeural',
    'Filipino': 'fil-PH-BlessicaNeural',
}


def tts(text, output_path, target_language='中文', voice='zh-CN-XiaoxiaoNeural', speed=1.0):
    if os.path.exists(output_path):
        logger.info(f'TTS {text} 已存在')
        return
    if not voice:
        voice = language_map.get(target_language, 'zh-CN-XiaoxiaoNeural')
    mp3_path = output_path.replace('.wav', '.mp3') if output_path.endswith('.wav') else output_path
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    for retry in range(3):
        try:
            rate_pct = int(round((float(speed or 1.0) - 1.0) * 100))
            rate = f'{rate_pct:+d}%'
            asyncio.run(edge_tts.Communicate(text, voice, rate=rate).save(mp3_path))
            logger.info(f'TTS {text}')
            return
        except Exception as e:
            logger.warning(f'TTS {text} 失败 ({retry + 1}/3)')
            logger.warning(e)
    raise RuntimeError(f'EdgeTTS 失败: {text}')


if __name__ == '__main__':
    speaker_wav = r'videos/村长台钓加拿大/20240805 英文无字幕 阿里这小子在水城威尼斯发来问候/audio_vocals.wav'
    while True:
        text = input('请输入：')
        tts(text, f'playground/{text}.wav', target_language='中文')
