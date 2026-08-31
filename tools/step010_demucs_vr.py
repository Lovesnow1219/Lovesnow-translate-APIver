import os
import time
import gc
from loguru import logger
from dotenv import load_dotenv
from .utils import save_wav, normalize_wav
from .step011_replicate_demucs import separate_with_replicate, use_replicate

load_dotenv()

try:
    import torch
    auto_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
except Exception:
    torch = None
    auto_device = 'cpu'

# 全域变量
separator = None
model_loaded = False  # 新增標記，追蹤模型是否已載入
current_model_config = {}  # 新增变量，儲存目前載入模型的設定


def init_demucs():
    """
    初始化Demucs模型。
    如果模型已經初始化，直接返回而不重新載入。
    """
    global separator, model_loaded
    if not model_loaded:
        separator = load_model()
        model_loaded = True
    else:
        logger.info("Demucs模型已經載入，跳過初始化")


def load_model(model_name: str = "htdemucs_ft", device: str = 'auto', progress: bool = True,
               shifts: int = 5):
    """
    載入Demucs模型。
    如果相同設定的模型已載入，直接返回現有模型而不重新載入。
    """
    global separator, model_loaded, current_model_config

    if separator is not None:
        # 檢查是否需要重新載入模型（設定不同）
        requested_config = {
            'model_name': model_name,
            'device': 'auto' if device == 'auto' else device,
            'shifts': shifts
        }

        if current_model_config == requested_config:
            logger.info(f'Demucs模型已載入且設定相同，重用現有模型')
            return separator
        else:
            logger.info(f'Demucs模型設定改變，需要重新載入')
            # 釋放現有模型資源
            release_model()

    logger.info(f'載入Demucs模型: {model_name}')
    t_start = time.time()

    from demucs.api import Separator
    device_to_use = auto_device if device == 'auto' else device
    separator = Separator(model_name, device=device_to_use, progress=progress, shifts=shifts)

    # 儲存目前模型設定
    current_model_config = {
        'model_name': model_name,
        'device': 'auto' if device == 'auto' else device,
        'shifts': shifts
    }

    model_loaded = True
    t_end = time.time()
    logger.info(f'Demucs模型載入完成，用時 {t_end - t_start:.2f} 秒')

    return separator


def release_model():
    """
    釋放模型资源，避免記憶體洩漏
    """
    global separator, model_loaded, current_model_config

    if separator is not None:
        logger.info('正在釋放Demucs模型资源...')
        # 删除引用
        separator = None
        # 强制垃圾回收
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

        model_loaded = False
        current_model_config = {}
        logger.info('Demucs模型资源已釋放')


def separate_audio(folder: str, model_name: str = "htdemucs_ft", device: str = 'auto', progress: bool = True,
                   shifts: int = 5) -> None:
    """
    分離音訊檔案
    """
    global separator
    audio_path = os.path.join(folder, 'audio.wav')
    if not os.path.exists(audio_path):
        return None, None
    vocal_output_path = os.path.join(folder, 'audio_vocals.wav')
    instruments_output_path = os.path.join(folder, 'audio_instruments.wav')

    if os.path.exists(vocal_output_path) and os.path.exists(instruments_output_path):
        logger.info(f'音訊已分離: {folder}')
        return vocal_output_path, instruments_output_path

    logger.info(f'正在分離音訊: {folder}')

    if use_replicate(model_name):
        return separate_with_replicate(
            audio_path, vocal_output_path, instruments_output_path,
            model_name=model_name, shifts=shifts)

    try:
        # 确保模型已載入並且設定正確
        if not model_loaded or current_model_config.get('model_name') != model_name or \
                (current_model_config.get('device') == 'auto') != (device == 'auto') or \
                current_model_config.get('shifts') != shifts:
            load_model(model_name, device, progress, shifts)

        t_start = time.time()

        try:
            origin, separated = separator.separate_audio_file(audio_path)
        except Exception as e:
            logger.error(f'音訊分離出錯: {e}')
            # 在發生錯誤時嘗試重新載入模型一次
            release_model()
            load_model(model_name, device, progress, shifts)
            logger.info(f'已重新載入模型，重試分離...')
            origin, separated = separator.separate_audio_file(audio_path)

        t_end = time.time()
        logger.info(f'音訊分離完成，用時 {t_end - t_start:.2f} 秒')

        vocals = separated['vocals'].numpy().T
        instruments = None
        for k, v in separated.items():
            if k == 'vocals':
                continue
            if instruments is None:
                instruments = v
            else:
                instruments += v
        instruments = instruments.numpy().T

        save_wav(vocals, vocal_output_path, sample_rate=44100)
        logger.info(f'已儲存人声: {vocal_output_path}')

        save_wav(instruments, instruments_output_path, sample_rate=44100)
        logger.info(f'已儲存伴奏: {instruments_output_path}')

        return vocal_output_path, instruments_output_path

    except Exception as e:
        logger.error(f'分離音訊失敗: {str(e)}')
        # 出現錯誤，釋放模型资源並重新拋出异常
        release_model()
        raise


def extract_audio_from_video(folder: str) -> bool:
    """
    從影片中提取音訊
    """
    video_path = os.path.join(folder, 'download.mp4')
    if not os.path.exists(video_path):
        return False
    audio_path = os.path.join(folder, 'audio.wav')
    if os.path.exists(audio_path):
        logger.info(f'音訊已提取: {folder}')
        return True
    logger.info(f'正在從影片提取音訊: {folder}')

    os.system(
        f'ffmpeg -loglevel error -i "{video_path}" -vn -acodec pcm_s16le -ar 44100 -ac 2 "{audio_path}"')

    time.sleep(1)
    logger.info(f'音訊提取完成: {folder}')
    return True


def separate_all_audio_under_folder(root_folder: str, model_name: str = "htdemucs_ft",
                                    progress: bool = True, shifts: int = 5, device: str = 'auto') -> None:
    """
    分離資料夾下所有音訊
    """
    global separator
    vocal_output_path, instruments_output_path = None, None

    try:
        for subdir, dirs, files in os.walk(root_folder):
            if 'download.mp4' not in files:
                continue
            if 'audio.wav' not in files:
                extract_audio_from_video(subdir)
            if 'audio_vocals.wav' not in files:
                vocal_output_path, instruments_output_path = separate_audio(subdir, model_name, device, progress,
                                                                            shifts)
            elif 'audio_vocals.wav' in files and 'audio_instruments.wav' in files:
                vocal_output_path = os.path.join(subdir, 'audio_vocals.wav')
                instruments_output_path = os.path.join(subdir, 'audio_instruments.wav')
                logger.info(f'音訊已分離: {subdir}')

        logger.info(f'已完成所有音訊分離: {root_folder}')
        return f'所有音訊分離完成: {root_folder}', vocal_output_path, instruments_output_path

    except Exception as e:
        logger.error(f'分離音訊過程中出錯: {str(e)}')
        # 出現任何錯誤，釋放模型资源
        release_model()
        raise


if __name__ == '__main__':
    folder = r"videos"
    separate_all_audio_under_folder(folder, shifts=0)