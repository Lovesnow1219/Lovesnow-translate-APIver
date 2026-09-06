import json
import os
import queue
import re
import shutil
import subprocess
import threading
import time
import traceback

from loguru import logger

VIDEO_EXTS = ('.mp4', '.mkv', '.webm', '.mov', '.avi', '.m4v', '.flv', '.mpeg', '.mpg')


def _as_path(value):
    if value is None or value is False:
        return None
    if isinstance(value, dict):
        value = value.get('path') or value.get('name') or value.get('orig_name')
    value = str(value).strip().strip('"').strip("'")
    return value or None


def is_url_source(text):
    text = (text or '').strip()
    lowered = text.lower()
    return lowered.startswith(('http://', 'https://', 'www.')) or 'bilibili.com' in lowered or 'youtube.com' in lowered or 'youtu.be' in lowered


def looks_like_local_video(text):
    path = _as_path(text)
    if not path or is_url_source(path):
        return False
    ext = os.path.splitext(path)[1].lower()
    if os.path.isdir(path):
        return True
    if ext in VIDEO_EXTS:
        return True
    if re.match(r'^[a-zA-Z]:[\\/]', path) or path.startswith('\\\\') or path.startswith('./') or path.startswith('.\\'):
        return os.path.exists(path)
    return os.path.isfile(path)


def collect_jobs(url, local_file=None):
    jobs = []
    uploaded = _as_path(local_file)
    if uploaded:
        jobs.append(('local', uploaded))
    text = (url or '').strip()
    if not text:
        return jobs
    for chunk in re.split(r'[\n\r，]+', text):
        chunk = chunk.strip().strip('"').strip("'")
        if not chunk:
            continue
        if looks_like_local_video(chunk):
            jobs.append(('local', chunk))
        else:
            jobs.append(('url', normalize_media_url(chunk)))
    return jobs


def _copy_as_download_mp4(src, dest):
    if os.path.abspath(src) == os.path.abspath(dest):
        return dest
    ext = os.path.splitext(src)[1].lower()
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if ext == '.mp4':
        shutil.copy2(src, dest)
        return dest
    copy = subprocess.run(
        ['ffmpeg', '-y', '-i', src, '-c', 'copy', '-movflags', '+faststart', dest],
        capture_output=True, text=True, check=False,
    )
    if copy.returncode == 0 and os.path.isfile(dest) and os.path.getsize(dest) > 0:
        return dest
    transcode = subprocess.run(
        ['ffmpeg', '-y', '-i', src, '-c:v', 'libx264', '-c:a', 'aac', '-movflags', '+faststart', dest],
        capture_output=True, text=True, check=False,
    )
    if transcode.returncode == 0 and os.path.isfile(dest) and os.path.getsize(dest) > 0:
        return dest
    shutil.copy2(src, dest)
    return dest


def prepare_local_video(src, root_folder, preferred_name=None):
    src = os.path.abspath(os.path.expanduser(_as_path(src)))
    if os.path.isdir(src):
        found = None
        for name in ('download.mp4', 'video.mp4'):
            cand = os.path.join(src, name)
            if os.path.isfile(cand):
                found = cand
                break
        if found is None:
            for name in os.listdir(src):
                if os.path.splitext(name)[1].lower() in VIDEO_EXTS:
                    found = os.path.join(src, name)
                    break
        if found is None:
            raise FileNotFoundError(f'資料夾裡找不到影片: {src}')
        src = found
    if not os.path.isfile(src):
        raise FileNotFoundError(f'找不到本地影片: {src}')

    folder = os.path.dirname(src)
    base = os.path.basename(preferred_name or src)
    if os.path.basename(src) in ('download.mp4', 'video.mp4'):
        dest = os.path.join(folder, 'download.mp4')
        if os.path.abspath(src) != os.path.abspath(dest):
            _copy_as_download_mp4(src, dest)
        logger.info(f'使用本地影片資料夾: {folder}')
        return dest

    stem = os.path.splitext(base)[0] or 'local_video'
    new_folder = os.path.join(root_folder, stem)
    os.makedirs(new_folder, exist_ok=True)
    dest = os.path.join(new_folder, 'download.mp4')
    _copy_as_download_mp4(src, dest)
    logger.info(f'已準備本地影片: {src} -> {dest}')
    return dest


from .download import (
    download_single_video,
    get_info_list_from_url,
    get_target_folder,
    normalize_media_url,
)
from .demucs import separate_all_audio_under_folder
from .asr import transcribe_all_audio_under_folder
from .translation import translate_all_transcript_under_folder
from .tts import generate_all_wavs_under_folder
from .synthesize import synthesize_all_video_under_folder


def initialize_models(tts_method, asr_method, diarization):
    logger.info(f'雲端流程：人聲分離 Replicate、識別 {asr_method}、配音 {tts_method}')


def process_video(info, root_folder, resolution,
                  demucs_model, device, shifts,
                  asr_method, whisper_model, batch_size, diarization, whisper_min_speakers, whisper_max_speakers,
                  translation_method, translation_target_language,
                  tts_method, tts_target_language, voice,
                  subtitles, speed_up, fps, background_music, bgm_volume, video_volume,
                  target_resolution, max_retries, progress_callback=None,
                  force_retranslate=False, force_redub=False,
                  source_language='中文'):
    """
    處理單個影片的完整流程，增加了進度回呼函式

    Args:
        progress_callback: 回呼函式，用於回報進度和狀態，格式為 progress_callback(progress_percent, status_message)
    """
    local_time = time.localtime()

    # 定義進度階段和權重
    stages = [
        ("下載影片...", 10),  # 10%
        ("人聲分離...", 15),  # 15%
        ("語音識別與修稿...", 20),  # 20%
        ("字幕翻譯...", 25),  # 25%
        ("AI語音合成...", 20),  # 20%
        ("影片合成...", 10)  # 10%
    ]

    current_stage = 0
    progress_base = 0

    from tools.cost_tracker import begin_session, finish_session, mark_stage
    from tools.job_control import JobStopped, check_stop
    begin_session(progress_cb=progress_callback)
    mark_stage('準備處理...', 0)
    try:
        for retry in range(max_retries):
            current_stage = 0
            progress_base = 0
            try:
                check_stop()
                is_local = isinstance(info, str) and os.path.isfile(info)
                stage_name, stage_weight = stages[current_stage]
                if is_local:
                    stage_name = '準備本地影片...'
                mark_stage(stage_name, progress_base)

                if is_local:
                    folder = os.path.dirname(info)
                else:
                    folder = get_target_folder(info, root_folder)
                    if folder is None:
                        error_msg = f'無法取得影片目標資料夾: {info["title"]}'
                        logger.warning(error_msg)
                        return False, None, error_msg

                    folder = download_single_video(info, root_folder, resolution)
                    if folder is None:
                        error_msg = f'下載影片失敗: {info["title"]}'
                        logger.warning(error_msg)
                        return False, None, error_msg

                logger.info(f'處理影片: {folder}')
                begin_session(folder)
                if force_retranslate:
                    from tools.target_language import clear_translation_cache
                    keep_bible = False
                    summary_path = os.path.join(folder, 'summary.json')
                    if os.path.isfile(summary_path):
                        try:
                            with open(summary_path, 'r', encoding='utf-8') as handle:
                                keep_bible = bool((json.load(handle) or {}).get('outline_locked'))
                        except Exception:
                            keep_bible = False
                    logger.info(f'強制重翻：{folder} keep_bible={keep_bible} language={translation_target_language}')
                    clear_translation_cache(folder, keep_bible=keep_bible, language=translation_target_language)
                elif force_redub:
                    from tools.target_language import clear_tts_cache
                    logger.info(f'強制重配：{folder}')
                    clear_tts_cache(folder)

                # 完成下載階段，進入人聲分離階段
                current_stage += 1
                progress_base += stage_weight
                stage_name, stage_weight = stages[current_stage]
                check_stop()
                mark_stage(stage_name, progress_base)

                try:
                    status, vocals_path, _ = separate_all_audio_under_folder(
                        folder, model_name=demucs_model, device=device, progress=True, shifts=shifts)
                    logger.info(f'人聲分離完成: {vocals_path}')
                except JobStopped:
                    raise
                except Exception as e:
                    stack_trace = traceback.format_exc()
                    error_msg = f'人聲分離失敗: {str(e)}\n{stack_trace}'
                    logger.error(error_msg)
                    raise

                # 完成人聲分離階段，進入語音識別階段
                current_stage += 1
                progress_base += stage_weight
                stage_name, stage_weight = stages[current_stage]
                check_stop()
                mark_stage(stage_name, progress_base)

                try:
                    status, result_json = transcribe_all_audio_under_folder(
                        folder, asr_method=asr_method, whisper_model_name=whisper_model, device=device,
                        batch_size=batch_size, diarization=diarization,
                        min_speakers=whisper_min_speakers,
                        max_speakers=whisper_max_speakers,
                        language=source_language)
                    logger.info(f'語音識別完成: {status}')
                except JobStopped:
                    raise
                except Exception as e:
                    stack_trace = traceback.format_exc()
                    error_msg = f'語音識別失敗: {str(e)}\n{stack_trace}'
                    logger.error(error_msg)
                    return False, None, error_msg

                # 完成語音識別階段，進入翻譯階段
                current_stage += 1
                progress_base += stage_weight
                stage_name, stage_weight = stages[current_stage]
                check_stop()
                mark_stage(stage_name, progress_base)

                try:
                    status, summary, translation = translate_all_transcript_under_folder(
                        folder, method=translation_method, target_language=translation_target_language)
                    logger.info(f'翻譯完成: {status}')
                except JobStopped:
                    raise
                except Exception as e:
                    stack_trace = traceback.format_exc()
                    error_msg = f'翻譯失敗: {str(e)}\n{stack_trace}'
                    logger.error(error_msg)
                    return False, None, error_msg

                # 完成翻譯階段，進入語音合成階段
                current_stage += 1
                progress_base += stage_weight
                stage_name, stage_weight = stages[current_stage]
                check_stop()
                mark_stage(stage_name, progress_base)

                try:
                    status, synth_path, _ = generate_all_wavs_under_folder(
                        folder, method=tts_method, target_language=tts_target_language, voice=voice)
                    logger.info(f'語音合成完成: {synth_path}')
                except JobStopped:
                    raise
                except Exception as e:
                    stack_trace = traceback.format_exc()
                    error_msg = f'語音合成失敗: {str(e)}\n{stack_trace}'
                    logger.error(error_msg)
                    return False, None, error_msg

                # 完成語音合成階段，進入影片合成階段
                current_stage += 1
                progress_base += stage_weight
                stage_name, stage_weight = stages[current_stage]
                check_stop()
                mark_stage(stage_name, progress_base)

                try:
                    status, output_video = synthesize_all_video_under_folder(
                        folder, subtitles=subtitles, speed_up=speed_up, fps=fps, resolution=target_resolution,
                        background_music=background_music, bgm_volume=bgm_volume, video_volume=video_volume)
                    logger.info(f'影片合成完成: {output_video}')
                except JobStopped:
                    raise
                except Exception as e:
                    stack_trace = traceback.format_exc()
                    error_msg = f'影片合成失敗: {str(e)}\n{stack_trace}'
                    logger.error(error_msg)
                    return False, None, error_msg

                # 完成所有階段，回報100%進度
                mark_stage("處理完成!", 100)

                return True, output_video, "處理成功"
            except JobStopped:
                raise
            except Exception as e:
                stack_trace = traceback.format_exc()
                error_msg = f'處理影片時發生錯誤 {info["title"] if isinstance(info, dict) else info}: {str(e)}\n{stack_trace}'
                logger.error(error_msg)
                if retry < max_retries - 1:
                    logger.info(f'嘗試重試 {retry + 2}/{max_retries}...')
                else:
                    return False, None, error_msg

        return False, None, f"達到最大重試次數: {max_retries}"
    finally:
        finish_session()


def do_everything(root_folder, url, resolution='1080p',
                  shifts=1,
                  target_language='English',
                  subtitles=True, speed_up=1.00, fps=30, target_resolution='1080p',
                  max_workers=3, max_retries=5,
                  background_music=None, bgm_volume=0.5, video_volume=1.0,
                  progress_callback=None, local_file=None,
                  demucs_model='Replicate', asr_method='OpenAI',
                  whisper_model='gpt-4o-transcribe-diarize',
                  translation_method='OpenAI', tts_method='Fish',
                  num_videos=1, voice=None, device='auto', batch_size=32,
                  whisper_min_speakers=None, whisper_max_speakers=None,
                  diarization=False,
                  words_per_sec=None, translate_workers=None,
                  force_retranslate=False, force_redub=False,
                  translation_model=None, translation_effort=None,
                  review_model=None, review_effort=None,
                  source_language='中文'):
    """
    處理整個影片處理流程，增加了進度回呼函式

    Args:
        progress_callback: 回呼函式，用於回報進度和狀態，格式為 progress_callback(progress_percent, status_message)
    """
    from tools.api_settings import apply_model_choices
    from tools.cost_tracker import clear_last_markdowns, last_cost_markdown
    from tools.job_control import JobStopped, check_stop
    from tools.target_language import split_target_language
    apply_model_choices(
        translation_model=translation_model,
        translation_effort=translation_effort,
        review_model=review_model,
        review_effort=review_effort,
    )
    clear_last_markdowns()
    translation_target_language, tts_target_language = split_target_language(target_language)
    if words_per_sec not in (None, '', False):
        try:
            os.environ['DUBBING_WORDS_PER_SEC'] = str(float(words_per_sec))
        except (TypeError, ValueError):
            pass
    if translate_workers not in (None, '', False):
        try:
            os.environ['TRANSLATE_WORKERS'] = str(max(0, int(float(translate_workers))))
        except (TypeError, ValueError):
            pass

    def _done(status, video):
        md = last_cost_markdown()
        if md and '尚無成本資料' not in md:
            status = f'{status}\n\n{md}'
        return status, video

    try:
        success_list = []
        fail_list = []
        error_details = []

        # 紀錄處理開始資訊和所有參數
        logger.info("-" * 50)
        logger.info(f"開始處理任務: {url}")
        logger.info(f"參數: 輸出資料夾={root_folder}, 影片數量={num_videos}, 解析度={resolution}")
        logger.info(f"人聲分離: 模型={demucs_model}, 裝置={device}, 移位次數={shifts}")
        logger.info(f"語音識別: 方法={asr_method}, 模型={whisper_model}, 原片語言={source_language}")
        logger.info(
            f"翻譯: 方法={translation_method}, 目標語言={translation_target_language}, "
            f"模型={translation_model or os.getenv('MODEL_NAME')}, "
            f"推理={translation_effort or os.getenv('OPENAI_REASONING_EFFORT')}"
        )
        logger.info(
            f"審稿: 模型={review_model or os.getenv('REVIEW_MODEL_NAME')}, "
            f"推理={review_effort or os.getenv('REVIEW_REASONING_EFFORT')}"
        )
        logger.info(f"語音合成: 方法={tts_method}, 目標語言={tts_target_language}, 聲音={voice}")
        logger.info(f"影片合成: 字幕={subtitles}, 速度={speed_up}, FPS={fps}, 解析度={target_resolution}")
        logger.info("-" * 50)

        jobs = collect_jobs(url, local_file)
        if not jobs:
            return _done('請上傳本地影片，或填入網址 / 本機路徑。不必兩者都填。', None)

        def _run_process_video(info):
            return process_video(
                info, root_folder, resolution,
                demucs_model, device, shifts,
                asr_method, whisper_model, batch_size, diarization, whisper_min_speakers, whisper_max_speakers,
                translation_method, translation_target_language,
                tts_method, tts_target_language, voice,
                subtitles, speed_up, fps, background_music, bgm_volume, video_volume,
                target_resolution, max_retries, progress_callback,
                force_retranslate=force_retranslate, force_redub=force_redub,
                source_language=source_language,
            )

        # 初始化模型（改用新的初始化函式）
        try:
            if progress_callback:
                progress_callback(5, "初始化模型中...")
            initialize_models(tts_method, asr_method, diarization)
        except Exception as e:
            stack_trace = traceback.format_exc()
            logger.error(f"初始化模型失敗: {str(e)}\n{stack_trace}")
            return _done(f"初始化模型失敗: {str(e)}", None)

        out_video = None
        local_jobs = [src for kind, src in jobs if kind == 'local']
        url_jobs = [src for kind, src in jobs if kind == 'url']

        for src in local_jobs:
            check_stop()
            try:
                preferred = None
                if local_file is not None:
                    if isinstance(local_file, dict):
                        preferred = local_file.get('orig_name') or local_file.get('name')
                    else:
                        preferred = getattr(local_file, 'orig_name', None)
                dest = prepare_local_video(src, root_folder, preferred_name=preferred)
                success, output_video, error_msg = _run_process_video(dest)
                if success:
                    success_list.append(src)
                    out_video = output_video
                    logger.info(f"影片處理成功: {dest}")
                else:
                    fail_list.append(src)
                    error_details.append(f"{src}: {error_msg}")
                    logger.error(f"影片處理失敗: {dest}, 錯誤: {error_msg}")
            except JobStopped:
                return _done('已中止', out_video)
            except Exception as e:
                stack_trace = traceback.format_exc()
                fail_list.append(src)
                error_details.append(f"{src}: {str(e)}")
                logger.error(f"處理本地影片失敗: {str(e)}\n{stack_trace}")

        if url_jobs:
            try:
                if progress_callback:
                    progress_callback(10, "取得影片資訊中...")
                videos_info = []
                for video_info in get_info_list_from_url(url_jobs, num_videos):
                    videos_info.append(video_info)
                if not videos_info:
                    if not success_list:
                        return _done("取得影片資訊失敗，請檢查網址是否正確", out_video)
                    error_details.append("取得影片資訊失敗，請檢查網址是否正確")
                for info in videos_info:
                    check_stop()
                    try:
                        success, output_video, error_msg = _run_process_video(info)
                        label = info['title'] if isinstance(info, dict) else info
                        if success:
                            success_list.append(info)
                            out_video = output_video
                            logger.info(f"成功處理影片: {label}")
                        else:
                            fail_list.append(info)
                            error_details.append(f"{label}: {error_msg}")
                            logger.error(f"處理影片失敗: {label}, 錯誤: {error_msg}")
                    except JobStopped:
                        return _done('已中止', out_video)
                    except Exception as e:
                        stack_trace = traceback.format_exc()
                        fail_list.append(info)
                        error_details.append(f"{info}: {str(e)}")
                        logger.error(f"處理影片出錯: {info}, 錯誤: {str(e)}\n{stack_trace}")
            except Exception as e:
                stack_trace = traceback.format_exc()
                logger.error(f"取得影片清單失敗: {str(e)}\n{stack_trace}")
                if not success_list:
                    return _done(f"取得影片清單失敗: {str(e)}", out_video)
                error_details.append(str(e))

        # 紀錄處理結果彙總
        logger.info("-" * 50)
        logger.info(f"處理完成: 成功={len(success_list)}, 失敗={len(fail_list)}")
        if error_details:
            logger.info("失敗詳情:")
            for detail in error_details:
                logger.info(f"  - {detail}")

        return _done(f'成功: {len(success_list)}\n失敗: {len(fail_list)}', out_video)

    except JobStopped:
        return _done('已中止', None)
    except Exception as e:
        # 捕捉整體處理過程中的任何錯誤
        stack_trace = traceback.format_exc()
        error_msg = f"處理過程中發生錯誤: {str(e)}\n{stack_trace}"
        logger.error(error_msg)
        return _done(error_msg, None)


def stream_do_everything(root_folder, url, *args, local_file=None, **kwargs):
    """Yield (status, video, cost_markdown) so the WebUI can stream progress."""
    from tools.cost_tracker import current_session, last_cost_markdown, live_cost_markdown
    from tools.job_control import JobStopped, bind_job, finish_job, is_stopped, request_stop, start_job

    job = start_job()
    q = queue.Queue()
    holder = {'status': '準備中...', 'video': None}

    def cb(pct, msg):
        md = live_cost_markdown() if current_session() is not None else ''
        q.put(('progress', msg, md))

    def worker():
        bind_job(job)
        try:
            status, video = do_everything(
                root_folder, url, *args,
                progress_callback=cb,
                local_file=local_file,
                **kwargs,
            )
            holder['status'] = '已中止' if is_stopped() else status
            holder['video'] = video
        except JobStopped:
            holder['status'] = '已中止'
            holder['video'] = None
        except Exception as e:
            holder['status'] = f'處理失敗: {e}\n{traceback.format_exc()}'
            holder['video'] = None
        finally:
            finish_job(job)
            q.put(('done', None, None))

    worker_thread = threading.Thread(target=worker, daemon=True)
    worker_thread.start()
    try:
        yield '準備中...\n合成狀態會即時顯示各步驟進度與成本。\n可只上傳本地影片，不必填網址。', None, '尚未開始'
        while True:
            try:
                kind, msg, md = q.get(timeout=0.3)
            except queue.Empty:
                continue
            if kind == 'done':
                final = holder['status']
                if is_stopped() and (not final or final == '準備中...'):
                    final = '已中止'
                yield final, holder['video'], last_cost_markdown()
                break
            if is_stopped():
                yield '正在中止…正在收尾目前這次 API 呼叫，之後不會再進下一步。', None, md or ''
                continue
            yield msg, None, md
    except GeneratorExit:
        request_stop()
        raise


if __name__ == '__main__':
    do_everything(
        root_folder='videos',
        url='https://www.bilibili.com/video/BV1kr421M7vz/',
        translation_method='LLM',
        # translation_method = 'Google Translate', translation_target_language = '簡體中文',
    )