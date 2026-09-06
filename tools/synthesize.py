# -*- coding: utf-8 -*-
import json
import os
import re
import shutil
import string
import subprocess
import threading
import time
import random
import traceback
from queue import Empty, Queue

from loguru import logger


def split_text(input_data,
               punctuations=['，', '；', '：', '。', '？', '！', '\n', '”', ',', '.', '?', '!', ';', ':']):
    # Sentence-ending punctuation for Chinese and English subtitles

    def is_punctuation(char):
        return char in punctuations

    output_data = []
    for item in input_data:
        start = item["start"]
        text = item.get("translation") or ''
        speaker = item.get("speaker", "SPEAKER_00")
        original_text = item.get("text") or ''
        if not text.strip():
            output_data.append({
                "start": round(item["start"], 3),
                "end": round(item["end"], 3),
                "text": original_text,
                "translation": text,
                "speaker": speaker
            })
            continue
        sentence_start = 0
        duration_per_char = (item["end"] - item["start"]) / max(len(text), 1)
        for i, char in enumerate(text):
            # If the character is a punctuation, split the sentence
            if not is_punctuation(char) and i != len(text) - 1:
                continue
            if i - sentence_start < 5 and i != len(text) - 1:
                continue
            if i < len(text) - 1 and is_punctuation(text[i+1]):
                continue
            sentence = text[sentence_start:i+1]
            sentence_end = start + duration_per_char * len(sentence)

            # Append the new item
            output_data.append({
                "start": round(start, 3),
                "end": round(sentence_end, 3),
                "text": original_text,
                "translation": sentence,
                "speaker": speaker
            })

            # Update the start for the next sentence
            start = sentence_end
            sentence_start = i + 1

    return output_data
    
def format_timestamp(seconds):
    """Converts seconds to the SRT time format."""
    millisec = int((seconds - int(seconds)) * 1000)
    hours, seconds = divmod(int(seconds), 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02},{millisec:03}"


def _escape_ffmpeg_subtitles_path(path):
    """Escape a Windows path for FFmpeg's subtitles filter (colons, backslashes, spaces)."""
    path = os.path.abspath(path).replace('\\', '/')
    path = path.replace(':', r'\:')
    path = path.replace("'", r"\'")
    return f"'{path}'"


_SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')
_CLAUSE_SPLIT_RE = re.compile(r'(?<=[,;:、，；：—–])\s+')
_MAX_CUE_CHARS = 68


def _wrap_srt_text(text, max_line_char=42):
    text = ' '.join((text or '').split())
    if not text:
        return ''
    if ' ' not in text:
        if len(text) <= max_line_char:
            return text
        return '\n'.join(text[i:i + max_line_char] for i in range(0, len(text), max_line_char))
    lines, current = [], text.split()[0]
    for word in text.split()[1:]:
        if len(current) + 1 + len(word) <= max_line_char:
            current = f'{current} {word}'
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return '\n'.join(lines)


def _split_words(text, max_chars):
    words = (text or '').split()
    if not words:
        return []
    out, buf = [], words[0]
    for word in words[1:]:
        cand = f'{buf} {word}'
        if len(cand) <= max_chars:
            buf = cand
        else:
            out.append(buf)
            buf = word
    out.append(buf)
    return out


def _merge_short_cues(parts, max_chars):
    out = []
    for part in parts:
        if out and (len(out[-1]) < 22 or len(part) < 16):
            out[-1] = f'{out[-1]} {part}'.strip()
        else:
            out.append(part)
    flat = []
    for part in out:
        if len(part) <= max_chars + 12:
            flat.append(part)
        else:
            flat.extend(_split_words(part, max_chars))
    return flat


def _split_cue_texts(text, max_chars=_MAX_CUE_CHARS):
    text = ' '.join((text or '').split())
    if not text:
        return []
    sentences = [part.strip() for part in _SENT_SPLIT_RE.split(text) if part.strip()] or [text]
    pieces = []
    for sentence in sentences:
        if len(sentence) <= max_chars:
            pieces.append(sentence)
            continue
        clauses = [part.strip() for part in _CLAUSE_SPLIT_RE.split(sentence) if part.strip()] or [sentence]
        buf = ''
        for clause in clauses:
            cand = f'{buf} {clause}'.strip() if buf else clause
            if len(cand) <= max_chars:
                buf = cand
                continue
            if buf:
                pieces.append(buf)
            if len(clause) <= max_chars:
                buf = clause
            else:
                pieces.extend(_split_words(clause, max_chars))
                buf = ''
        if buf:
            pieces.append(buf)
    return _merge_short_cues(pieces, max_chars)


def _picture_span(line):
    try:
        start = float(line.get('orig_start') if line.get('orig_start') is not None else line.get('start'))
        end = float(line.get('orig_end') if line.get('orig_end') is not None else line.get('end'))
    except (TypeError, ValueError):
        return None
    if end < start + 0.06:
        return None
    return start, end


def _apportion_cues(start, end, pieces):
    weights = [max(1, len(part)) for part in pieces]
    total = sum(weights) or 1
    span = max(0.12, end - start)
    cursor = start
    cues = []
    for i, (part, weight) in enumerate(zip(pieces, weights)):
        nxt = end if i == len(pieces) - 1 else cursor + span * (weight / total)
        cues.append((cursor, max(cursor + 0.06, nxt), part))
        cursor = cues[-1][1]
    if cues:
        cues[-1] = (cues[-1][0], end, cues[-1][2])
    return cues


def _line_cues(line):
    if not line or line.get('skip_tts'):
        return []
    text = (line.get('translation') or '').strip()
    if not text:
        return []
    span = _picture_span(line)
    if not span:
        return []
    start, end = span
    return _apportion_cues(start, end, _split_cue_texts(text))


def generate_srt(translation, srt_path, speed_up=1, max_line_char=42, folder=None):
    cues = []
    for line in translation or []:
        cues.extend(_line_cues(line))
    with open(srt_path, 'w', encoding='utf-8') as handle:
        written = 0
        for i, (start, end, text) in enumerate(cues):
            if i + 1 < len(cues):
                end = min(end, cues[i + 1][0])
            if end < start + 0.06:
                continue
            written += 1
            handle.write(f'{written}\n')
            handle.write(
                f'{format_timestamp(start / speed_up)} --> {format_timestamp(end / speed_up)}\n'
            )
            handle.write(f'{_wrap_srt_text(text, max_line_char)}\n\n')


def get_aspect_ratio(video_path):
    command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
               '-show_entries', 'stream=width,height', '-of', 'json', video_path]
    result = subprocess.run(command, capture_output=True, text=True)
    dimensions = json.loads(result.stdout)['streams'][0]
    return dimensions['width'] / dimensions['height']


def convert_resolution(aspect_ratio, resolution='1080p'):
    if aspect_ratio < 1:
        width = int(resolution[:-1])
        height = int(width / aspect_ratio)
    else:
        height = int(resolution[:-1])
        width = int(height * aspect_ratio)
    width = width - width % 2
    height = height - height % 2
    return width, height


def _srt_copy_for_ffmpeg(srt_path):
    temp_dir = os.path.join(os.environ.get('SystemRoot', r'C:\Windows'), 'Temp', 'lovesnow-translate')
    try:
        os.makedirs(temp_dir, exist_ok=True)
    except OSError:
        temp_dir = 'temp'
        os.makedirs(temp_dir, exist_ok=True)
    dest = os.path.join(temp_dir, f'sub_{os.getpid()}.srt')
    shutil.copyfile(srt_path, dest)
    return dest


def _video_encoder_args():
    return ['-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20', '-pix_fmt', 'yuv420p']


def _ffmpeg_output_path(command):
    for item in reversed(command or []):
        text = str(item)
        if text.endswith(('.mp4', '.mkv', '.webm', '.mov', '.wav', '.m4a')):
            return text
    return None


def _output_is_complete(path, duration):
    if not path or not os.path.exists(path):
        return False
    try:
        if os.path.getsize(path) < 10000:
            return False
    except OSError:
        return False
    if not duration or duration <= 1:
        return True
    try:
        from tools.audio_chunks import media_duration
        got = media_duration(path)
    except Exception:
        return False
    return got >= float(duration) * 0.97


def _newer(path, than):
    try:
        return os.path.isfile(path) and os.path.getmtime(path) > os.path.getmtime(than) + 1
    except OSError:
        return False


def _mux_is_current(folder, final_video, duration):
    if not _output_is_complete(final_video, duration):
        return False
    for name in ('audio_combined.wav', 'translation.json', 'download.mp4'):
        if _newer(os.path.join(folder, name), final_video):
            return False
    return True


def _swap_output_path(command, new_path):
    cmd = list(command)
    old = _ffmpeg_output_path(cmd)
    if old:
        for i in range(len(cmd) - 1, -1, -1):
            if str(cmd[i]) == old:
                cmd[i] = new_path
                break
    return cmd


def _finish_mux_file(tmp_path, final_path, duration):
    if tmp_path and os.path.isfile(tmp_path) and _output_is_complete(tmp_path, duration):
        os.replace(tmp_path, final_path)
        return True
    if _output_is_complete(final_path, duration):
        if tmp_path and os.path.isfile(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        return True
    return False


def _run_ffmpeg(command, what='FFmpeg', duration=None):
    from tools.cost_tracker import mark_stage
    from tools.job_control import JobStopped, is_stopped

    cmd = list(command)
    final_path = _ffmpeg_output_path(cmd)
    tmp_path = f'{final_path}.muxing.mp4' if final_path else None
    if tmp_path:
        cmd = _swap_output_path(cmd, tmp_path)
    if cmd and str(cmd[0]).lower().endswith('ffmpeg'):
        cmd[1:1] = ['-nostdin', '-loglevel', 'error', '-progress', 'pipe:1', '-nostats', '-hide_banner']
    logger.info(f'{what}: {" ".join(str(x) for x in cmd)}')
    # Never pipe stderr: libass subtitle logs fill the Windows 64KB pipe and deadlock ffmpeg.
    popen_kw = dict(
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    if os.name == 'nt':
        popen_kw['creationflags'] = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
    proc = subprocess.Popen(cmd, **popen_kw)
    lines = Queue()

    def _read_stdout():
        buf = b''
        try:
            stream = proc.stdout
            while stream:
                chunk = stream.read(256)
                if not chunk:
                    break
                buf += chunk
                while b'\n' in buf:
                    raw, buf = buf.split(b'\n', 1)
                    lines.put(raw.decode('utf-8', 'replace'))
        except Exception:
            pass
        finally:
            lines.put(None)

    threading.Thread(target=_read_stdout, name='ffmpeg-stdout', daemon=True).start()
    seconds = 0.0
    last_report = 0.0
    last_line_at = time.time()
    stalled = False
    try:
        while True:
            if is_stopped():
                proc.kill()
                raise JobStopped('已中止')
            try:
                raw = lines.get(timeout=1.0)
            except Empty:
                if proc.poll() is not None and lines.empty():
                    break
                idle = time.time() - last_line_at
                near_end = duration and duration > 1 and seconds >= duration * 0.95
                if idle > (20 if near_end else 90):
                    logger.error(f'{what} 超過 {20 if near_end else 90} 秒沒有進度，中止以免卡在最後 1%')
                    stalled = True
                    proc.kill()
                    break
                continue
            if raw is None:
                break
            last_line_at = time.time()
            line = (raw or '').strip()
            if line.startswith('out_time_ms='):
                try:
                    seconds = max(0.0, int(line.split('=', 1)[1]) / 1_000_000)
                except ValueError:
                    continue
            now = time.time()
            if now - last_report < 1.5:
                continue
            last_report = now
            if duration and duration > 1:
                frac = min(1.0, seconds / duration)
                mark_stage(f'影片合成中 {seconds:.0f}/{duration:.0f} 秒', 90 + 9 * frac)
                logger.info(f'{what} {seconds:.0f}/{duration:.0f}s')
            else:
                mark_stage(f'影片合成中 {seconds:.0f} 秒', 92)
        try:
            code = proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            code = proc.wait()
            stalled = True
    except JobStopped:
        if proc.poll() is None:
            proc.kill()
        raise
    if code == 0 and _finish_mux_file(tmp_path, final_path, duration or 0):
        return True
    if _finish_mux_file(tmp_path, final_path, duration):
        logger.warning(f'{what} 行程異常結束，但輸出檔已完整，視為成功: {final_path}')
        return True
    if tmp_path and os.path.isfile(tmp_path):
        try:
            os.remove(tmp_path)
        except OSError:
            pass
    if stalled:
        logger.error(f'{what} 卡住後輸出不完整')
        return False
    logger.error(f'{what} 失敗')
    return False


def _publish_named_video(folder, video_path):
    """Keep one Explorer file: {folder}_{Language}.mp4. Drop leftover video.mp4."""
    from tools.target_language import drop_working_video_alias, titled_video_path

    dest = titled_video_path(folder)
    if video_path and os.path.isfile(video_path):
        same = os.path.normcase(os.path.abspath(dest)) == os.path.normcase(os.path.abspath(video_path))
        if not same:
            try:
                if os.path.isfile(dest):
                    os.remove(dest)
                os.replace(video_path, dest)
                logger.info(f'成片已存成：{dest}')
            except OSError:
                try:
                    shutil.copy2(video_path, dest)
                    os.remove(video_path)
                    logger.info(f'成片已複製成：{dest}')
                except OSError as exc:
                    logger.warning(f'無法寫入片名影片：{exc}')
                    drop_working_video_alias(folder, dest)
                    return video_path
    drop_working_video_alias(folder, dest)
    return dest if os.path.isfile(dest) else video_path


def synthesize_video(folder, subtitles=True, speed_up=1.00, fps=30, resolution='1080p', background_music=None, watermark_path=None, bgm_volume=0.5, video_volume=1.0):
    translation_path = os.path.join(folder, 'translation.json')
    input_audio = os.path.join(folder, 'audio_combined.wav')
    input_video = os.path.join(folder, 'download.mp4')
    
    if not os.path.exists(translation_path) or not os.path.exists(input_audio):
        return
    
    with open(translation_path, 'r', encoding='utf-8') as f:
        translation = json.load(f)

    srt_path = os.path.join(folder, 'subtitles.srt')
    from tools.target_language import titled_video_path
    final_video = titled_video_path(folder)
    duration = None
    try:
        from tools.audio_chunks import media_duration
        duration = media_duration(input_video)
    except Exception:
        duration = None
    if subtitles and _mux_is_current(folder, final_video, duration):
        logger.info(f'影片已合成且音訊未更新，跳過重燒: {final_video}')
        return _publish_named_video(folder, final_video)
    generate_srt(translation, srt_path, speed_up, folder=folder)
    aspect_ratio = get_aspect_ratio(input_video)
    width, height = convert_resolution(aspect_ratio, resolution)
    font_size = int(width / 128)
    outline = int(round(font_size / 8))
    speed_up = float(speed_up or 1.0)
    need_speed = abs(speed_up - 1.0) > 0.001
    srt_temp = None
    started = time.time()

    try:
        video_steps = []
        if need_speed:
            video_steps.append(f'setpts=PTS/{speed_up}')
        video_steps.append(f'scale={width}:{height}')
        video_steps.append(f'fps={int(fps)}')
        mux_duration = duration
        try:
            from tools.audio_chunks import media_duration as _media_duration
            audio_duration = _media_duration(input_audio)
        except Exception:
            audio_duration = None
        extra = 0.0
        if duration and audio_duration and audio_duration > duration + 0.08:
            extra = float(audio_duration - duration)
            mux_duration = float(audio_duration)
            video_steps.append(f'tpad=stop_mode=clone:stop_duration={extra:.3f}')

        inputs = ['-i', input_video, '-i', input_audio]
        next_idx = 2
        wm_idx = None
        bgm_idx = None
        if watermark_path:
            inputs += ['-i', watermark_path]
            wm_idx = next_idx
            next_idx += 1
        if background_music:
            inputs += ['-i', background_music]
            bgm_idx = next_idx

        video_out = '[v]' if not subtitles and wm_idx is None else '[v0]'
        graph = [f"[0:v]{','.join(video_steps)}{video_out}"]
        vlabel = video_out
        if wm_idx is not None:
            video_out = '[v]' if not subtitles else '[v1]'
            graph.append(f'[{wm_idx}:v]scale=iw*0.15:ih*0.15[wm]')
            graph.append(f'{vlabel}[wm]overlay=W-w-10:H-h-10{video_out}')
            vlabel = video_out

        if subtitles:
            srt_temp = _srt_copy_for_ffmpeg(srt_path)
            srt_escaped = _escape_ffmpeg_subtitles_path(srt_temp)
            style = (
                f"FontName=Arial,FontSize={font_size},PrimaryColour=&HFFFFFF,"
                f"OutlineColour=&H000000,Outline={outline},WrapStyle=2"
            )
            graph.append(f"{vlabel}subtitles={srt_escaped}:force_style='{style}'[v]")

        audio_map = '1:a'
        if need_speed and bgm_idx is not None:
            graph.append(f'[1:a]atempo={speed_up},volume={video_volume}[va]')
            graph.append(f'[{bgm_idx}:a]volume={bgm_volume}[ba]')
            graph.append('[va][ba]amix=inputs=2:duration=first[a]')
            audio_map = '[a]'
        elif need_speed:
            graph.append(f'[1:a]atempo={speed_up}[a]')
            audio_map = '[a]'
        elif bgm_idx is not None:
            graph.append(f'[1:a]volume={video_volume}[va]')
            graph.append(f'[{bgm_idx}:a]volume={bgm_volume}[ba]')
            graph.append('[va][ba]amix=inputs=2:duration=first[a]')
            audio_map = '[a]'

        command = [
            'ffmpeg', '-y', '-threads', '0',
            *inputs,
            '-filter_complex', ';'.join(graph),
            '-map', '[v]',
            '-map', audio_map,
            *_video_encoder_args(),
            '-c:a', 'aac', '-b:a', '192k',
            final_video,
        ]
        if mux_duration and mux_duration > 1:
            command[-1:-1] = ['-t', f'{mux_duration:.3f}']
            logger.info(f'開始合成影片，片長約 {mux_duration:.0f} 秒')
            from tools.cost_tracker import mark_stage
            mark_stage(f'影片合成中 0/{mux_duration:.0f} 秒', 90)
        ok = _run_ffmpeg(command, '一次合成影片', duration=mux_duration)
        if not ok and subtitles:
            logger.warning('含字幕一次合成失敗，改為先無字幕再燒字幕')
            plain = synthesize_video(
                folder, subtitles=False, speed_up=speed_up, fps=fps, resolution=resolution,
                background_music=background_music, watermark_path=watermark_path,
                bgm_volume=bgm_volume, video_volume=video_volume,
            )
            if not plain or not os.path.exists(plain):
                return None
            subtitled = final_video.replace('.mp4', '_subtitles.mp4')
            burned = add_subtitles(plain, srt_path, subtitled, method='ffmpeg')
            if burned and os.path.exists(subtitled):
                os.replace(subtitled, final_video)
            else:
                logger.warning('字幕燒錄失敗，保留無字幕配音影片')
        elif not ok:
            return None
        if not os.path.exists(final_video):
            logger.error(f'合成影片未生成: {final_video}')
            return None
        logger.info(f'影片合成完成，用時 {time.time() - started:.1f}s: {final_video}')
        return _publish_named_video(folder, final_video)
    finally:
        if srt_temp and os.path.exists(srt_temp):
            try:
                os.remove(srt_temp)
            except OSError:
                pass


def add_subtitles(video_path, srt_path, output_path, subtitle_filter=None, method='ffmpeg'):
    """
    给视频文件添加字幕。

    参数：
        video_path (str): 输入视频文件的路径。
        srt_path (str): .srt 字幕文件的路径。
        output_path (str): 输出视频文件的路径。
        subtitle_filter (str): 自定义字幕过滤器，默认为None，使用标准filter。
        method (str): 使用的方法 ('moviepy' 或 'ffmpeg')，默认为 'ffmpeg'。

    返回：
        bool: 成功返回 True，失败返回 False。
    """
    try:
        # Prefer a space-free temp dir so FFmpeg's subtitles filter can parse the path
        temp_dir = os.path.join(os.environ.get('SystemRoot', r'C:\Windows'), 'Temp', 'lovesnow-translate')
        try:
            os.makedirs(temp_dir, exist_ok=True)
        except OSError:
            temp_dir = "temp"
            os.makedirs(temp_dir, exist_ok=True)

        # 生成随机字符串作为临时文件名
        random_string = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
        temp_video_path = os.path.join(temp_dir, f"temp_video_{random_string}.mp4")

        random_string = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
        temp_srt_path = os.path.join(temp_dir, f"temp_srt_{random_string}.srt")

        random_string = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
        temp_output_path = os.path.join(temp_dir, f"temp_output_{random_string}.mp4")

        # 检查源文件是否存在
        if not os.path.exists(video_path):
            logger.error(f"输入视频文件不存在: {video_path}")
            return False

        if not os.path.exists(srt_path):
            logger.error(f"字幕文件不存在: {srt_path}")
            return False

        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 开始复制原始文件到临时文件
        shutil.copyfile(video_path, temp_video_path)
        shutil.copyfile(srt_path, temp_srt_path)

        # 使用绝对路径避免路径问题
        temp_video_path = os.path.abspath(temp_video_path)
        temp_srt_path = os.path.abspath(temp_srt_path)
        temp_output_path = os.path.abspath(temp_output_path)
        # 开始检查确认字幕文件是否存在
        if not os.path.exists(temp_srt_path):
            logger.error(f"字幕文件不存在: {temp_srt_path}")
            return False
        # 开始检查确认视频文件是否存在
        if not os.path.exists(temp_video_path):
            logger.error(f"输入视频文件不存在: {temp_video_path}")
            return False

        if method == 'moviepy':
            from moviepy import VideoFileClip, TextClip, CompositeVideoClip
            from moviepy.video.tools.subtitles import SubtitlesClip

            # 使用 moviepy 添加字幕
            video = VideoFileClip(temp_video_path)
            generator = lambda txt: TextClip(txt, font='font/SimHei.ttf', fontsize=24, color='white')
            subtitles = SubtitlesClip(temp_srt_path, generator)
            final_video = video.copy()

            final_video = final_video.set_subtitles(subtitles)
            # 保存视频
            final_video.write_videofile(temp_output_path, fps=video.fps)

            # 复制回原始位置
            if os.path.exists(temp_output_path):
                shutil.copyfile(temp_output_path, output_path)
                logger.info(f"字幕添加成功，输出到: {output_path}")
                return True
            else:
                logger.error(f"输出文件未生成: {temp_output_path}")
                return False

        elif method == 'ffmpeg':
            # 使用 ffmpeg 添加字幕
            try:
                # 获取字体文件的绝对路径
                font_dir = os.path.abspath("./font")

                # 构建字幕过滤器，使用文件名引用
                style = "FontName=Arial,FontSize=18,PrimaryColour=&HFFFFFF,OutlineColour=&H000000,Outline=2,WrapStyle=2"
                srt_escaped = _escape_ffmpeg_subtitles_path(temp_srt_path)
                filter_option = f"subtitles={srt_escaped}:force_style='{style}'"

                command = [
                    'ffmpeg',
                    '-y',
                    '-threads', '0',
                    '-i', temp_video_path,
                    '-vf', filter_option,
                    '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20',
                    '-c:a', 'copy',
                    temp_output_path,
                ]

                logger.info(f"执行FFmpeg命令: {' '.join(command)}")
                if not _run_ffmpeg(command, '燒錄字幕'):
                    raise subprocess.CalledProcessError(1, command)

                # 检查是否成功生成输出文件
                if os.path.exists(temp_output_path):
                    # 确保输出目录存在
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    shutil.copyfile(temp_output_path, output_path)
                    logger.info(f"字幕添加成功，输出到: {output_path}")
                    return True
                else:
                    logger.error(f"FFmpeg执行成功但输出文件未生成: {temp_output_path}")
                    return False

            except subprocess.CalledProcessError as e:
                logger.error(f"FFmpeg命令执行失败: {e}")
                stderr_output = e.stderr.decode('utf-8', errors='ignore') if e.stderr else "No stderr output"
                logger.error(f"FFmpeg错误输出: {stderr_output}")
                return False

            except Exception as e:
                logger.error(f"添加字幕时发生错误: {str(e)}")
                import traceback
                logger.error(f"错误堆栈: {traceback.format_exc()}")
                return False
        else:
            logger.error(f"不支持的方法: {method}. 请使用 'moviepy' 或 'ffmpeg'")
            return False

    except Exception as e:
        logger.error(f"添加字幕时发生错误: {str(e)}")
        import traceback
        logger.debug(f"错误详情: {traceback.format_exc()}")
        return False
    finally:
        # 清理临时文件
        temp_files = [temp_video_path, temp_srt_path, temp_output_path]
        if method == 'ffmpeg':
            temp_files.append(os.path.join(temp_dir, "subtitles.srt"))

        for temp_file in temp_files:
            if temp_file and os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except Exception as e:
                    logger.debug(f"无法删除临时文件 {temp_file}: {e}")

def synthesize_all_video_under_folder(folder, subtitles=True, speed_up=1.00, fps=30, resolution='1080p', background_music=None, bgm_volume=0.5, video_volume=1.0, watermark_path="f_logo.png"):
    watermark_path = None if not os.path.exists(watermark_path) else watermark_path
    output_video = None
    for root, dirs, files in os.walk(folder):
        if 'download.mp4' in files:
            output_video = synthesize_video(root, subtitles=subtitles,
                            speed_up=speed_up, fps=fps, resolution=resolution,
                            background_music=background_music,
                            watermark_path=watermark_path, bgm_volume=bgm_volume, video_volume=video_volume)
        # if 'download.mp4' in files and 'video.mp4' not in files:
        #     output_video = synthesize_video(root, subtitles=subtitles,
        #                      speed_up=speed_up, fps=fps, resolution=resolution,
        #                      background_music=background_music,
        #                      watermark_path=watermark_path, bgm_volume=bgm_volume, video_volume=video_volume)
        # elif 'video.mp4' in files:
        #     output_video = os.path.join(root, 'video.mp4')
        #     logger.info(f'Video already synthesized in {folder}')
    return f'Synthesized all videos under {folder}', output_video

if __name__ == '__main__':
    folder = r"videos"
    synthesize_all_video_under_folder(folder, subtitles=True)