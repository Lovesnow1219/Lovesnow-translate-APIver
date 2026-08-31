import json
import os
import re
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

import yt_dlp
from loguru import logger

_INVISIBLE = dict.fromkeys(map(ord, '\ufeff\u200b\u200c\u200d\u2060'), None)
_YOUTUBE_HOSTS = {'youtu.be', 'youtube.com', 'www.youtube.com', 'm.youtube.com', 'music.youtube.com'}
_YOUTUBE_ID = re.compile(r'^[\w-]{11}$')


class _QuietYtdlpLogger:
    """Keep yt-dlp off Windows/Gradio stdout; those writes raise Errno 22."""

    def debug(self, msg):
        text = str(msg)
        if text.startswith('[debug] '):
            return
        logger.debug(text)

    def info(self, msg):
        logger.info(msg)

    def warning(self, msg):
        logger.warning(msg)

    def error(self, msg):
        logger.error(msg)


def sanitize_title(title):
    # Only keep numbers, letters, Chinese characters, and spaces
    title = re.sub(r'[^\w\u4e00-\u9fff \d_-]', '', title)
    # Replace multiple spaces with a single space
    title = re.sub(r'\s+', ' ', title)
    return title.strip(' .')


def normalize_media_url(url):
    text = (url or '').strip().strip('"').strip("'").translate(_INVISIBLE)
    if text.startswith('www.'):
        text = 'https://' + text
    parsed = urlparse(text)
    host = (parsed.hostname or '').lower()
    if host not in _YOUTUBE_HOSTS:
        return text

    video_id = None
    extra = {}
    path = parsed.path or ''
    if host == 'youtu.be':
        video_id = path.strip('/').split('/')[0]
    elif '/shorts/' in path or '/live/' in path or '/embed/' in path:
        video_id = path.rstrip('/').split('/')[-1]
    else:
        qs = parse_qs(parsed.query)
        video_id = (qs.get('v') or [None])[0]
        if qs.get('list'):
            extra['list'] = qs['list'][0]
        if qs.get('t'):
            extra['t'] = qs['t'][0]
        elif qs.get('start'):
            extra['t'] = qs['start'][0]

    if not video_id or not _YOUTUBE_ID.match(video_id):
        return text
    query = urlencode({'v': video_id, **extra})
    return urlunparse(('https', 'www.youtube.com', '/watch', '', query, ''))


def get_target_folder(info, folder_path):
    sanitized_title = sanitize_title(info['title'])
    sanitized_uploader = sanitize_title(info.get('uploader', 'Unknown'))
    upload_date = info.get('upload_date', 'Unknown')
    if upload_date == 'Unknown':
        return None

    output_folder = os.path.join(
        folder_path, sanitized_uploader, f'{upload_date} {sanitized_title}')

    return output_folder


def _cookiefile():
    return 'cookies.txt' if os.path.exists('cookies.txt') else None


def _base_ydl_opts(**extra):
    opts = {
        'quiet': True,
        'no_warnings': True,
        'noprogress': True,
        'logger': _QuietYtdlpLogger(),
        'retries': 5,
        'extractor_retries': 3,
        'fragment_retries': 5,
        'socket_timeout': 30,
        'windowsfilenames': True,
    }
    cookie = _cookiefile()
    if cookie:
        opts['cookiefile'] = cookie
    if os.name == 'nt':
        opts['source_address'] = '0.0.0.0'
    for key, value in extra.items():
        if value is not None:
            opts[key] = value
    return opts


def download_single_video(info, folder_path, resolution='1080p'):
    sanitized_title = sanitize_title(info['title'])
    sanitized_uploader = sanitize_title(info.get('uploader', 'Unknown'))
    upload_date = info.get('upload_date', 'Unknown')
    if upload_date == 'Unknown':
        return None

    output_folder = os.path.join(folder_path, sanitized_uploader, f'{upload_date} {sanitized_title}')
    if os.path.exists(os.path.join(output_folder, 'download.mp4')):
        logger.info(f'Video already downloaded in {output_folder}')
        return output_folder

    resolution = str(resolution).replace('p', '')
    page_url = normalize_media_url(
        info.get('webpage_url') or info.get('original_url') or info.get('url') or ''
    )
    ydl_opts = _base_ydl_opts(
        format=f'bestvideo[ext=mp4][height<={resolution}]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        writeinfojson=True,
        writethumbnail=True,
        outtmpl=os.path.join(output_folder, 'download'),
        ignoreerrors=True,
    )

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([page_url])
    logger.info(f'Video downloaded in {output_folder}')
    return output_folder

def download_videos(info_list, folder_path, resolution='1080p'):
    for info in info_list:
        output_folder = download_single_video(info, folder_path, resolution)
    return output_folder

def get_info_list_from_url(url, num_videos):
    if isinstance(url, str):
        url = [url]
    urls = [normalize_media_url(u) for u in url if u]

    try:
        playlistend = max(1, int(num_videos or 1))
    except (TypeError, ValueError):
        playlistend = 1

    ydl_opts = _base_ydl_opts(
        skip_download=True,
        playlistend=playlistend,
        ignoreerrors=True,
    )

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for u in urls:
            try:
                result = ydl.extract_info(u, download=False)
            except OSError as e:
                logger.warning(f'取得影片資訊時發生系統錯誤，改走相容模式重試: {e}')
                fallback = dict(ydl_opts)
                fallback.pop('source_address', None)
                with yt_dlp.YoutubeDL(fallback) as ydl2:
                    result = ydl2.extract_info(u, download=False)
            if not result:
                logger.error(f'無法解析影片資訊: {u}')
                continue
            if 'entries' in result:
                for video_info in result['entries']:
                    if video_info:
                        yield video_info
            else:
                yield result


def download_from_url(url, folder_path, resolution='1080p', num_videos=1):
    if isinstance(url, str):
        url = [url]
    video_info_list = list(get_info_list_from_url(url, num_videos))
    example_output_folder = download_videos(video_info_list, folder_path, resolution)
    download_info_json = None
    info_path = os.path.join(example_output_folder, 'download.info.json')
    if example_output_folder and os.path.exists(info_path):
        download_info_json = json.load(open(info_path, 'r', encoding='utf-8'))
    return f"All videos have been downloaded under the {folder_path} folder", os.path.join(example_output_folder, 'download.mp4'), download_info_json

if __name__ == '__main__':
    # Example usage
    # Youtube Title: How to Install and Use yt-dlp [2024] [Quick and Easy!] [4 Minute Tutorial] [Windows 11]
    url = 'https://www.youtube.com/watch?v=5aYwU4nj5QA'
    # Bilibili Title 高清无字幕 | 英语听力 | Taylor Swift纽约大学2022届毕业典礼演讲 | Commencement Speech at NYU
    url = 'https://www.bilibili.com/video/BV1KZ4y1h7ke/'
    # Bilbili Title 奥巴马开学演讲，纯英文字幕
    url = 'https://www.bilibili.com/video/BV1Tt411P72Q/'
    # Playlist
    # Bilibili 【TED演讲/Ed合集】精选50篇-对应文稿第1-50篇【无字幕】
    # url = 'https://www.bilibili.com/video/BV1YQ4y1371P/'
    url = 'https://www.bilibili.com/video/BV1kr421M7vz/' # (英文无字幕) 阿里这小子在水城威尼斯发来问候
    folder_path = 'videos'
    os.makedirs(folder_path, exist_ok=True)
    download_from_url(url, folder_path)
