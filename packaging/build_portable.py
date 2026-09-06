# -*- coding: utf-8 -*-
"""Build a one-click Windows portable zip (bundled Python + ffmpeg, no secrets)."""
import os
import shutil
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = Path(__file__).resolve().parent / 'cache'
PYTHON_URL = 'https://www.python.org/ftp/python/3.10.11/python-3.10.11-embed-amd64.zip'
GET_PIP_URL = 'https://bootstrap.pypa.io/get-pip.py'
FFMPEG_URL = 'https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip'
STAMP = time.strftime('%Y-%m-%d')


def log(message):
    print(message, flush=True)


def download(url, dest):
    dest = Path(dest)
    if dest.is_file() and dest.stat().st_size > 1024:
        log(f'已有快取：{dest.name}')
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + '.part')
    log(f'下載 {url}')
    req = urllib.request.Request(url, headers={'User-Agent': 'Lovesnow-translate-packager'})
    with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, 'wb') as handle:
        shutil.copyfileobj(resp, handle)
    tmp.replace(dest)
    return dest


def unzip(archive, dest):
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(dest)


def copy_project(dest):
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for name in ('webui.py', 'requirements.txt', 'env.example', 'NOTICE', '.gitignore'):
        src = ROOT / name
        if src.is_file():
            shutil.copy2(src, dest / name)

    def ignore(_dir, names):
        skip = {
            '__pycache__', '.git', '.venv', 'venv', 'videos', 'temp', 'tmp',
            'flagged', 'models', 'packaging', '.idea',
        }
        return [name for name in names if name in skip or name.endswith('.pyc')]

    shutil.copytree(ROOT / 'tools', dest / 'tools', dirs_exist_ok=True, ignore=ignore)
    if (ROOT / 'assets').is_dir():
        shutil.copytree(ROOT / 'assets', dest / 'assets', dirs_exist_ok=True, ignore=ignore)
    if (ROOT / 'font').is_dir():
        shutil.copytree(ROOT / 'font', dest / 'font', dirs_exist_ok=True, ignore=ignore)
    rules_src = ROOT / '.cursor' / 'rules' / 'dubbing-pipeline.mdc'
    rules_dst = dest / '.cursor' / 'rules'
    if rules_src.is_file():
        rules_dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(rules_src, rules_dst / 'dubbing-pipeline.mdc')
    (dest / 'videos').mkdir(exist_ok=True)
    (dest / 'temp').mkdir(exist_ok=True)
    shutil.copy2(Path(__file__).with_name('launch_webui.py'), dest / 'launch_webui.py')
    shutil.copy2(Path(__file__).with_name('start.bat'), dest / '啟動 Lovesnow.bat')
    shutil.copy2(Path(__file__).with_name('start.bat'), dest / 'Start-Lovesnow.bat')
    shutil.copy2(Path(__file__).with_name('README_FRIEND.txt'), dest / '給朋友.txt')
    shutil.copy2(Path(__file__).with_name('README_FRIEND.txt'), dest / 'README.txt')
    (dest / 'VERSION.txt').write_text(
        f'Lovesnow-translate portable\nbuilt {STAMP}\npython 3.10.11\n',
        encoding='utf-8',
    )
    env_path = dest / '.env'
    if env_path.exists():
        env_path.unlink()


def enable_site(python_dir):
    pth = next(Path(python_dir).glob('python*._pth'), None)
    if not pth:
        raise RuntimeError('embeddable Python 缺少 ._pth')
    text = pth.read_text(encoding='utf-8').replace('#import site', 'import site')
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    if '../..' not in text:
        lines = text.split('\n')
        out = []
        inserted = False
        for line in lines:
            out.append(line)
            if line.strip() == '.' and not inserted:
                out.append('../..')
                inserted = True
        if not inserted:
            out.insert(0, '../..')
        text = '\n'.join(out)
    if 'import site' not in text:
        text += '\nimport site\n'
    if not text.endswith('\n'):
        text += '\n'
    pth.write_text(text, encoding='utf-8')
    site_custom = Path(python_dir) / 'lib' / 'site-packages' / 'sitecustomize.py'
    site_custom.parent.mkdir(parents=True, exist_ok=True)
    site_custom.write_text(
        'import sys\n'
        'from pathlib import Path\n'
        'root = Path(__file__).resolve().parents[4]\n'
        'root_s = str(root)\n'
        'if root_s not in sys.path:\n'
        '    sys.path.insert(0, root_s)\n',
        encoding='utf-8',
    )


def run(cmd, cwd=None):
    import subprocess

    log('> ' + ' '.join(str(part) for part in cmd))
    result = subprocess.run(list(map(str, cmd)), cwd=cwd)
    if result.returncode != 0:
        raise RuntimeError(f'命令失敗：{cmd}')


def install_python(runtime_dir):
    python_dir = Path(runtime_dir) / 'python'
    if not (python_dir / 'python.exe').is_file():
        archive = download(PYTHON_URL, CACHE / 'python-3.10.11-embed-amd64.zip')
        if python_dir.exists():
            shutil.rmtree(python_dir)
        unzip(archive, python_dir)
    enable_site(python_dir)
    get_pip = download(GET_PIP_URL, CACHE / 'get-pip.py')
    python = python_dir / 'python.exe'
    if not (python_dir / 'Scripts' / 'pip.exe').is_file():
        run([python, get_pip, '--no-warn-script-location'])
    run([
        python, '-m', 'pip', 'install', '--upgrade', 'pip', 'wheel',
        '--no-warn-script-location',
    ])
    run([
        python, '-m', 'pip', 'install', '--no-warn-script-location',
        '-r', str(ROOT / 'requirements.txt'),
    ])
    import subprocess
    subprocess.run(
        [str(python), '-m', 'pip', 'uninstall', '-y', 'hf-gradio'],
        check=False,
    )
    return python


def install_ffmpeg(runtime_dir):
    ffmpeg_dir = Path(runtime_dir) / 'ffmpeg'
    ffmpeg_dir.mkdir(parents=True, exist_ok=True)
    dest_ff = ffmpeg_dir / 'ffmpeg.exe'
    dest_probe = ffmpeg_dir / 'ffprobe.exe'
    if dest_ff.is_file() and dest_probe.is_file():
        return
    try:
        archive = download(FFMPEG_URL, CACHE / 'ffmpeg-release-essentials.zip')
        extract = CACHE / 'ffmpeg-essentials'
        if extract.exists():
            shutil.rmtree(extract)
        unzip(archive, extract)
        bins = list(extract.rglob('ffmpeg.exe'))
        if not bins:
            raise RuntimeError('essentials zip 裡沒有 ffmpeg.exe')
        bin_dir = bins[0].parent
        shutil.copy2(bin_dir / 'ffmpeg.exe', dest_ff)
        probe = bin_dir / 'ffprobe.exe'
        if probe.is_file():
            shutil.copy2(probe, dest_probe)
    except Exception as exc:
        log(f'下載 essentials 失敗，改複製本機 ffmpeg：{exc}')
        which = shutil.which('ffmpeg')
        probe = shutil.which('ffprobe')
        if not which or not probe:
            raise RuntimeError('找不到 ffmpeg／ffprobe') from exc
        shutil.copy2(which, dest_ff)
        shutil.copy2(probe, dest_probe)


def verify(python, dest):
    run([
        python, '-c',
        'import gradio, librosa, openai, dotenv, tools.do_everything, webui; '
        'assert gradio.__version__.startswith("4."), gradio.__version__; '
        'print("import ok", gradio.__version__)',
    ], cwd=dest)


def zip_folder(src, zip_path):
    src = Path(src)
    zip_path = Path(zip_path)
    if zip_path.exists():
        zip_path.unlink()
    log(f'壓縮 {zip_path}')
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in src.rglob('*'):
            if not path.is_file() or path.name == '.env':
                continue
            rel = Path(src.name) / path.relative_to(src)
            info = zipfile.ZipInfo(str(rel).replace('\\', '/'))
            info.flag_bits |= 0x800
            info.compress_type = zipfile.ZIP_DEFLATED
            info.date_time = time.localtime(path.stat().st_mtime)[:6]
            zf.writestr(info, path.read_bytes(), compresslevel=6)


def default_out_dir():
    desktop = Path.home() / 'Desktop'
    if not desktop.is_dir():
        desktop = Path.home() / 'OneDrive' / 'Desktop'
    if not desktop.is_dir():
        desktop = Path.home()
    return desktop / 'Lovesnow-translate-portable'


def main():
    args = [arg for arg in sys.argv[1:] if not arg.startswith('-')]
    reuse = '--reuse' in sys.argv
    out_dir = Path(args[0]) if args else default_out_dir()
    if out_dir.exists() and not reuse:
        log(f'清掉舊資料夾 {out_dir}')
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f'輸出：{out_dir}')
    copy_project(out_dir)
    runtime = out_dir / 'runtime'
    python = install_python(runtime)
    install_ffmpeg(runtime)
    verify(python, out_dir)
    zip_path = out_dir.with_name(out_dir.name + '.zip')
    zip_folder(out_dir, zip_path)
    size_mb = zip_path.stat().st_size / (1024 * 1024)
    log(f'完成：{zip_path} ({size_mb:.0f} MB)')
    log('資料夾也可直接點「啟動 Lovesnow.bat」試跑。沒有打包 .env。')


if __name__ == '__main__':
    main()
