# -*- coding: utf-8 -*-
"""Start the local WebUI and open the browser when it is ready."""
import os
import sys
import threading
import time
import webbrowser

ROOT = os.path.dirname(os.path.abspath(__file__))
RUNTIME_PYTHON = os.path.join(ROOT, 'runtime', 'python')
RUNTIME_FFMPEG = os.path.join(ROOT, 'runtime', 'ffmpeg')
URL = 'http://127.0.0.1:6006/'


def _prepend_path(*parts):
    extra = [part for part in parts if part and os.path.isdir(part)]
    if not extra:
        return
    os.environ['PATH'] = os.pathsep.join(extra + [os.environ.get('PATH', '')])


def _wait_and_open():
    import urllib.request

    for _ in range(90):
        try:
            urllib.request.urlopen(URL, timeout=1)
            webbrowser.open(URL)
            return
        except Exception:
            time.sleep(0.5)


def main():
    os.chdir(ROOT)
    sys.path.insert(0, ROOT)
    os.environ['PYTHONPATH'] = ROOT
    os.environ['PYTHONNOUSERSITE'] = '1'
    os.environ['NO_PROXY'] = '127.0.0.1,localhost,::1'
    os.environ['no_proxy'] = '127.0.0.1,localhost,::1'
    _prepend_path(
        RUNTIME_PYTHON,
        os.path.join(RUNTIME_PYTHON, 'Scripts'),
        RUNTIME_FFMPEG,
    )
    os.makedirs(os.path.join(ROOT, 'videos'), exist_ok=True)
    threading.Thread(target=_wait_and_open, daemon=True).start()
    print('Lovesnow-translate 啟動中…')
    print('瀏覽器會打開', URL)
    print('第一次請先到「API 設定」填金鑰。關掉這個視窗就會停止。')
    print()
    import webui  # noqa: F401  (webui launches on import of __main__ only)

    if hasattr(webui, 'app'):
        webui.app.queue(default_concurrency_limit=4)
        assets = os.path.join(ROOT, 'assets')
        webui.app.launch(
            server_name='127.0.0.1',
            server_port=6006,
            share=False,
            inbrowser=False,
            show_error=True,
            show_api=False,
            allowed_paths=[assets],
        )


if __name__ == '__main__':
    try:
        import gradio.networking as _net
        _net.url_ok = lambda url: True
    except Exception:
        pass
    main()
