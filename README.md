# Lovesnow-translate

Cloud-API video dubbing and translation WebUI.

人聲分離用 Replicate Demucs，辨識／翻譯／審稿用 OpenAI，配音用 Fish Audio。金鑰只放本機 `.env`，不要提交。

## 用法

```bash
git clone https://github.com/Lovesnow1219/Lovesnow-translate-APIver.git Lovesnow-translate
cd Lovesnow-translate
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
.\.venv\Scripts\python.exe webui.py
```

瀏覽器打開 http://127.0.0.1:6006  
先到 **API 設定** 填金鑰，再到 **一鍵自動化** 跑影片。

需要系統上的 `ffmpeg`。Windows 也可打便攜包：`python packaging/build_portable.py`。

## 金鑰

複製 `env.example` 成 `.env`，或只在 WebUI 裡儲存：

- `OPENAI_API_KEY` — 辨識、審稿、翻譯
- `FISH_API_KEY` — 配音
- `REPLICATE_API_TOKEN` — 人聲分離

## 資料夾

| 路徑 | 用途 |
| --- | --- |
| `webui.py` | Gradio 介面 |
| `tools/` | 入片 → 分離 → 辨識 → 翻譯 → 審稿 → 配音 → 合成 |
| `.env` | 本機金鑰（已 gitignore） |
| `videos/劇名/` | 那一集的中間檔與成片 |
| `packaging/` | Windows 一鍵便攜包腳本 |

劇情大綱只活在該集的 `source_bible.json`／`summary.json`，不寫進 `tools/`。
