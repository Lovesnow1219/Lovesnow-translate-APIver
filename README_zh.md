# Lovesnow-translate

[English](README.md) | [繁體中文](README_zh.md)

雲端 API 影片配音／翻譯 WebUI。

人聲分離走 Replicate Demucs，辨識／翻譯／審稿走 OpenAI，配音走 Fish Audio，合成用本機 `ffmpeg`。

金鑰只放本機 `.env`（已加入 gitignore），不要提交。

**倉庫：** https://github.com/Lovesnow1219/Lovesnow-translate-APIver

## 環境

- Windows、macOS 或 Linux
- Python 3.10
- 系統 `PATH` 上有 `ffmpeg`
- OpenAI、Fish Audio、Replicate 帳號與金鑰

## 安裝

```bash
git clone https://github.com/Lovesnow1219/Lovesnow-translate-APIver.git Lovesnow-translate
cd Lovesnow-translate
git checkout api-cloud-webui
python -m venv .venv
```

Windows：

```bat
.\.venv\Scripts\pip install -r requirements.txt
.\.venv\Scripts\python.exe webui.py
```

macOS / Linux：

```bash
.venv/bin/pip install -r requirements.txt
.venv/bin/python webui.py
```

瀏覽器打開 http://127.0.0.1:6006

1. 到 **API 設定** 填金鑰並儲存。
2. 到 **一鍵自動化** 跑影片。

Windows 一鍵便攜包（內建 Python + ffmpeg）：

```bat
.\.venv\Scripts\python.exe packaging\build_portable.py
```

## 金鑰怎麼取得

把 `env.example` 複製成 `.env`，或在 WebUI「API 設定」貼上後儲存。不要把 `.env` 提交或傳給別人。

| 變數 | 用途 | 申請頁 |
| --- | --- | --- |
| `OPENAI_API_KEY` | 辨識、翻譯、審稿 | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) |
| `FISH_API_KEY` | 配音 | [fish.audio/app/api-keys](https://fish.audio/app/api-keys) |
| `REPLICATE_API_TOKEN` | 人聲分離 | [replicate.com/account/api-tokens](https://replicate.com/account/api-tokens) |

### OpenAI

1. 到 [platform.openai.com/signup](https://platform.openai.com/signup) 註冊。這是 **API 平台**，不是 ChatGPT Plus。
2. 在 [Billing](https://platform.openai.com/settings/organization/billing) 綁信用卡。
3. 打開 [API keys](https://platform.openai.com/api-keys) → **Create new secret key**。
4. 複製一次出現的 `sk-…`，填進 `OPENAI_API_KEY`。

第二、三把 OpenAI（`OPENAI_TRANSLATE_API_KEY`、`OPENAI_TRANSLATE_API_KEY_2`）可選，只加速翻譯。審稿固定用主 key。

### Fish Audio

1. 到 [fish.audio/auth/signup](https://fish.audio/auth/signup) 註冊並驗證信箱。
2. 打開 [API keys](https://fish.audio/app/api-keys) → **Create New Key**。
3. 複製後填 `FISH_API_KEY`。官方說明：[docs.fish.audio — API key](https://docs.fish.audio/developer-guide/getting-started/api-key)。

第二把 Fish（`FISH_API_KEY_2`）可選，只加配音併發。

### Replicate

1. 到 [replicate.com](https://replicate.com) 註冊。
2. 打開 [API tokens](https://replicate.com/account/api-tokens) → **Create token**。
3. 複製 `r8_…` 填 `REPLICATE_API_TOKEN`。

費用算在這三個帳號。WebUI 成本面板只是估計。

## 單次費用預估

以下是一次**一鍵自動化**（中文片源 → 英文配音）的**牌價估計**。若業者還有免費額，帳單會更低（Fish 免費 TTS 檔常顯示 **$0**）。

本管線實測大約（2026）：

| 情況 | 大約總額 | 說明 |
| --- | --- | --- |
| 第一次整集，約 8–12 分鐘 | **US$0.90–1.50**（匯率 32 約 NT$30–50） | 人聲分離＋辨識＋翻譯＋審稿＋配音 |
| 只重翻／重審 | **US$0.40–1.20** | 人聲與辨識已有快取 |
| 長集、對白密、Sol 審稿 | **US$2–4+** | 句子多、審稿輪次多 |

約 10 分鐘、第一次整跑的分項：

| 步驟 | 大約 | 估計器用的牌價 |
| --- | --- | --- |
| 人聲分離（Replicate） | US$0.10–0.20 | T4 約 $0.000225／GPU 秒 |
| 辨識（OpenAI `gpt-4o-transcribe-diarize`） | US$0.03–0.15 | 約 $0.006／音訊分鐘；若 API 回 token 則改算 token |
| 翻譯（OpenAI `gpt-5.6-terra`、`xhigh`） | US$0.45–1.00 | 每百萬 token $2／$12（輸入／輸出） |
| 審稿（OpenAI `gpt-5.6-sol`、`high`） | US$0.07–1.40 | 每百萬 token $4／$20；含紙上、整表品管、配音後 |
| 配音（Fish） | 常為 **US$0** | 免費 `s2.1-pro-free`；付費約 $15／百萬 UTF-8 bytes |
| 合成（`ffmpeg`） | US$0 | 本機 |

跑完後該集資料夾會寫 `cost.json`／`cost.txt`。WebUI「API 設定」可打開這份估計。官方牌價會變；程式內 OpenAI 單價上次核對是 2026-08-27。

## 管線

本機檔或網址 → Demucs → OpenAI 辨識 → 中文大綱＋修句 → OpenAI 翻譯 → 審稿 → Fish 配音 → `ffmpeg` 合成。

每一集在 `videos/<劇名>/`。劇情只活在該集的 `source_bible.json`／`summary.json`，不寫進 `tools/`。成片檔名是 `{資料夾名}_{Language}.mp4`。

## 資料夾

| 路徑 | 用途 |
| --- | --- |
| `webui.py` | Gradio（API 設定、一鍵、分步） |
| `tools/` | 管線步驟 |
| `.env` | 本機金鑰（已 gitignore） |
| `videos/` | 各集工作檔與成片（已 gitignore） |
| `packaging/` | Windows 便攜打包 |
| `font/` | 字幕字型 |

## 授權

見 `NOTICE`。這棵樹是 Lovesnow-translate 的雲端 API 版本。
