# Lovesnow-translate

[English](README.md) | [繁體中文](README_zh.md)

Cloud-API video dubbing and translation WebUI.

Vocal separation runs on Replicate Demucs. Recognition, translation, and review use OpenAI. TTS uses Fish Audio. Muxing uses local `ffmpeg`.

API keys stay in a local `.env` (gitignored). Never commit them.

**Repo:** https://github.com/Lovesnow1219/Lovesnow-translate-APIver

## Requirements

- Windows, macOS, or Linux
- Python 3.10
- `ffmpeg` on `PATH`
- Accounts and keys for OpenAI, Fish Audio, and Replicate

## Setup

```bash
git clone https://github.com/Lovesnow1219/Lovesnow-translate-APIver.git Lovesnow-translate
cd Lovesnow-translate
git checkout api-cloud-webui
python -m venv .venv
```

Windows:

```bat
.\.venv\Scripts\pip install -r requirements.txt
.\.venv\Scripts\python.exe webui.py
```

macOS / Linux:

```bash
.venv/bin/pip install -r requirements.txt
.venv/bin/python webui.py
```

Open http://127.0.0.1:6006

1. Fill keys on **API 設定** and save.
2. Run a video on **一鍵自動化**.

Windows one-click zip (bundles Python + ffmpeg):

```bat
.\.venv\Scripts\python.exe packaging\build_portable.py
```

## Keys

Copy `env.example` to `.env`, or paste the keys in the WebUI tab **API 設定** and save. Never commit `.env`.

| Variable | Used for | Where to get it |
| --- | --- | --- |
| `OPENAI_API_KEY` | ASR, translation, review | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) |
| `FISH_API_KEY` | Dubbing | [fish.audio/app/api-keys](https://fish.audio/app/api-keys) |
| `REPLICATE_API_TOKEN` | Vocal separation | [replicate.com/account/api-tokens](https://replicate.com/account/api-tokens) |

### OpenAI

1. Create an account at [platform.openai.com/signup](https://platform.openai.com/signup). This is the **API platform**, not a ChatGPT Plus login.
2. Add billing at [platform.openai.com/settings/organization/billing](https://platform.openai.com/settings/organization/billing).
3. Open [API keys](https://platform.openai.com/api-keys) → **Create new secret key**.
4. Copy the `sk-…` value once and put it in `OPENAI_API_KEY`.

Optional second/third OpenAI keys (`OPENAI_TRANSLATE_API_KEY`, `OPENAI_TRANSLATE_API_KEY_2`) only speed up translation. Review always uses the main key.

### Fish Audio

1. Sign up at [fish.audio/auth/signup](https://fish.audio/auth/signup) and verify email.
2. Open [API keys](https://fish.audio/app/api-keys) → **Create New Key**.
3. Copy the key into `FISH_API_KEY`. Official guide: [docs.fish.audio — API key](https://docs.fish.audio/developer-guide/getting-started/api-key).

A second Fish account key (`FISH_API_KEY_2`) is optional and only adds TTS concurrency.

### Replicate

1. Sign up at [replicate.com](https://replicate.com).
2. Open [API tokens](https://replicate.com/account/api-tokens) → **Create token**.
3. Copy the `r8_…` value into `REPLICATE_API_TOKEN`.

Usage is billed on those three accounts. The WebUI cost panel is only an estimate.

## Cost per run

These are **list-price estimates** for one **one-click** job (Chinese source → English dub). Your invoice can be lower if a provider still has free quota (Fish often shows **$0** on the free TTS tier).

Typical measured totals on this pipeline (2026):

| Job | Typical total | When |
| --- | --- | --- |
| First full run, ~8–12 min episode | **US$0.90–1.50** (about NT$30–50 at 32 FX) | Demucs + ASR + translate + review + TTS |
| Retranslate / review only | **US$0.40–1.20** | Vocals and ASR already cached |
| Long or talk-heavy episode, Sol review | **US$2–4+** | More lines and more review passes |

Rough split on a first ~10 min run:

| Step | Typical | List rate used by the estimator |
| --- | --- | --- |
| Demucs (Replicate) | US$0.10–0.20 | T4 about $0.000225 / GPU-second |
| ASR (OpenAI `gpt-4o-transcribe-diarize`) | US$0.03–0.15 | about $0.006 / audio minute, or token billing if the API returns tokens |
| Translate (OpenAI `gpt-5.6-terra`, `xhigh`) | US$0.45–1.00 | $2 / $12 per 1M input / output tokens |
| Review (OpenAI `gpt-5.6-sol`, `high`) | US$0.07–1.40 | $4 / $20 per 1M tokens; paper + shipping QC + after-dub |
| TTS (Fish) | often **US$0** | free `s2.1-pro-free`; paid is $15 / 1M UTF-8 bytes |
| Mux (`ffmpeg`) | US$0 | local |

After a run, the episode folder writes `cost.json` / `cost.txt`. The WebUI **API 設定** tab can open that estimate. Official model prices change; the code last checked OpenAI list rates on 2026-08-27.

## Pipeline

Local file or URL → Demucs → OpenAI ASR → Chinese outline + repair → OpenAI translation → review → Fish TTS → `ffmpeg` mux.

Each episode lives in `videos/<title>/`. Plot notes stay in that folder’s `source_bible.json` and `summary.json`, not in `tools/`. The finished file is `{folder}_{Language}.mp4`.

## Layout

| Path | Role |
| --- | --- |
| `webui.py` | Gradio UI (API settings, one-click, step-by-step) |
| `tools/` | Pipeline steps |
| `.env` | Local keys (gitignored) |
| `videos/` | Per-episode working files and output (gitignored) |
| `packaging/` | Windows portable packer |
| `font/` | Subtitle font |

## License

See `NOTICE`. This tree is the cloud-API fork of Lovesnow-translate.
