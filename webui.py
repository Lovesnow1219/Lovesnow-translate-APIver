import gradio as gr
import gradio_client.utils as _gcu

_orig_schema_type = _gcu._json_schema_to_python_type

def _safe_schema_type(schema, defs):
    if schema is True or schema == {}:
        return "Any"
    if schema is False or schema is None:
        return "None"
    if not isinstance(schema, dict):
        return "Any"
    return _orig_schema_type(schema, defs)

_gcu._json_schema_to_python_type = _safe_schema_type

from tools.download import download_from_url
from tools.demucs import separate_all_audio_under_folder
from tools.asr import transcribe_all_audio_under_folder
from tools.translation import translate_all_transcript_under_folder
from tools.tts import generate_all_wavs_under_folder
from tools.synthesize import synthesize_all_video_under_folder
from tools.do_everything import stream_do_everything
from tools.api_settings_ui import build_api_settings_ui
from tools.speaker_edit import (
    _speaker_row_fields,
    episode_folder,
    load_bible_fields,
    load_speaker_table,
    parse_one_line_payload,
    redub_speakers,
    retranslate_checked_lines,
    retranslate_from_bible,
    retranslate_one_line,
    save_bible_fields,
    save_speaker_table,
    selected_line_indices,
)
from tools.target_language import SOURCE_LANGUAGES, TARGET_LANGUAGES, split_target_language, ui_language_label
from tools.translation_review import (
    apply_selected_review,
    review_folder,
    review_pick_choices,
    review_status_line,
    review_suggested_values,
    review_table_rows,
)
from tools.api_settings import (
    REASONING_EFFORTS,
    TRANSLATION_MODELS,
    apply_model_choices,
    current_model_value,
)
from tools.video_folders import folder_dropdown, folder_picker, list_video_folders, refresh_folder_dropdown
from dotenv import load_dotenv
load_dotenv(override=False)

_RESOLUTIONS = ['4320p', '2160p', '1440p', '1080p', '720p', '480p', '360p', '240p', '144p']
DEMUCS_METHOD = 'Replicate'
ASR_METHOD = 'OpenAI'
ASR_MODEL = 'gpt-4o-transcribe-diarize'
TRANSLATE_METHOD = 'OpenAI'
TTS_METHOD = 'Fish'


def _ui_theme():
    from gradio.themes.utils import colors, fonts, sizes
    peach = colors.Color(
        name='peach',
        c50='#fffaf2',
        c100='#fff1dd',
        c200='#fcd49d',
        c300='#f4b183',
        c400='#e89a5c',
        c500='#d4894a',
        c600='#c06f38',
        c700='#9a552c',
        c800='#7a4324',
        c900='#5e4534',
        c950='#3d2c22',
    )
    cocoa = colors.Color(
        name='cocoa',
        c50='#fbf7f1',
        c100='#f3ebe0',
        c200='#e6d3bc',
        c300='#d4b896',
        c400='#b89570',
        c500='#9a734f',
        c600='#7d5b3f',
        c700='#5e4534',
        c800='#463328',
        c900='#2f221b',
        c950='#1c1410',
    )
    return gr.themes.Soft(
        primary_hue=peach,
        secondary_hue=peach,
        neutral_hue=cocoa,
        radius_size=sizes.radius_lg,
        font=(
            fonts.GoogleFont('Nunito'),
            'Microsoft JhengHei',
            'PingFang TC',
            'ui-sans-serif',
            'sans-serif',
        ),
    ).set(
        body_background_fill='#fffaf2',
        body_text_color='#5e4534',
        block_background_fill='#fffdf8',
        block_border_color='#f0dcc4',
        block_border_width='1px',
        block_label_background_fill='#fff1dd',
        block_label_text_color='#7a4324',
        block_title_background_fill='#fff1dd',
        block_title_text_color='#7a4324',
        input_background_fill='#fffefb',
        input_border_color='#edd6b8',
        button_primary_background_fill='#d4894a',
        button_primary_background_fill_hover='#e89a5c',
        button_primary_text_color='#fffaf2',
        button_secondary_background_fill='#fff1dd',
        button_secondary_background_fill_hover='#fcd49d',
        button_secondary_text_color='#5e4534',
        button_cancel_background_fill='#c45c4a',
        button_cancel_background_fill_hover='#d67a6a',
        button_cancel_text_color='#fffaf2',
        slider_color='#d4894a',
        checkbox_background_color_selected='#d4894a',
        checkbox_border_color_selected='#c06f38',
        checkbox_label_background_fill_selected='#d4894a',
        link_text_color='#c06f38',
        table_even_background_fill='#fffdf8',
        table_odd_background_fill='#fff6e8',
        table_border_color='#f0dcc4',
        panel_background_fill='#fffaf2',
        shadow_drop='0 1px 4px rgba(94, 69, 52, 0.08)',
        shadow_drop_lg='0 2px 8px rgba(94, 69, 52, 0.10)',
    )


my_theme = _ui_theme()
api_settings_interface = build_api_settings_ui(theme=my_theme)


def _resolution(label='解析度'):
    return gr.Dropdown(choices=_RESOLUTIONS, value='1080p', label=label)


def _accordion(title):
    return gr.Accordion(title, open=False)


def _shifts():
    return gr.Slider(
        minimum=1,
        maximum=10,
        step=1,
        value=1,
        label='移位次數 Number of shifts',
        info='提高會讓人聲更乾淨，講者比較不容易被殘留配樂或切片爆音拆亂；時間與費用幾乎倍增。有背景音樂的片子建議 2，人聲裡還聽得到樂器或多出莫名講者再試 5。變更後會重跑分離與識別。預設 1。',
    )


def _source_language(label='原片語言', info=None):
    return gr.Dropdown(
        choices=SOURCE_LANGUAGES,
        value='中文',
        label=label,
        info=info or '原片對白語言，給語音識別用。預設中文。',
        allow_custom_value=False,
        filterable=False,
    )


def _target_language(label='目標語言', info=None):
    return gr.Dropdown(
        choices=TARGET_LANGUAGES,
        value='English',
        label=label,
        info=info or '完整支援：英文、日文、越南文、中文／粵語。標「部分支援」的語言可翻可配，但語氣詞與漏句規則尚未對齊。',
        allow_custom_value=False,
        filterable=False,
    )


def _model_dropdown(env_key, label, info):
    return gr.Dropdown(
        choices=TRANSLATION_MODELS if env_key.endswith('MODEL_NAME') else REASONING_EFFORTS,
        value=current_model_value(env_key),
        label=label,
        info=info,
        allow_custom_value=False,
        filterable=False,
    )


def _translation_model():
    return _model_dropdown(
        'MODEL_NAME',
        '翻譯模型',
        'luna 量大較快；terra 較穩；sol 最嚴也最慢最貴。',
    )


def _translation_effort():
    return _model_dropdown(
        'OPENAI_REASONING_EFFORT',
        '翻譯推理',
        '翻譯建議 xhigh。sol 開 max 會更慢更貴。',
    )


def _review_model():
    return _model_dropdown(
        'REVIEW_MODEL_NAME',
        '審稿模型',
        '旗艦審稿用 sol，跟桌面編輯同一套標準。',
    )


def _review_effort():
    return _model_dropdown(
        'REVIEW_REASONING_EFFORT',
        '審稿推理',
        'sol 用 high。整表審稿，不要降到 medium。',
    )


def _review_pick_update(folder, language=None, selected=None):
    choices = review_pick_choices(folder, language)
    valid = {str(value) for _label, value in choices}
    if selected is None:
        kept = []
    else:
        kept = [str(item) for item in selected if str(item) in valid]
    return gr.update(choices=choices, value=kept)


def do_everything_with_cost(
    root_folder, url, local_file, dl_res, shifts, source_lang, target_lang,
    translation_model, translation_effort, review_model, review_effort,
    subtitles, speed_up, fps, out_res, bgm, bgm_vol, video_vol,
    words_per_sec, translate_workers, force_retranslate,
):
    for status, video, cost_md in stream_do_everything(
        root_folder, url,
        dl_res, shifts, target_lang, subtitles, speed_up, fps, out_res,
        1, 3, bgm, bgm_vol, video_vol,
        local_file=local_file,
        words_per_sec=words_per_sec,
        translate_workers=translate_workers,
        force_retranslate=force_retranslate,
        translation_model=translation_model,
        translation_effort=translation_effort,
        review_model=review_model,
        review_effort=review_effort,
        source_language=source_lang,
    ):
        yield status, (gr.update() if video is None else video), cost_md


def demucs_from_ui(folder, progress, shifts):
    return separate_all_audio_under_folder(folder, model_name=DEMUCS_METHOD, progress=progress, shifts=shifts)


def asr_from_ui(folder, source_language):
    return transcribe_all_audio_under_folder(folder, ASR_METHOD, ASR_MODEL, language=source_language)


def translate_from_ui(folder, language, translation_model, translation_effort):
    apply_model_choices(translation_model=translation_model, translation_effort=translation_effort)
    trans_lang, _ = split_target_language(language)
    return translate_all_transcript_under_folder(folder, TRANSLATE_METHOD, trans_lang)


def tts_from_ui(folder, language):
    _, tts_lang = split_target_language(language)
    return generate_all_wavs_under_folder(folder, TTS_METHOD, tts_lang)


def _bgm_fields(filepath=False):
    with gr.Accordion('背景音樂（可選，通常不必調）', open=False):
        kwargs = {'label': '背景音樂', 'sources': ['upload']}
        if filepath:
            kwargs['type'] = 'filepath'
        bgm = gr.Audio(**kwargs)
        bgm_vol = gr.Slider(minimum=0, maximum=1, step=0.05, label='背景音樂音量', value=0.5)
        video_vol = gr.Slider(minimum=0, maximum=1, step=0.05, label='影片音量', value=1.0)
    return bgm, bgm_vol, video_vol


def _page_background_css():
    import base64
    from pathlib import Path
    path = Path(__file__).resolve().parent / 'assets' / 'webui-bg.jpg'
    if not path.is_file():
        return ''
    uri = 'data:image/jpeg;base64,' + base64.b64encode(path.read_bytes()).decode('ascii')
    return f"""
html, body, .gradio-container, gradio-app {{
  background-color: #fffaf2 !important;
}}
body::before,
.gradio-container::before {{
  content: "";
  position: fixed;
  inset: 0;
  z-index: 0;
  pointer-events: none;
  background-image: url("{uri}");
  background-repeat: no-repeat;
  background-position: center 38%;
  background-size: min(70vmin, 720px);
  opacity: 0.32;
  mix-blend-mode: multiply;
}}
.gradio-container .contain,
.gradio-container > .main,
.gradio-container .wrap,
.gradio-container .tabs,
.gradio-container .tabitem {{
  background: transparent !important;
}}
.gradio-container {{
  position: relative;
  z-index: 1;
}}
"""


_UI_CSS = _page_background_css() + """
.tab-nav {
  background: rgba(255, 250, 242, 0.78) !important;
  border-bottom: 1px solid #f0dcc4 !important;
  gap: 8px !important;
  padding: 10px 10px 0 !important;
}
.tab-nav button {
  background: #fff1dd !important;
  color: #5e4534 !important;
  border: 1px solid #f0dcc4 !important;
  border-radius: 18px 18px 0 0 !important;
  font-weight: 700 !important;
}
.tab-nav button.selected,
.tab-nav button[aria-selected="true"] {
  background: #d4894a !important;
  color: #fffaf2 !important;
  border-color: #d4894a !important;
}
.markdown h1, .markdown h2, .markdown h3,
.prose h1, .prose h2, .prose h3,
.gradio-container h1, .gradio-container h2, .gradio-container h3 {
  color: #5e4534 !important;
}
label, .block-info, .form .info {
  color: #7a4324 !important;
}
.accordion {
  background: rgba(255, 253, 248, 0.88) !important;
  border: 1px solid #f0dcc4 !important;
  border-radius: 16px !important;
}
footer, footer a, .footer {
  color: #b89570 !important;
}
#live-cost-panel {
  max-height: min(280px, 36vh);
  overflow: auto !important;
  border: 1px solid var(--border-color-primary);
  border-radius: 8px;
  padding: 2px 10px 8px;
}
#live-cost-panel,
#live-cost-panel .md,
#live-cost-panel .prose,
#live-cost-panel * {
  max-width: 100%;
  overflow-wrap: anywhere;
  word-break: break-word;
}
#live-cost-panel pre, #live-cost-panel code {
  white-space: pre-wrap;
  word-break: break-all;
}
#review-suggest-row {
  align-items: flex-start !important;
}
#review-apply-col {
  padding-top: 1.85em;
  flex: 0 0 auto !important;
}
#review-suggest-table,
#review-suggest-table .table-wrap {
  min-height: 360px !important;
  max-height: 360px !important;
}
#review-pick,
#review-pick .wrap,
#review-pick .form {
  min-height: 132px;
  max-height: 220px;
  overflow-y: auto !important;
}
#review-pick [data-testid="checkbox-group"] {
  display: flex !important;
  flex-direction: column !important;
  align-items: stretch !important;
  flex-wrap: nowrap !important;
  gap: 2px !important;
}
#review-pick label {
  background: transparent !important;
  border: none !important;
  border-radius: 0 !important;
  box-shadow: none !important;
  padding: 3px 0 !important;
  margin: 0 !important;
}
#review-pick input[type="checkbox"] {
  -webkit-appearance: checkbox !important;
  appearance: checkbox !important;
  width: 16px !important;
  height: 16px !important;
  min-width: 16px !important;
  min-height: 16px !important;
  margin: 0 8px 0 0 !important;
  border: 1.5px solid #6b7280 !important;
  border-radius: 3px !important;
  background: #fff !important;
  accent-color: var(--color-accent, #d4894a) !important;
  flex: 0 0 16px !important;
  opacity: 1 !important;
}
#speaker-table .table-wrap {
  max-height: 520px !important;
}
#speaker-table table th:first-child,
#speaker-table table td:first-child {
  width: 44px !important;
  min-width: 44px !important;
  max-width: 52px !important;
  text-align: center !important;
  vertical-align: middle !important;
  padding-left: 6px !important;
  padding-right: 6px !important;
}
#speaker-table table th:last-child,
#speaker-table table td:last-child {
  width: 72px !important;
  min-width: 72px !important;
  max-width: 80px !important;
  text-align: center !important;
  vertical-align: middle !important;
  padding-left: 4px !important;
  padding-right: 4px !important;
}
#speaker-table td.lovesnow-check-cell {
  cursor: pointer;
}
#speaker-table td.lovesnow-trans-cell {
  cursor: pointer;
}
#speaker-table .lovesnow-row-trans {
  display: inline-flex !important;
  align-items: center;
  justify-content: center;
  min-width: 56px;
  height: 26px;
  padding: 0 8px !important;
  border: 1px solid var(--color-accent, #d4894a) !important;
  border-radius: 10px !important;
  background: #fffdf8 !important;
  color: var(--color-accent, #7a4324) !important;
  font-size: 12px !important;
  line-height: 1 !important;
  cursor: pointer;
}
#speaker-table .lovesnow-row-trans:hover {
  background: var(--color-accent, #d4894a) !important;
  color: #fffaf2 !important;
}
#line-one-trans {
  position: absolute !important;
  left: -9999px !important;
  width: 1px !important;
  height: 1px !important;
  overflow: hidden !important;
}
#speaker-table .lovesnow-row-check {
  -webkit-appearance: checkbox !important;
  appearance: checkbox !important;
  width: 16px !important;
  height: 16px !important;
  min-width: 16px !important;
  min-height: 16px !important;
  margin: 0 auto !important;
  border: 1.5px solid #6b7280 !important;
  border-radius: 3px !important;
  background: #fff !important;
  accent-color: var(--color-accent, #d4894a) !important;
  cursor: pointer;
  opacity: 1 !important;
}
#review-status textarea {
  height: 176px !important;
  min-height: 176px !important;
  max-height: 176px !important;
  overflow-y: auto !important;
  resize: none !important;
}
"""

_UI_JS = """
() => {
  const destEl = () => document.querySelector('#line-pick-ids textarea, #line-pick-ids input');
  const checked = () => {
    if (!(window.__lovesnowChecked instanceof Set)) window.__lovesnowChecked = new Set();
    return window.__lovesnowChecked;
  };
  const normIdx = (value) => {
    const text = String(value == null ? '' : value).trim();
    if (!text) return '';
    const num = parseInt(text, 10);
    return Number.isFinite(num) ? String(num) : '';
  };
  const parseIds = (value) => String(value || '').split(/[,，、\\s]+/).map(normIdx).filter(Boolean);
  const setNativeValue = (el, value) => {
    if (!el) return;
    const proto = el.tagName === 'TEXTAREA'
      ? window.HTMLTextAreaElement.prototype
      : window.HTMLInputElement.prototype;
    const desc = Object.getOwnPropertyDescriptor(proto, 'value');
    if (desc && desc.set) desc.set.call(el, value);
    else el.value = value;
    el.dispatchEvent(new Event('input', { bubbles: true }));
    el.dispatchEvent(new Event('change', { bubbles: true }));
  };
  const writeDest = () => {
    const ids = [...checked()].sort((a, b) => Number(a) - Number(b)).join(',');
    window.__lovesnowWritingDest = true;
    setNativeValue(destEl(), ids);
    setTimeout(() => { window.__lovesnowWritingDest = false; }, 0);
  };
  const remember = (idx, on) => {
    const id = normIdx(idx);
    if (id === '') return;
    window.__lovesnowActiveRow = id;
    if (on) checked().add(id);
    else checked().delete(id);
    writeDest();
  };
  const bindCell = (td) => {
    if (td.dataset.lovesnowBound) return;
    td.dataset.lovesnowBound = '1';
    const block = (e) => {
      e.preventDefault();
      e.stopPropagation();
      if (e.stopImmediatePropagation) e.stopImmediatePropagation();
    };
    ['pointerdown', 'mousedown', 'click'].forEach((type) => {
      td.addEventListener(type, (e) => {
        block(e);
        if (type !== 'pointerdown') return;
        const box = td.querySelector('input.lovesnow-row-check');
        const id = box && box.dataset.index;
        if (!box || id === undefined || id === '') return;
        box.checked = !box.checked;
        remember(id, box.checked);
      }, true);
    });
  };
  const bindIndexCell = (td, idx) => {
    if (!td || td.dataset.lovesnowIdxBound) return;
    td.dataset.lovesnowIdxBound = '1';
    td.addEventListener('pointerdown', () => {
      window.__lovesnowActiveRow = idx;
      if (checked().size === 0) remember(idx, true);
    }, true);
  };
  const readSource = (tr) => {
    const td = tr && tr.children[4];
    if (!td) return '';
    const field = td.querySelector('input, textarea');
    if (field) return String(field.value || '').trim();
    const wrap = td.querySelector('.cell-wrap') || td;
    return String(wrap.innerText || wrap.textContent || '').trim();
  };
  const clickOneTrans = () => {
    const root = document.querySelector('#line-one-trans');
    const btn = root && (root.querySelector('button') || root);
    if (btn) btn.click();
  };
  const bindTrans = (td, tr, idx) => {
    if (!td || td.dataset.lovesnowTransBound) return;
    td.dataset.lovesnowTransBound = '1';
    const block = (e) => {
      e.preventDefault();
      e.stopPropagation();
      if (e.stopImmediatePropagation) e.stopImmediatePropagation();
    };
    ['pointerdown', 'mousedown', 'click'].forEach((type) => {
      td.addEventListener(type, (e) => {
        block(e);
        if (type !== 'click') return;
        window.__lovesnowActiveRow = idx;
        window.__lovesnowOne = { idx, source: readSource(tr) };
        clickOneTrans();
      }, true);
    });
  };
  const patchTransCell = (td, tr, idx) => {
    if (!td) return;
    bindTrans(td, tr, idx);
    td.classList.add('lovesnow-trans-cell');
    let btn = td.querySelector('button.lovesnow-row-trans');
    if (btn) {
      btn.dataset.index = idx;
      return;
    }
    const wrap = td.querySelector('.cell-wrap') || td;
    [...wrap.childNodes].forEach((node) => {
      if (node.nodeType === 3) node.textContent = '';
      else if (node.classList && !node.classList.contains('lovesnow-row-trans')) node.style.display = 'none';
    });
    wrap.classList.add('lovesnow-trans-cell');
    btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'lovesnow-row-trans';
    btn.dataset.index = idx;
    btn.textContent = '單句翻';
    wrap.appendChild(btn);
  };
  const patchRow = (tr) => {
    const td = tr.children[0];
    const idx = normIdx(tr.children[1] && tr.children[1].textContent);
    if (!td || idx === '') return;
    bindCell(td);
    bindIndexCell(tr.children[1], idx);
    let cb = td.querySelector('input.lovesnow-row-check');
    if (cb) {
      cb.dataset.index = idx;
      cb.checked = checked().has(idx);
    } else {
      const wrap = td.querySelector('.cell-wrap') || td;
      wrap.querySelectorAll('span').forEach((span) => { span.style.display = 'none'; });
      td.classList.add('lovesnow-check-cell');
      wrap.classList.add('lovesnow-check-cell');
      cb = document.createElement('input');
      cb.type = 'checkbox';
      cb.className = 'lovesnow-row-check';
      cb.dataset.index = idx;
      cb.checked = checked().has(idx);
      wrap.appendChild(cb);
    }
    if (tr.children.length >= 7) patchTransCell(tr.children[6], tr, idx);
  };
  const patchAll = () => {
    document.querySelectorAll('#speaker-table table tbody tr').forEach(patchRow);
    try {
      const focused = document.querySelector('#speaker-table td.focus, #speaker-table td:focus');
      const row = focused && focused.closest('tr');
      const idx = normIdx(row && row.children[1] && row.children[1].textContent);
      if (idx !== '') window.__lovesnowActiveRow = idx;
    } catch (e) {}
    const dest = destEl();
    if (dest && !dest.dataset.lovesnowBound) {
      dest.dataset.lovesnowBound = '1';
      dest.addEventListener('input', () => {
        if (window.__lovesnowWritingDest) return;
        window.__lovesnowChecked = new Set(parseIds(dest.value));
        document.querySelectorAll('#speaker-table table tbody tr').forEach(patchRow);
      });
    }
  };
  if (!window.__lovesnowLinePickBound) {
    window.__lovesnowLinePickBound = true;
    let timer = 0;
    const watch = () => {
      const root = document.querySelector('#speaker-table');
      if (!root || root.dataset.lovesnowWatched) {
        patchAll();
        return;
      }
      root.dataset.lovesnowWatched = '1';
      const obs = new MutationObserver(() => {
        clearTimeout(timer);
        timer = setTimeout(patchAll, 80);
      });
      obs.observe(root, { childList: true, subtree: true });
      patchAll();
    };
    const ready = new MutationObserver(watch);
    ready.observe(document.documentElement, { childList: true, subtree: true });
    watch();
  }
  patchAll();
}
"""

_RETRANSLATE_JS = """
(folder, language, line_ids, model, effort) => {
  try {
    const add = (bag, value) => {
      const num = parseInt(String(value == null ? '' : value).trim(), 10);
      if (Number.isFinite(num) && num >= 0) bag.add(String(num));
    };
    const ids = new Set();
    if (window.__lovesnowChecked && typeof window.__lovesnowChecked.forEach === 'function') {
      window.__lovesnowChecked.forEach((item) => add(ids, item));
    }
    String(line_ids || '').split(/[,，、\\s]+/).forEach((item) => add(ids, item));
    document.querySelectorAll('#speaker-table input.lovesnow-row-check:checked').forEach((el) => add(ids, el.dataset.index));
    if (ids.size === 0) {
      add(ids, window.__lovesnowActiveRow);
      const focused = document.querySelector('#speaker-table td.focus, #speaker-table td:focus');
      const row = focused && focused.closest('tr');
      if (row && row.children[1]) add(ids, row.children[1].textContent);
    }
    const joined = [...ids].sort((a, b) => Number(a) - Number(b)).join(',');
    return [folder, language, joined, model, effort];
  } catch (e) {
    return [folder, language, line_ids || '', model, effort];
  }
}
"""

_ONE_JS = """
(folder, language, payload, model, effort) => {
  try {
    const one = window.__lovesnowOne || {};
    let idx = one.idx;
    let source = one.source;
    if (idx == null || String(idx).trim() === '') idx = window.__lovesnowActiveRow;
    if (source == null || String(source).trim() === '') {
      const focused = document.querySelector('#speaker-table td.focus, #speaker-table td:focus');
      const row = focused && focused.closest('tr');
      const srcTd = row && row.children[4];
      const field = srcTd && srcTd.querySelector('input, textarea');
      source = field ? field.value : (srcTd ? srcTd.innerText : '');
      if ((idx == null || String(idx).trim() === '') && row && row.children[1]) {
        idx = row.children[1].textContent;
      }
    }
    return [folder, language, String(idx == null ? '' : idx).trim() + '\\n' + String(source || '').trim(), model, effort];
  } catch (e) {
    return [folder, language, payload || '', model, effort];
  }
}
"""

# 一鍵自動化介面
with gr.Blocks(theme=my_theme, css=_UI_CSS) as full_auto_interface:
    with gr.Row():
        with gr.Column():
            folder = folder_picker('影片輸出資料夾')
            url = gr.Textbox(
                label='影片網址 或本機路徑（可選）',
                placeholder='YouTube / Bilibili 網址，或貼上本機路徑例如 C:\\影片\\a.mp4。可留空，改用下方上傳。',
                value='',
            )
            local_file = gr.File(
                label='本地影片（可選，不必填網址）',
                file_count='single',
                file_types=['.mp4', '.mkv', '.webm', '.mov', '.avi', '.m4v', '.flv'],
            )
            source_lang = _source_language()
            target_lang = _target_language()
            with gr.Row():
                translation_model = _translation_model()
                translation_effort = _translation_effort()
            with gr.Row():
                review_model = _review_model()
                review_effort = _review_effort()

            with gr.Accordion('人聲分離', open=False):
                shifts = _shifts()
            with gr.Accordion('影片輸出', open=False):
                dl_res = _resolution('下載解析度')
                out_res = _resolution('輸出解析度')
                subtitles = gr.Checkbox(label='加入字幕', value=True)
                speed_up = gr.Slider(minimum=0.5, maximum=2, step=0.05, label='加速倍數', value=1.00)
                fps = gr.Slider(minimum=1, maximum=60, step=1, label='幀率', value=30)
            bgm, bgm_vol, video_vol = _bgm_fields()
            force_retranslate = gr.Checkbox(
                label='強制重翻',
                value=False,
                info='清掉譯文後重翻。若大綱已鎖定會沿用大綱。',
            )
            with gr.Accordion('進階', open=True):
                words_per_sec = gr.Slider(
                    minimum=1.5,
                    maximum=5.0,
                    step=0.1,
                    value=2.2,
                    label='英文語速（每秒詞）',
                    info='只影響英文譯文字數上限。越低越短、比較不會趕。Fish 英文約 2.2。',
                )
                translate_workers = gr.Slider(
                    minimum=0,
                    maximum=16,
                    step=1,
                    value=0,
                    label='翻譯併發',
                    info='0＝依金鑰自動（1 把 4、2 把 7、3 把 10）。手動填會覆寫。',
                )

            with gr.Row():
                submit = gr.Button('Submit', variant='primary')
                stop = gr.Button('中止', variant='stop')
        with gr.Column():
            status = gr.Textbox(label='合成狀態（即時進度與各步驟成本）', lines=16, max_lines=24, autoscroll=True)
            cost_md = gr.Markdown(
                label='本片 API 成本（即時估計）',
                value='處理過程會在這裡更新各步驟估計費用。長內容可在此框內捲動。',
                elem_id='live-cost-panel',
            )
            result_video = gr.Video(label='合成影片範例結果')

    def stop_running_job():
        from tools.job_control import request_stop
        request_stop()
        return '正在中止…正在收尾目前這次 API 呼叫，之後不會再進下一步。'

    run_event = submit.click(
        fn=do_everything_with_cost,
        inputs=[
            folder, url, local_file, dl_res, shifts,
            source_lang, target_lang, translation_model, translation_effort,
            review_model, review_effort,
            subtitles, speed_up, fps, out_res,
            bgm, bgm_vol, video_vol,
            words_per_sec, translate_workers, force_retranslate,
        ],
        outputs=[status, result_video, cost_md],
    )
    stop.click(
        fn=stop_running_job,
        inputs=None,
        outputs=[status],
        cancels=[run_event],
    )


# 下載影片介面
download_interface = gr.Interface(
    fn=download_from_url,
    inputs=[
        gr.Textbox(label='影片網址', placeholder='請輸入Youtube或Bilibili的影片、播放清單或頻道的URL',
                   value='https://www.bilibili.com/video/BV1kr421M7vz/'),
        folder_dropdown('影片輸出資料夾'),
    ],
    additional_inputs=[_resolution()],
    additional_inputs_accordion=_accordion('下載設定'),
    outputs=[
        gr.Textbox(label='下載狀態'),
        gr.Video(label='範例影片'),
        gr.Json(label='下載資訊')
    ],
)

# 人聲分離介面
demucs_interface = gr.Interface(
    fn=demucs_from_ui,
    inputs=[
        folder_dropdown('影片資料夾'),
    ],
    additional_inputs=[
        gr.Checkbox(label='顯示進度條', value=True),
        _shifts(),
    ],
    additional_inputs_accordion=_accordion('分離設定'),
    outputs=[
        gr.Text(label='分離結果狀態'),
        gr.Audio(label='人聲音訊'),
        gr.Audio(label='伴奏音訊')
    ],
)

# AI智慧語音識別介面
asr_inference = gr.Interface(
    fn=asr_from_ui,
    inputs=[
        folder_dropdown('影片資料夾'),
    ],
    additional_inputs=[
        _source_language(),
    ],
    additional_inputs_accordion=_accordion('識別設定'),
    outputs=[
        gr.Text(label='語音識別狀態'),
        gr.Json(label='識別結果詳情')
    ],
)

# 翻譯字幕介面
translation_interface = gr.Interface(
    fn=translate_from_ui,
    inputs=[
        folder_dropdown('影片資料夾'),
    ],
    additional_inputs=[
        _target_language(),
        _translation_model(),
        _translation_effort(),
    ],
    additional_inputs_accordion=_accordion('翻譯設定'),
    outputs=[
        gr.Text(label='翻譯狀態'),
        gr.Json(label='總結結果'),
        gr.Json(label='翻譯結果')
    ],
)

# AI語音合成介面
tts_interface = gr.Interface(
    fn=tts_from_ui,
    inputs=[
        folder_dropdown('影片資料夾'),
    ],
    additional_inputs=[
        _target_language(),
    ],
    additional_inputs_accordion=_accordion('配音設定'),
    outputs=[
        gr.Text(label='合成狀態'),
        gr.Audio(label='合成語音'),
        gr.Audio(label='原始音訊')
    ],
)

# 影片合成介面
with gr.Blocks(theme=my_theme) as synthesize_video_interface:
    with gr.Row():
        with gr.Column():
            syn_folder = folder_picker('影片資料夾')
            with gr.Accordion('影片輸出', open=False):
                syn_subtitles = gr.Checkbox(label='加入字幕', value=True)
                syn_speed = gr.Slider(minimum=0.5, maximum=2, step=0.05, label='加速倍數', value=1.00)
                syn_fps = gr.Slider(minimum=1, maximum=60, step=1, label='幀率', value=30)
                syn_res = _resolution()
            syn_bgm, syn_bgm_vol, syn_video_vol = _bgm_fields(filepath=True)
            syn_submit = gr.Button('Submit', variant='primary')
        with gr.Column():
            syn_status = gr.Text(label='合成狀態')
            syn_video = gr.Video(label='合成影片')
    syn_submit.click(
        fn=synthesize_all_video_under_folder,
        inputs=[syn_folder, syn_subtitles, syn_speed, syn_fps, syn_res, syn_bgm, syn_bgm_vol, syn_video_vol],
        outputs=[syn_status, syn_video],
    )

def _folder_update(current, resolved):
    import os
    folders = list_video_folders()
    resolved = os.path.normpath(resolved or '')
    current = os.path.normpath(current or '')
    if resolved and resolved not in folders:
        folders.append(resolved)
    if current == resolved and resolved in folders:
        return gr.update()
    value = resolved if resolved in folders else (current if current in folders else (folders[0] if folders else 'videos'))
    return gr.update(choices=folders, value=value)


def _reload_speaker_keep_status(folder, language, prev_status):
    loaded = list(_load_speaker_ui(folder, language))
    prev = (prev_status or '').strip()
    if prev:
        loaded[5] = f'{prev}\n{loaded[5]}'
    return tuple(loaded)


def _load_speaker_ui(folder, language):
    try:
        resolved, rows, status = load_speaker_table(folder, language)
        title, plot, outline, glossary, voices, bible_status = load_bible_fields(resolved)
        return (
            _folder_update(folder, resolved),
            rows,
            '',
            review_table_rows(resolved, language),
            _review_pick_update(resolved, language),
            f'{status}\n{bible_status}\n{review_status_line(resolved, language)}',
            title, plot, outline, glossary, voices,
        )
    except Exception as exc:
        title, plot, outline, glossary, voices, bible_status = ('', '', '', '', '', '')
        try:
            title, plot, outline, glossary, voices, bible_status = load_bible_fields(folder)
        except Exception:
            pass
        return (
            refresh_folder_dropdown(folder), [], '', [], gr.update(choices=[], value=[]),
            f'{exc}\n{bible_status}', title, plot, outline, glossary, voices,
        )


def _sync_folder_language(folder):
    from tools.target_language import load_dub_meta
    from tools.translation_versions import snapshot_active

    try:
        resolved = episode_folder(folder)
        snapshot_active(resolved)
        return gr.update(value=ui_language_label(load_dub_meta(resolved).get('translation') or 'English'))
    except Exception:
        return gr.update()


def _folder_picked(folder, language):
    from tools.target_language import load_dub_meta
    from tools.translation_versions import snapshot_active

    try:
        resolved = episode_folder(folder)
        snapshot_active(resolved)
        language = ui_language_label(load_dub_meta(resolved).get('translation') or language or 'English')
    except Exception:
        resolved = folder
    loaded = _load_speaker_ui(resolved, language)
    return (loaded[0], language, *loaded[1:])


def _review_speaker_ui(folder, language, review_model, review_effort):
    import queue
    import threading

    updates = queue.Queue()
    holder = {'resolved': folder, 'lang': language, 'error': None}

    def on_progress(message):
        updates.put(('progress', message))

    def worker():
        try:
            apply_model_choices(review_model=review_model, review_effort=review_effort)
            resolved = episode_folder(folder)
            trans_lang, _tts_lang = split_target_language(language)
            holder['resolved'] = resolved
            holder['lang'] = trans_lang
            review_folder(
                resolved, trans_lang, method=TRANSLATE_METHOD, progress_callback=on_progress,
            )
            updates.put(('done', None))
        except Exception as exc:
            holder['error'] = exc
            updates.put(('error', str(exc)))

    threading.Thread(target=worker, daemon=True).start()
    log = ['審稿開始，請不要連點。']
    yield folder, [], gr.update(), log[0]
    while True:
        kind, payload = updates.get()
        if kind == 'progress':
            log.append(payload)
            yield folder, [], gr.update(), '\n'.join(log[-16:])
            continue
        if kind == 'error':
            yield folder, [], gr.update(), f'審稿失敗：{payload}\n' + '\n'.join(log[-12:])
            return
        resolved = holder['resolved']
        trans_lang = holder['lang']
        yield (
            gr.update(value=resolved),
            review_table_rows(resolved, trans_lang),
            _review_pick_update(resolved, trans_lang),
            review_status_line(resolved, trans_lang) + '\n' + '\n'.join(log[-12:]),
        )
        return


def _apply_review_ui(folder, language, selected, review_table):
    try:
        resolved, _applied, _skipped, status, kept = apply_selected_review(
            folder, language, selected, review_table,
        )
        _folder, rows, table_status = load_speaker_table(resolved, language)
        return (
            gr.update(value=resolved),
            rows,
            '',
            review_table_rows(resolved, language),
            _review_pick_update(resolved, language, selected=kept),
            f'{status}\n{table_status}\n{review_status_line(resolved, language)}',
        )
    except Exception as exc:
        return folder, [], '', [], gr.update(), f'套用審稿失敗：{exc}'


def _review_select_suggested(folder, language):
    return gr.update(value=review_suggested_values(folder, language))


def _review_select_none():
    return gr.update(value=[])


def _save_speaker_ui(folder, language, table):
    try:
        resolved, status, *_ = save_speaker_table(folder, table, language)
        return gr.update(value=resolved), status
    except Exception as exc:
        return folder, str(exc)


def _pick_speaker_row(table, evt: gr.SelectData):
    if evt is None or evt.index is None:
        return gr.update()
    index = evt.index
    row_i = index[0] if isinstance(index, (list, tuple)) else index
    rows = table if isinstance(table, list) else []
    picked = None
    if isinstance(row_i, int) and 0 <= row_i < len(rows):
        fields = _speaker_row_fields(rows[row_i])
        if fields:
            try:
                picked = int(float(fields['index']))
            except (TypeError, ValueError):
                picked = row_i
    if picked is None and isinstance(row_i, int) and row_i >= 0:
        picked = row_i
    if picked is None:
        return gr.update()
    return str(picked)


def _retranslate_checked_ui(folder, language, selected, translation_model, translation_effort):
    picked = selected_line_indices(selected)
    if not picked:
        raise gr.Error('請先在對白表點選或打勾要重翻的句子，再按「重翻勾選句」。')
    try:
        apply_model_choices(translation_model=translation_model, translation_effort=translation_effort)
        resolved, status, _ok = retranslate_checked_lines(
            folder, language, table=None, selected=picked, method=TRANSLATE_METHOD,
        )
        if not _ok and '請先' in (status or ''):
            raise gr.Error(status)
        loaded = list(_load_speaker_ui(resolved, language))
        loaded[0] = _folder_update(folder, resolved)
        loaded[5] = f'{status}\n{loaded[5]}'
        return tuple(loaded)
    except gr.Error:
        raise
    except Exception as exc:
        raise gr.Error(f'重翻勾選句失敗：{exc}')


def _retranslate_one_ui(folder, language, payload, translation_model, translation_effort):
    index, source = parse_one_line_payload(payload)
    if index is None:
        raise gr.Error('請按該列右邊的「單句翻」。')
    if not source:
        raise gr.Error('這一句沒有中文原文，請先改「原文」再按單句翻。')
    try:
        apply_model_choices(translation_model=translation_model, translation_effort=translation_effort)
        resolved, status, _ok = retranslate_one_line(
            folder, language, payload, method=TRANSLATE_METHOD,
        )
        if not _ok:
            raise gr.Error(status)
        loaded = list(_load_speaker_ui(resolved, language))
        loaded[0] = _folder_update(folder, resolved)
        loaded[5] = f'{status}\n{loaded[5]}'
        return tuple(loaded)
    except gr.Error:
        raise
    except Exception as exc:
        raise gr.Error(f'單句翻失敗：{exc}')


def _redub_speaker_ui(folder, language, table):
    try:
        status, audio, video = redub_speakers(folder, language, table)
        return status, audio, video
    except Exception as exc:
        return str(exc), None, None


def _save_bible_ui(folder, title, plot, outline, glossary, voices):
    try:
        resolved, status = save_bible_fields(folder, title, plot, outline, glossary, voices)
        return gr.update(value=resolved), status
    except Exception as exc:
        return folder, str(exc)


def _retranslate_bible_ui(folder, language, translation_model, translation_effort, title, plot, outline, glossary, voices):
    try:
        apply_model_choices(translation_model=translation_model, translation_effort=translation_effort)
        status, audio, video = retranslate_from_bible(
            folder, language, title, plot, outline, glossary, voices,
        )
        resolved, rows, table_status = load_speaker_table(folder, language)
        return (
            _folder_update(folder, resolved),
            rows,
            '',
            review_table_rows(resolved, language),
            _review_pick_update(resolved, language),
            f'{status}\n{table_status}\n{review_status_line(resolved, language)}',
            audio, video,
        )
    except Exception as exc:
        return folder, [], '', [], gr.update(), str(exc), None, None


with gr.Blocks(theme=my_theme, css=_UI_CSS, js=_UI_JS) as speaker_interface:
    gr.Markdown(
        '同一集可同時留英文、日文、越南文等多個語言版本，互不覆蓋。'
        '先選資料夾與語言版本再載入。可改「講者」、「原文」與「譯文」，也可改大綱後依大綱重翻。'
        '改完某一列的「原文」後，按該列右邊「單句翻」，只依那句中文重翻譯文，不自動配音。'
        '要一次處理多句，在最左欄打勾再按「重翻勾選句」；勾連續多句會把原文與時間併到第一句。'
        '不必重跑下載、分軌、辨識。儲存講者後會鎖住自動音高拆人。'
    )
    sp_folder = folder_picker('集數資料夾（從清單選，不能自己打路徑）')
    sp_lang = _target_language(
        '語言版本',
        info='載入、儲存、審稿、重配都用這個語言。已翻譯的語言不會被另一種蓋掉。',
    )
    with gr.Row():
        sp_model = _translation_model()
        sp_effort = _translation_effort()
    sp_load = gr.Button('載入對白與大綱')
    sp_table = gr.Dataframe(
        headers=['勾', '句', '開始秒', '講者', '原文', '譯文', '單句翻'],
        datatype=['str', 'number', 'number', 'str', 'str', 'str', 'str'],
        col_count=(7, 'fixed'),
        type='array',
        wrap=True,
        interactive=True,
        label='對白（改「原文」後按該列右邊「單句翻」；最左欄打勾可一次重翻多句）',
        column_widths=['44px', '56px', '80px', '120px', '28%', '28%', '72px'],
        elem_id='speaker-table',
    )
    sp_one_payload = gr.Textbox(value='', visible=False, elem_id='line-one-payload')
    sp_retranslate_one = gr.Button('單句翻此列', elem_id='line-one-trans')
    sp_line_ids = gr.Textbox(
        value='',
        label='已勾選的句號',
        info='打勾或點選那一列後，句號會出現在這裡。也可直接輸入例如 4 或 84,85。',
        lines=1,
        max_lines=1,
        elem_id='line-pick-ids',
    )
    with gr.Row():
        sp_save = gr.Button('儲存講者／譯文')
        sp_retranslate_sel = gr.Button('重翻勾選句')
        sp_redub = gr.Button('只重配音', variant='primary')
    with gr.Accordion('AI 審稿', open=True):
        gr.Markdown(
            '聲音轉成文字後，會先讓 AI 寫這一集的中文大綱，再依大綱修辨識台詞（錯字、斷句、講者、略過聽不清的殘句），然後才翻譯。'
            '譯文審稿只查譯文（人名拼法、漏譯、數字），不再改講者、也不再併中文卡。'
            '低嚴重度語氣詞不自動套用。沒有建議譯文的只留提醒。'
            '這裡按「重新審稿」仍只標不改，看過再按「套用審稿內容」。斷句條目標的是要保留的那一句，套用後會刪掉被併進去的卡片。'
            '也可在建議譯文表改字再套用。套用不會自動重配，也不會刪成片。'
            '一鍵配音／只重配音會在 Fish 之後再跑一輪對真實音檔的審稿（過長、對畫面延後），過關的建議會自動改短並重配。'
            '「重新審稿」會打 API，可能一分鐘到四分鐘；未完成前請不要連點。'
        )
        with gr.Row():
            sp_review_model = _review_model()
            sp_review_effort = _review_effort()
        sp_review_btn = gr.Button('重新審稿')
        sp_review_pick = gr.CheckboxGroup(
            choices=[],
            label='要套用的項目（可打勾複選）',
            info='每條顯示：句號｜秒數｜建議譯文。沒有建議譯文的句子不會出現在這裡。',
            elem_id='review-pick',
        )
        with gr.Row():
            sp_review_all = gr.Button('全選有建議的', scale=0, min_width=140)
            sp_review_none = gr.Button('全不選', scale=0, min_width=100)
        with gr.Row(elem_id='review-suggest-row'):
            with gr.Column(scale=5):
                sp_review = gr.Dataframe(
                    headers=['句', '開始秒', '建議譯文', '嚴重度', '問題'],
                    datatype=['number', 'number', 'str', 'str', 'str'],
                    col_count=(5, 'fixed'),
                    type='array',
                    wrap=True,
                    interactive=True,
                    label='建議譯文',
                    height=360,
                    elem_id='review-suggest-table',
                    column_widths=['56px', '80px', '36%', '70px', '28%'],
                )
            with gr.Column(scale=0, min_width=160, elem_id='review-apply-col'):
                sp_apply_review = gr.Button('套用審稿內容', variant='primary')
    with gr.Accordion('配音大綱（人名／劇情／語氣）', open=True):
        sp_title = gr.Textbox(label='標題', lines=1)
        sp_plot = gr.Textbox(label='劇情摘要', lines=3)
        sp_outline = gr.Textbox(label='大綱（按對白順序）', lines=8)
        sp_glossary = gr.Textbox(
            label='人名對照（中文=譯名，勿放篝火／金幣這類普通詞）',
            lines=4,
        )
        sp_voices = gr.Textbox(label='聲線／旁白／系統提示', lines=3)
        with gr.Row():
            sp_save_bible = gr.Button('儲存大綱')
            sp_retranslate = gr.Button('依大綱重翻並重配', variant='primary')
    sp_status = gr.Textbox(
        label='狀態',
        lines=8,
        max_lines=8,
        autoscroll=True,
        elem_id='review-status',
    )
    sp_audio = gr.Audio(label='重配後的人聲+伴奏')
    sp_video = gr.Video(label='重配後的影片')
    sp_bible_outs = [sp_title, sp_plot, sp_outline, sp_glossary, sp_voices]
    _load_outs = [sp_folder, sp_table, sp_line_ids, sp_review, sp_review_pick, sp_status, *sp_bible_outs]
    sp_load.click(_load_speaker_ui, inputs=[sp_folder, sp_lang], outputs=_load_outs)
    sp_folder.change(
        _folder_picked,
        inputs=[sp_folder, sp_lang],
        outputs=[sp_folder, sp_lang, sp_table, sp_line_ids, sp_review, sp_review_pick, sp_status, *sp_bible_outs],
    )
    sp_lang.change(_load_speaker_ui, inputs=[sp_folder, sp_lang], outputs=_load_outs)
    sp_review_btn.click(
        _review_speaker_ui,
        inputs=[sp_folder, sp_lang, sp_review_model, sp_review_effort],
        outputs=[sp_folder, sp_review, sp_review_pick, sp_status],
    )
    sp_apply_review.click(
        _apply_review_ui,
        inputs=[sp_folder, sp_lang, sp_review_pick, sp_review],
        outputs=[sp_folder, sp_table, sp_line_ids, sp_review, sp_review_pick, sp_status],
    ).success(
        _reload_speaker_keep_status,
        inputs=[sp_folder, sp_lang, sp_status],
        outputs=_load_outs,
    )
    sp_review_all.click(
        _review_select_suggested,
        inputs=[sp_folder, sp_lang],
        outputs=[sp_review_pick],
    )
    sp_review_none.click(_review_select_none, outputs=[sp_review_pick])
    sp_save.click(
        _save_speaker_ui,
        inputs=[sp_folder, sp_lang, sp_table],
        outputs=[sp_folder, sp_status],
    )
    sp_table.select(
        _pick_speaker_row,
        inputs=[sp_table],
        outputs=[sp_line_ids],
    )
    sp_retranslate_sel.click(
        _retranslate_checked_ui,
        inputs=[sp_folder, sp_lang, sp_line_ids, sp_model, sp_effort],
        outputs=_load_outs,
        js=_RETRANSLATE_JS,
        show_progress='full',
        trigger_mode='multiple',
    )
    sp_retranslate_one.click(
        _retranslate_one_ui,
        inputs=[sp_folder, sp_lang, sp_one_payload, sp_model, sp_effort],
        outputs=_load_outs,
        js=_ONE_JS,
        show_progress='full',
        trigger_mode='multiple',
    )
    sp_redub.click(
        _redub_speaker_ui,
        inputs=[sp_folder, sp_lang, sp_table],
        outputs=[sp_status, sp_audio, sp_video],
    ).success(
        _reload_speaker_keep_status,
        inputs=[sp_folder, sp_lang, sp_status],
        outputs=_load_outs,
    )
    sp_save_bible.click(
        _save_bible_ui,
        inputs=[sp_folder, *sp_bible_outs],
        outputs=[sp_folder, sp_status],
    )
    sp_retranslate.click(
        _retranslate_bible_ui,
        inputs=[sp_folder, sp_lang, sp_model, sp_effort, *sp_bible_outs],
        outputs=[sp_folder, sp_table, sp_line_ids, sp_review, sp_review_pick, sp_status, sp_audio, sp_video],
    ).success(
        _reload_speaker_keep_status,
        inputs=[sp_folder, sp_lang, sp_status],
        outputs=_load_outs,
    )

with gr.Blocks(theme=my_theme) as step_menu:
    gr.Markdown('平時用「一鍵自動化」。需要單獨重跑某一步、或改講者後只重配時再展開。')
    with gr.Accordion('改講者／譯文／大綱', open=True):
        speaker_interface.render()
    for name, iface in (
        ('自動下載影片', download_interface),
        ('人聲分離', demucs_interface),
        ('AI智慧語音識別', asr_inference),
        ('字幕翻譯', translation_interface),
        ('AI語音合成', tts_interface),
        ('影片合成', synthesize_video_interface),
    ):
        with gr.Accordion(name, open=False):
            iface.render()

app = gr.TabbedInterface(
    theme=my_theme,
    css=_UI_CSS,
    js=_UI_JS,
    interface_list=[
        api_settings_interface,
        full_auto_interface,
        step_menu,
    ],
    tab_names=[
        'API 設定',
        '一鍵自動化 One-Click',
        '分步操作',
    ],
    title='Lovesnow-translate 影片AI配音／翻譯'
)

if __name__ == '__main__':
    import os
    os.environ['NO_PROXY'] = '127.0.0.1,localhost,::1'
    os.environ['no_proxy'] = '127.0.0.1,localhost,::1'
    try:
        import gradio.networking as _net
        _net.url_ok = lambda url: True
    except Exception:
        pass
    app.queue(default_concurrency_limit=4)
    _assets = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets')
    app.launch(
        server_name="127.0.0.1",
        server_port=6006,
        share=False,
        inbrowser=False,
        show_error=True,
        show_api=False,
        allowed_paths=[_assets],
    )