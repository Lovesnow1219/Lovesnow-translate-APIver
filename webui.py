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

from tools.step000_video_downloader import download_from_url
from tools.step010_demucs_vr import separate_all_audio_under_folder
from tools.step020_asr import transcribe_all_audio_under_folder
from tools.step030_translation import translate_all_transcript_under_folder
from tools.step040_tts import generate_all_wavs_under_folder
from tools.step050_synthesize_video import synthesize_all_video_under_folder
from tools.do_everything import stream_do_everything
from tools.api_settings_ui import build_api_settings_ui
from tools.speaker_edit import (
    episode_folder,
    load_bible_fields,
    load_speaker_table,
    redub_speakers,
    retranslate_from_bible,
    save_bible_fields,
    save_speaker_table,
)
from tools.target_language import TARGET_LANGUAGES, split_target_language
from tools.translation_review import review_folder, review_status_line, review_table_rows
from tools.video_folders import folder_dropdown, folder_picker
from dotenv import load_dotenv
load_dotenv(override=False)

api_settings_interface = build_api_settings_ui()

_RESOLUTIONS = ['4320p', '2160p', '1440p', '1080p', '720p', '480p', '360p', '240p', '144p']
DEMUCS_METHOD = 'Replicate'
ASR_METHOD = 'OpenAI'
ASR_MODEL = 'gpt-4o-transcribe-diarize'
TRANSLATE_METHOD = 'OpenAI'
TTS_METHOD = 'Fish'


def _resolution(label='解析度'):
    return gr.Dropdown(choices=_RESOLUTIONS, value='1080p', label=label)


def _accordion(title):
    return gr.Accordion(title, open=False)


def _shifts():
    return gr.Slider(
        minimum=0,
        maximum=10,
        step=1,
        value=1,
        label='移位次數 Number of shifts',
        info='越高品質越好，但時間與費用幾乎倍增，通常不划算。預設 1 即可。',
    )


def _target_language(label='目標語言'):
    return gr.Dropdown(
        choices=TARGET_LANGUAGES,
        value='English',
        label=label,
        info='完整支援：英文、日文、越南文、中文／粵語。標「部分支援」的語言可翻可配，但語氣詞與漏句規則尚未對齊。',
        allow_custom_value=False,
        filterable=False,
    )


def do_everything_with_cost(
    root_folder, url, local_file, dl_res, shifts, target_lang,
    subtitles, speed_up, fps, out_res, bgm, bgm_vol, video_vol,
    words_per_sec, translate_workers, force_retranslate, force_redub,
):
    for status, video, cost_md in stream_do_everything(
        root_folder, url,
        dl_res, shifts, target_lang, subtitles, speed_up, fps, out_res,
        1, 3, bgm, bgm_vol, video_vol,
        local_file=local_file,
        words_per_sec=words_per_sec,
        translate_workers=translate_workers,
        force_retranslate=force_retranslate,
        force_redub=force_redub,
    ):
        yield status, (gr.update() if video is None else video), cost_md


def demucs_from_ui(folder, progress, shifts):
    return separate_all_audio_under_folder(folder, model_name=DEMUCS_METHOD, progress=progress, shifts=shifts)


def asr_from_ui(folder):
    return transcribe_all_audio_under_folder(folder, ASR_METHOD, ASR_MODEL)


def translate_from_ui(folder, language):
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


_UI_CSS = """
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
"""

# 一鍵自動化介面
with gr.Blocks(css=_UI_CSS) as full_auto_interface:
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
            target_lang = _target_language()

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
            force_redub = gr.Checkbox(
                label='強制重配',
                value=False,
                info='保留譯文，清掉配音後重配。強制重翻時不必勾。',
            )
            with gr.Accordion('進階', open=True):
                words_per_sec = gr.Slider(
                    minimum=1.5,
                    maximum=5.0,
                    step=0.1,
                    value=3.0,
                    label='英文語速（每秒詞）',
                    info='只影響英文譯文字數上限。越高句子越長。',
                )
                translate_workers = gr.Slider(
                    minimum=1,
                    maximum=32,
                    step=1,
                    value=8,
                    label='翻譯併發',
                    info='同時翻幾句。有兩把 OpenAI key 時可再高一點。',
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
        return '正在中止…目前這一步結束後就會停。'

    run_event = submit.click(
        fn=do_everything_with_cost,
        inputs=[
            folder, url, local_file, dl_res, shifts,
            target_lang, subtitles, speed_up, fps, out_res,
            bgm, bgm_vol, video_vol,
            words_per_sec, translate_workers, force_retranslate, force_redub,
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
with gr.Blocks() as synthesize_video_interface:
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

def _load_speaker_ui(folder):
    try:
        resolved, rows, status = load_speaker_table(folder)
        title, plot, outline, glossary, voices, bible_status = load_bible_fields(resolved)
        review_rows = review_table_rows(resolved)
        return (
            gr.update(value=resolved), rows, review_rows,
            f'{status}\n{bible_status}\n{review_status_line(resolved)}',
            title, plot, outline, glossary, voices,
        )
    except Exception as exc:
        title, plot, outline, glossary, voices, bible_status = ('', '', '', '', '', '')
        try:
            title, plot, outline, glossary, voices, bible_status = load_bible_fields(folder)
        except Exception:
            pass
        return folder, [], [], f'{exc}\n{bible_status}', title, plot, outline, glossary, voices


def _review_speaker_ui(folder, language):
    try:
        resolved = episode_folder(folder)
        trans_lang, _tts_lang = split_target_language(language)
        review_folder(resolved, trans_lang, method=TRANSLATE_METHOD)
        return (
            gr.update(value=resolved),
            review_table_rows(resolved),
            review_status_line(resolved),
        )
    except Exception as exc:
        return folder, [], f'審稿失敗：{exc}'


def _save_speaker_ui(folder, table):
    try:
        resolved, status, *_ = save_speaker_table(folder, table)
        return gr.update(value=resolved), status
    except Exception as exc:
        return folder, str(exc)


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


def _retranslate_bible_ui(folder, language, title, plot, outline, glossary, voices):
    try:
        status, audio, video = retranslate_from_bible(
            folder, language, title, plot, outline, glossary, voices,
        )
        resolved, rows, table_status = load_speaker_table(folder)
        return (
            gr.update(value=resolved), rows, review_table_rows(resolved),
            f'{status}\n{table_status}\n{review_status_line(resolved)}',
            audio, video,
        )
    except Exception as exc:
        return folder, [], [], str(exc), None, None


with gr.Blocks() as speaker_interface:
    gr.Markdown(
        '可改「講者」與「譯文」，也可改大綱後依大綱重翻。'
        '不必重跑下載、分軌、辨識。儲存講者後會鎖住自動音高拆人。'
    )
    sp_folder = folder_picker('集數資料夾（選到有 translation.json 的那層）')
    sp_load = gr.Button('載入對白與大綱')
    sp_table = gr.Dataframe(
        headers=['句', '開始秒', '講者', '原文', '譯文'],
        datatype=['number', 'number', 'str', 'str', 'str'],
        col_count=(5, 'fixed'),
        type='array',
        wrap=True,
        interactive=True,
        label='對白（可改「講者」與「譯文」）',
        column_widths=['56px', '80px', '140px', '32%', '32%'],
    )
    with gr.Row():
        sp_save = gr.Button('儲存講者／譯文')
        sp_redub = gr.Button('只重配音', variant='primary')
    sp_lang = _target_language()
    with gr.Accordion('AI 審稿（只標不改）', open=True):
        gr.Markdown('翻譯結束會自動跑。這裡可再跑一次。只寫報告，不會改「譯文」欄。')
        sp_review_btn = gr.Button('重新審稿')
        sp_review = gr.Dataframe(
            headers=['句', '開始秒', '嚴重度', '問題', '建議譯文'],
            datatype=['number', 'number', 'str', 'str', 'str'],
            col_count=(5, 'fixed'),
            type='array',
            wrap=True,
            interactive=False,
            label='審稿結果',
            column_widths=['56px', '80px', '80px', '40%', '40%'],
        )
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
    sp_status = gr.Textbox(label='狀態', lines=4)
    sp_audio = gr.Audio(label='重配後的人聲+伴奏')
    sp_video = gr.Video(label='重配後的影片')
    sp_bible_outs = [sp_title, sp_plot, sp_outline, sp_glossary, sp_voices]
    sp_load.click(
        _load_speaker_ui,
        inputs=[sp_folder],
        outputs=[sp_folder, sp_table, sp_review, sp_status, *sp_bible_outs],
    )
    sp_review_btn.click(
        _review_speaker_ui,
        inputs=[sp_folder, sp_lang],
        outputs=[sp_folder, sp_review, sp_status],
    )
    sp_save.click(_save_speaker_ui, inputs=[sp_folder, sp_table], outputs=[sp_folder, sp_status])
    sp_redub.click(
        _redub_speaker_ui,
        inputs=[sp_folder, sp_lang, sp_table],
        outputs=[sp_status, sp_audio, sp_video],
    )
    sp_save_bible.click(
        _save_bible_ui,
        inputs=[sp_folder, *sp_bible_outs],
        outputs=[sp_folder, sp_status],
    )
    sp_retranslate.click(
        _retranslate_bible_ui,
        inputs=[sp_folder, sp_lang, *sp_bible_outs],
        outputs=[sp_folder, sp_table, sp_review, sp_status, sp_audio, sp_video],
    )

my_theme = gr.themes.Soft()

with gr.Blocks() as step_menu:
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
    title='影片AI配音/翻譯工具 API版-lovesnow 開源自Linly-Dubbing'
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
    app.launch(
        server_name="127.0.0.1",
        server_port=6006,
        share=False,
        inbrowser=False,
        show_error=True,
        show_api=False,
    )