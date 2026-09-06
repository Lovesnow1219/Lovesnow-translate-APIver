# -*- coding: utf-8 -*-
import gradio as gr

from tools.api_settings import (
    ALL_KEYS,
    EXTRA_FIELDS,
    MODEL_DEFAULTS,
    MODEL_FIELD_CHOICES,
    MODEL_FIELDS,
    REQUIRED_FIELDS,
    extra_status_markdown,
    load_from_ui,
    model_status_markdown,
    save_from_ui,
    status_markdown,
)
from tools.cost_tracker import costs_under_folder
from tools.video_folders import folder_picker


def _show_folder_costs(folder):
    return costs_under_folder(folder or 'videos')


def _key_boxes(fields):
    boxes = []
    for _key, label, secret, placeholder, info in fields:
        boxes.append(gr.Textbox(
            label=label,
            type='password' if secret else 'text',
            placeholder=placeholder,
            info=info or None,
            max_lines=1,
        ))
    return boxes


def _model_boxes(fields):
    boxes = []
    for key, label, _secret, _placeholder, info in fields:
        boxes.append(gr.Dropdown(
            choices=MODEL_FIELD_CHOICES[key],
            value=MODEL_DEFAULTS[key],
            label=label,
            info=info,
            allow_custom_value=False,
            filterable=False,
        ))
    return boxes


def build_api_settings_ui(theme=None):
    with gr.Blocks(theme=theme) as demo:
        gr.Markdown(
            '## API 設定\n'
            '填入金鑰後按儲存，寫入專案根目錄的 `.env`（已加入 `.gitignore`，不會被提交）。\n'
            '重整頁面會從 `.env` 載回已存的 key；密碼框留空再存不會刪除。\n'
            '**中翻英 API 模式**左邊三把必要金鑰即可跑；右邊是可選的額外加速。'
        )
        boxes = []
        with gr.Row():
            with gr.Column():
                gr.Markdown('### 必要金鑰')
                status = gr.Markdown(value=status_markdown())
                boxes.extend(_key_boxes(REQUIRED_FIELDS))
            with gr.Column():
                gr.Markdown('### 額外加速（可選）')
                extra_status = gr.Markdown(value=extra_status_markdown())
                boxes.extend(_key_boxes(EXTRA_FIELDS))
        gr.Markdown('### 模型（翻譯／審稿）')
        model_status = gr.Markdown(value=model_status_markdown())
        with gr.Row():
            boxes.extend(_model_boxes(MODEL_FIELDS))
        result = gr.Textbox(label='儲存結果', interactive=False)
        with gr.Row():
            save_btn = gr.Button('儲存 API 設定', variant='primary')
            refresh_btn = gr.Button('重新整理狀態')
        save_btn.click(
            save_from_ui, inputs=boxes,
            outputs=[result, status, extra_status, model_status, *boxes],
        )
        refresh_btn.click(load_from_ui, outputs=[status, extra_status, model_status, *boxes])
        demo.load(load_from_ui, outputs=[status, extra_status, model_status, *boxes])
        gr.Markdown('## 影片 API 成本')
        cost_folder = folder_picker('要查看的資料夾')
        cost_md = gr.Markdown(value='跑完一鍵配音後，影片資料夾會寫入 `cost.json`。')
        cost_btn = gr.Button('查看資料夾成本')
        cost_btn.click(_show_folder_costs, inputs=cost_folder, outputs=cost_md)
        if len(boxes) != len(ALL_KEYS):
            raise RuntimeError('API 設定欄位與 ALL_KEYS 數量不一致')
    return demo
