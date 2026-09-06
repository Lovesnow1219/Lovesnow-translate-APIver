# -*- coding: utf-8 -*-
import os

import gradio as gr

_SKIP_FOLDER_NAMES = {'_demucs_chunks', 'wavs', 'SPEAKER', '__pycache__', 'translations'}


def list_video_folders():
    root = 'videos'
    os.makedirs(root, exist_ok=True)
    folders = [root]
    for dirpath, dirnames, _ in os.walk(root):
        dirnames[:] = sorted(
            name for name in dirnames
            if name not in _SKIP_FOLDER_NAMES and not name.startswith('.')
        )
        for name in dirnames:
            folders.append(os.path.normpath(os.path.join(dirpath, name)))
    return folders


def refresh_folder_dropdown(current):
    folders = list_video_folders()
    current = os.path.normpath(current) if current else ''
    value = current if current in folders else ('videos' if 'videos' in folders else (folders[0] if folders else 'videos'))
    if value not in folders:
        value = folders[0] if folders else 'videos'
    return gr.update(choices=folders, value=value)


def folder_dropdown(label='影片輸出資料夾'):
    folders = list_video_folders()
    value = 'videos' if 'videos' in folders else (folders[0] if folders else 'videos')
    return gr.Dropdown(
        choices=folders,
        value=value,
        label=label,
        allow_custom_value=False,
        filterable=False,
    )


def folder_picker(label='影片輸出資料夾'):
    with gr.Row():
        dropdown = folder_dropdown(label)
        refresh = gr.Button('重新整理', scale=0, min_width=110)
    refresh.click(refresh_folder_dropdown, inputs=dropdown, outputs=dropdown)
    return dropdown
