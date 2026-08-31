import os
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
                               QComboBox, QPushButton, QMessageBox, QGroupBox)

# 尝试导入实际的功能模块
try:
    from tools.step040_tts import generate_all_wavs_under_folder
    from tools.utils import SUPPORT_VOICE
except ImportError:
    # 定义临时的支持语音列表
    SUPPORT_VOICE = ['zh-CN-XiaoxiaoNeural', 'zh-CN-YunxiNeural',
                     'en-US-JennyNeural', 'ja-JP-NanamiNeural']


class TTSTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)

        # 视频文件夹
        self.video_folder = QLineEdit("videos")
        self.layout.addWidget(QLabel("视频文件夹"))
        self.layout.addWidget(self.video_folder)

        # AI语音生成方法
        self.tts_method = QComboBox()
        self.tts_method.addItems(['xtts', 'cosyvoice', 'EdgeTTS'])
        self.layout.addWidget(QLabel("AI语音生成方法"))
        self.layout.addWidget(self.tts_method)

        # 目标语言
        self.target_language = QComboBox()
        self.target_language.setEditable(False)
        self.target_language.addItems(['中文', 'English', '粤语', 'Japanese', 'Korean', 'Spanish', 'French',
                                       'Vietnamese', 'Thai', 'Indonesian', 'Malay', 'Filipino'])
        self.target_language.setCurrentText('中文')
        self.layout.addWidget(QLabel("目标语言"))
        self.layout.addWidget(self.target_language)

        # EdgeTTS声音选择
        self.edge_tts_voice = QComboBox()
        self.edge_tts_voice.addItems(SUPPORT_VOICE)
        self.edge_tts_voice.setCurrentText('zh-CN-XiaoxiaoNeural')
        self.layout.addWidget(QLabel("EdgeTTS声音选择"))
        self.layout.addWidget(self.edge_tts_voice)

        # 执行按钮
        self.run_button = QPushButton("开始生成语音")
        self.run_button.clicked.connect(self.run_tts)
        self.layout.addWidget(self.run_button)

        # 状态显示
        self.status_label = QLabel("准备就绪")
        self.layout.addWidget(QLabel("合成状态:"))
        self.layout.addWidget(self.status_label)

        # 音频播放控件
        synthesized_group = QGroupBox("合成语音")
        synthesized_layout = QVBoxLayout()
        self.synthesized_play_button = QPushButton("播放合成语音")
        synthesized_layout.addWidget(self.synthesized_play_button)
        synthesized_group.setLayout(synthesized_layout)

        original_group = QGroupBox("原始音频")
        original_layout = QVBoxLayout()
        self.original_play_button = QPushButton("播放原始音频")
        original_layout.addWidget(self.original_play_button)
        original_group.setLayout(original_layout)

        audio_layout = QHBoxLayout()
        audio_layout.addWidget(synthesized_group)
        audio_layout.addWidget(original_group)

        self.layout.addLayout(audio_layout)
        self.setLayout(self.layout)

    def run_tts(self):
        # 这里应该调用原始的generate_all_wavs_under_folder函数
        # 临时实现，实际应用中需要替换为真实的调用
        self.status_label.setText("生成中...")
        QMessageBox.information(self, "功能提示", "AI语音合成功能正在实现中...")

        # 实际应用中解除以下注释

        try:
            status, synthesized_path, original_path = generate_all_wavs_under_folder(
                self.video_folder.text(),
                self.tts_method.currentText(),
                self.target_language.currentText(),
                self.edge_tts_voice.currentText()
            )
            self.status_label.setText(status)
            if synthesized_path and os.path.exists(synthesized_path):
                self.synthesized_play_button.setEnabled(True)
            if original_path and os.path.exists(original_path):
                self.original_play_button.setEnabled(True)
        except Exception as e:
            self.status_label.setText(f"生成失败: {str(e)}")
