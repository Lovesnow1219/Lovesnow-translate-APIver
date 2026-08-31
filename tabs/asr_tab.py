import os
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QLineEdit,
                               QScrollArea, QComboBox, QCheckBox, QPushButton, QMessageBox)

from ui_components import RadioButtonGroup

# 尝试导入实际的功能模块
try:
    from tools.step020_asr import transcribe_all_audio_under_folder
except ImportError:
    pass


class ASRTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)

        # 创建一个滚动区域
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_widget = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_widget)

        # 视频文件夹
        self.video_folder = QLineEdit("videos")
        self.scroll_layout.addWidget(QLabel("视频文件夹"))
        self.scroll_layout.addWidget(self.video_folder)

        # ASR模型选择
        self.asr_model = QComboBox()
        self.asr_model.addItems(['WhisperX', 'FunASR'])
        self.scroll_layout.addWidget(QLabel("ASR模型选择"))
        self.scroll_layout.addWidget(self.asr_model)

        # WhisperX模型大小
        self.whisperx_size = RadioButtonGroup(['large', 'medium', 'small', 'base', 'tiny'], "WhisperX模型大小", 'large')
        self.scroll_layout.addWidget(self.whisperx_size)

        # 分离多个说话人
        self.separate_speakers = QCheckBox("分离多个说话人")
        self.separate_speakers.setChecked(True)
        self.scroll_layout.addWidget(self.separate_speakers)

        # 执行按钮
        self.run_button = QPushButton("开始识别")
        self.run_button.clicked.connect(self.run_asr)
        self.scroll_layout.addWidget(self.run_button)

        # 状态显示
        self.status_label = QLabel("准备就绪")
        self.scroll_layout.addWidget(QLabel("语音识别状态:"))
        self.scroll_layout.addWidget(self.status_label)

        # 识别结果详情
        self.result_detail = QLabel("识别结果将显示在这里")
        self.scroll_layout.addWidget(QLabel("识别结果详情:"))
        self.scroll_layout.addWidget(self.result_detail)

        # 设置滚动区域
        self.scroll_area.setWidget(self.scroll_widget)
        self.layout.addWidget(self.scroll_area)
        self.setLayout(self.layout)

    def run_asr(self):
        # 这里应该调用原始的transcribe_all_audio_under_folder函数
        # 临时实现，实际应用中需要替换为真实的调用
        self.status_label.setText("识别中...")
        QMessageBox.information(self, "功能提示", "AI智能语音识别功能正在实现中...")

        # 实际应用中解除以下注释

        try:
            status, result_json = transcribe_all_audio_under_folder(
                self.video_folder.text(),
                self.asr_model.currentText(),
                self.whisperx_size.value(),
                self.separate_speakers.isChecked(),
            )
            self.status_label.setText(status)
            self.result_detail.setText(str(result_json))
        except Exception as e:
            self.status_label.setText(f"识别失败: {str(e)}")
