import os
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
                               QPushButton, QMessageBox, QCheckBox, QGroupBox)

from ui_components import CustomSlider, RadioButtonGroup

# 尝试导入实际的功能模块
try:
    from tools.step010_demucs_vr import separate_all_audio_under_folder
except ImportError:
    pass


class DemucsTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)

        # 视频文件夹
        self.video_folder = QLineEdit("videos")
        self.layout.addWidget(QLabel("视频文件夹"))
        self.layout.addWidget(self.video_folder)

        # 模型
        self.model = RadioButtonGroup(
            ['htdemucs', 'htdemucs_ft', 'htdemucs_6s', 'hdemucs_mmi', 'mdx', 'mdx_extra', 'mdx_q', 'mdx_extra_q',
             'SIG'],
            "模型",
            'htdemucs_ft'
        )
        self.layout.addWidget(self.model)

        # 显示进度条
        self.show_progress = QCheckBox("显示进度条")
        self.show_progress.setChecked(True)
        self.layout.addWidget(self.show_progress)

        # 移位次数
        self.shifts = CustomSlider(0, 10, 1, "移位次数 Number of shifts", 5)
        self.layout.addWidget(self.shifts)

        # 执行按钮
        self.run_button = QPushButton("开始分离")
        self.run_button.clicked.connect(self.run_separation)
        self.layout.addWidget(self.run_button)

        # 状态显示
        self.status_label = QLabel("准备就绪")
        self.layout.addWidget(QLabel("分离结果状态:"))
        self.layout.addWidget(self.status_label)

        # 音频播放控件
        vocals_group = QGroupBox("人声音频")
        vocals_layout = QVBoxLayout()
        self.vocals_play_button = QPushButton("播放人声")
        vocals_layout.addWidget(self.vocals_play_button)
        vocals_group.setLayout(vocals_layout)

        accompaniment_group = QGroupBox("伴奏音频")
        accompaniment_layout = QVBoxLayout()
        self.accompaniment_play_button = QPushButton("播放伴奏")
        accompaniment_layout.addWidget(self.accompaniment_play_button)
        accompaniment_group.setLayout(accompaniment_layout)

        audio_layout = QHBoxLayout()
        audio_layout.addWidget(vocals_group)
        audio_layout.addWidget(accompaniment_group)

        self.layout.addLayout(audio_layout)
        self.setLayout(self.layout)

    def run_separation(self):
        # 这里应该调用原始的separate_all_audio_under_folder函数
        # 临时实现，实际应用中需要替换为真实的调用
        self.status_label.setText("分离中...")
        QMessageBox.information(self, "功能提示", "人声分离功能正在实现中...")

        # 实际应用中解除以下注释

        try:
            status, vocals_path, accompaniment_path = separate_all_audio_under_folder(
                self.video_folder.text(),
                self.model.value(),
                self.show_progress.isChecked(),
                self.shifts.value()
            )
            self.status_label.setText(status)
            if vocals_path and os.path.exists(vocals_path):
                self.vocals_play_button.setEnabled(True)
            if accompaniment_path and os.path.exists(accompaniment_path):
                self.accompaniment_play_button.setEnabled(True)
        except Exception as e:
            self.status_label.setText(f"分离失败: {str(e)}")
