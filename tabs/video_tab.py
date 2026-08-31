import os
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QLineEdit,
                               QScrollArea, QCheckBox, QPushButton, QMessageBox)

from ui_components import (FloatSlider, CustomSlider,
                           AudioSelector, VideoPlayer, ResolutionDropdown)

# 尝试导入实际的功能模块
try:
    from tools.step050_synthesize_video import synthesize_all_video_under_folder
except ImportError:
    pass


class SynthesizeVideoTab(QWidget):
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

        # 添加字幕
        self.add_subtitles = QCheckBox("添加字幕")
        self.add_subtitles.setChecked(True)
        self.scroll_layout.addWidget(self.add_subtitles)

        # 加速倍数
        self.speed_factor = FloatSlider(0.5, 2, 0.05, "加速倍数", 1.00)
        self.scroll_layout.addWidget(self.speed_factor)

        # 帧率
        self.frame_rate = CustomSlider(1, 60, 1, "帧率", 30)
        self.scroll_layout.addWidget(self.frame_rate)

        # 背景音乐
        self.background_music = AudioSelector("背景音乐")
        self.scroll_layout.addWidget(self.background_music)

        # 背景音乐音量
        self.bg_music_volume = FloatSlider(0, 1, 0.05, "背景音乐音量", 0.5)
        self.scroll_layout.addWidget(self.bg_music_volume)

        # 视频音量
        self.video_volume = FloatSlider(0, 1, 0.05, "视频音量", 1.0)
        self.scroll_layout.addWidget(self.video_volume)

        # 分辨率
        self.resolution = ResolutionDropdown('分辨率', '1080p')
        self.scroll_layout.addWidget(self.resolution)

        # 执行按钮
        self.run_button = QPushButton("开始合成视频")
        self.run_button.clicked.connect(self.run_synthesis)
        self.scroll_layout.addWidget(self.run_button)

        # 状态显示
        self.status_label = QLabel("准备就绪")
        self.scroll_layout.addWidget(QLabel("合成状态:"))
        self.scroll_layout.addWidget(self.status_label)

        # 视频播放器
        self.video_player = VideoPlayer("合成视频")
        self.scroll_layout.addWidget(self.video_player)

        # 设置滚动区域
        self.scroll_area.setWidget(self.scroll_widget)
        self.layout.addWidget(self.scroll_area)
        self.setLayout(self.layout)

    def run_synthesis(self):
        # 这里应该调用原始的synthesize_all_video_under_folder函数
        # 临时实现，实际应用中需要替换为真实的调用
        self.status_label.setText("合成中...")
        QMessageBox.information(self, "功能提示", "视频合成功能正在实现中...")

        # 实际应用中解除以下注释

        try:
            status, video_path = synthesize_all_video_under_folder(
                self.video_folder.text(),
                self.add_subtitles.isChecked(),
                self.speed_factor.value(),
                self.frame_rate.value(),
                self.resolution.value(),
                self.background_music.value(),
                self.bg_music_volume.value(),
                self.video_volume.value(),
            )
            self.status_label.setText(status)
            if video_path and os.path.exists(video_path):
                self.video_player.set_video(video_path)
        except Exception as e:
            self.status_label.setText(f"合成失败: {str(e)}")
