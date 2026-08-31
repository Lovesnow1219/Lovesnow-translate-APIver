import os
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QLineEdit,
                               QPushButton, QMessageBox)

from ui_components import VideoPlayer, ResolutionDropdown

# 尝试导入实际的功能模块
try:
    from tools.step000_video_downloader import download_from_url
except ImportError:
    pass


class DownloadTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)

        # 视频URL
        self.video_url = QLineEdit()
        self.video_url.setPlaceholderText("请输入Youtube或Bilibili的视频、播放列表或频道的URL")
        self.video_url.setText("https://www.bilibili.com/video/BV1kr421M7vz/")
        self.layout.addWidget(QLabel("视频URL"))
        self.layout.addWidget(self.video_url)

        # 视频输出文件夹
        self.video_folder = QLineEdit("videos")
        self.layout.addWidget(QLabel("视频输出文件夹"))
        self.layout.addWidget(self.video_folder)

        # 分辨率
        self.resolution = ResolutionDropdown('分辨率', '1080p')
        self.layout.addWidget(self.resolution)

        # 执行按钮
        self.run_button = QPushButton("开始下载")
        self.run_button.clicked.connect(self.run_download)
        self.layout.addWidget(self.run_button)

        # 状态显示
        self.status_label = QLabel("准备就绪")
        self.layout.addWidget(QLabel("下载状态:"))
        self.layout.addWidget(self.status_label)

        # 视频播放器
        self.video_player = VideoPlayer("示例视频")
        self.layout.addWidget(self.video_player)

        # 下载信息
        self.download_info = QLabel("下载信息将显示在这里")
        self.layout.addWidget(QLabel("下载信息:"))
        self.layout.addWidget(self.download_info)

        self.setLayout(self.layout)

    def run_download(self):
        # 这里应该调用原始的download_from_url函数
        # 临时实现，实际应用中需要替换为真实的调用
        self.status_label.setText("下载中...")
        QMessageBox.information(self, "功能提示", "下载功能正在实现中...")

        # 实际应用中解除以下注释

        try:
            status, video_path, info = download_from_url(
                self.video_url.text(),
                self.video_folder.text(),
                self.resolution.value(),
            )
            self.status_label.setText(status)
            if video_path and os.path.exists(video_path):
                self.video_player.set_video(video_path)
            self.download_info.setText(str(info))
        except Exception as e:
            self.status_label.setText(f"下载失败: {str(e)}")
