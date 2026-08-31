import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QTabWidget
from PySide6.QtCore import Qt

from tabs.full_auto_tab import FullAutoTab
from tabs.settings_tab import SettingsTab
from tabs.download_tab import DownloadTab
from tabs.demucs_tab import DemucsTab
from tabs.asr_tab import ASRTab
from tabs.translation_tab import TranslationTab
from tabs.tts_tab import TTSTab
from tabs.video_tab import SynthesizeVideoTab


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("视频AI配音/翻译工具 API版-lovesnow 開源自Linly-Dubbing")
        self.resize(1024, 768)

        self.tab_widget = QTabWidget()
        self.full_auto_tab = FullAutoTab()
        self.settings_tab = SettingsTab()
        self.settings_tab.config_changed.connect(self.full_auto_tab.update_config)

        self.tab_widget.addTab(self.full_auto_tab, "一键自动化 One-Click")
        self.tab_widget.addTab(self.settings_tab, "配置页面 Settings")
        self.tab_widget.addTab(DownloadTab(), "自动下载视频")
        self.tab_widget.addTab(DemucsTab(), "人声分离")
        self.tab_widget.addTab(ASRTab(), "AI智能语音识别")
        self.tab_widget.addTab(TranslationTab(), "字幕翻译")
        self.tab_widget.addTab(TTSTab(), "AI语音合成")
        self.tab_widget.addTab(SynthesizeVideoTab(), "视频合成")
        self.setCentralWidget(self.tab_widget)


def main():
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
