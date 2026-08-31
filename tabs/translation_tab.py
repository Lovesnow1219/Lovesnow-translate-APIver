import os
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QLineEdit,
                               QComboBox, QPushButton, QMessageBox, QScrollArea)

# 尝试导入实际的功能模块
try:
    from tools.step030_translation import translate_all_transcript_under_folder
    from tools.target_language import TARGET_LANGUAGES
except ImportError:
    TARGET_LANGUAGES = [
        'English', 'Japanese', '越南文', '簡體中文', '繁體中文', '粵語',
        'Korean（部分支援）', '西班牙文（部分支援）', 'French（部分支援）',
        '泰文（部分支援）', '印尼文（部分支援）', '馬來文（部分支援）', '菲律賓文（部分支援）',
    ]


class TranslationTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)

        # 视频文件夹
        self.video_folder = QLineEdit("videos")
        self.layout.addWidget(QLabel("视频文件夹"))
        self.layout.addWidget(self.video_folder)

        # 翻译方式
        self.translation_method = QComboBox()
        self.translation_method.addItems(['OpenAI', 'LLM', 'Google Translate', 'Bing Translate', 'Ernie'])
        self.translation_method.setCurrentText('LLM')
        self.layout.addWidget(QLabel("翻译方式"))
        self.layout.addWidget(self.translation_method)

        # 目标语言
        self.target_language = QComboBox()
        self.target_language.setEditable(False)
        self.target_language.addItems(TARGET_LANGUAGES)
        self.layout.addWidget(QLabel("目标语言"))
        self.layout.addWidget(self.target_language)

        # 执行按钮
        self.run_button = QPushButton("开始翻译")
        self.run_button.clicked.connect(self.run_translation)
        self.layout.addWidget(self.run_button)

        # 状态显示
        self.status_label = QLabel("准备就绪")
        self.layout.addWidget(QLabel("翻译状态:"))
        self.layout.addWidget(self.status_label)

        # 总结结果
        self.summary_label = QLabel("总结结果将显示在这里")
        self.layout.addWidget(QLabel("总结结果:"))
        self.layout.addWidget(self.summary_label)

        # 翻译结果
        self.translation_result = QLabel("翻译结果将显示在这里")
        self.layout.addWidget(QLabel("翻译结果:"))

        # 使用滚动区域显示详细结果
        result_scroll = QScrollArea()
        result_scroll.setWidgetResizable(True)
        result_scroll.setWidget(self.translation_result)
        self.layout.addWidget(result_scroll)

        self.setLayout(self.layout)

    def run_translation(self):
        # 这里应该调用原始的translate_all_transcript_under_folder函数
        # 临时实现，实际应用中需要替换为真实的调用
        self.status_label.setText("翻译中...")
        QMessageBox.information(self, "功能提示", "字幕翻译功能正在实现中...")

        # 实际应用中解除以下注释

        try:
            status, summary, translation = translate_all_transcript_under_folder(
                self.video_folder.text(),
                self.translation_method.currentText(),
                self.target_language.currentText()
            )
            self.status_label.setText(status)
            self.summary_label.setText(str(summary))
            self.translation_result.setText(str(translation))
        except Exception as e:
            self.status_label.setText(f"翻译失败: {str(e)}")
