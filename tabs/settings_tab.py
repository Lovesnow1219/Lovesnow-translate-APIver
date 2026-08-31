import os
import json
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea,
                               QPushButton, QMessageBox, QSplitter)
from PySide6.QtCore import Qt, Signal

from ui_components import (CustomSlider, FloatSlider, RadioButtonGroup,
                           AudioSelector, VideoPlayer, ResolutionDropdown)

try:
    from tools.target_language import TARGET_LANGUAGES, translation_language
except ImportError:
    TARGET_LANGUAGES = [
        'English', 'Japanese', '越南文', '簡體中文', '繁體中文', '粵語',
        'Korean（部分支援）', '西班牙文（部分支援）', 'French（部分支援）',
        '泰文（部分支援）', '印尼文（部分支援）', '馬來文（部分支援）', '菲律賓文（部分支援）',
    ]

    def translation_language(label):
        return label


class SettingsTab(QWidget):
    """
    配置页面，允许用户设置所有处理参数并保存到config.json
    """
    # 定义配置变更信号
    config_changed = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        self.load_config()

    def init_ui(self):
        """初始化配置页面UI"""
        self.layout = QVBoxLayout(self)

        # 创建一个滚动区域用于容纳配置项
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_widget = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_widget)

        # 添加所有配置控件
        self.add_config_widgets()

        # 创建保存配置按钮
        self.button_layout = QHBoxLayout()
        self.save_config_button = QPushButton("保存配置")
        self.save_config_button.clicked.connect(self.save_config)
        self.save_config_button.setMinimumHeight(40)
        self.save_config_button.setStyleSheet("background-color: #4CAF50; color: white;")
        self.reset_config_button = QPushButton("重置配置")
        self.reset_config_button.clicked.connect(self.reset_config)
        self.reset_config_button.setMinimumHeight(40)
        self.button_layout.addWidget(self.reset_config_button)
        self.button_layout.addWidget(self.save_config_button)

        # 设置滚动区域
        self.scroll_area.setWidget(self.scroll_widget)
        self.layout.addWidget(self.scroll_area)
        self.layout.addLayout(self.button_layout)
        self.setLayout(self.layout)

    def add_config_widgets(self):
        """添加所有配置控件"""
        # 视频配置
        self.scroll_layout.addWidget(QLabel("=== 视频下载配置 ==="))
        self.scroll_layout.addWidget(QLabel("视频输出文件夹"))
        self.video_folder = self.add_label_value("videos", "视频输出到此文件夹")

        # 分辨率
        self.resolution = ResolutionDropdown('分辨率', '1080p')
        self.scroll_layout.addWidget(self.resolution)

        # 音频处理配置
        self.scroll_layout.addWidget(QLabel("=== 音频处理配置 ==="))
        # 模型
        self.model = RadioButtonGroup(
            ['htdemucs', 'htdemucs_ft', 'htdemucs_6s', 'hdemucs_mmi', 'mdx', 'mdx_extra', 'mdx_q', 'mdx_extra_q',
             'SIG'],
            "人声分离模型",
            'htdemucs_ft'
        )
        self.scroll_layout.addWidget(self.model)

        # 移位次数
        self.shifts = CustomSlider(0, 10, 1, "移位次数 Number of shifts", 5)
        self.scroll_layout.addWidget(self.shifts)

        # ASR配置
        self.scroll_layout.addWidget(QLabel("=== 语音识别配置 ==="))
        # ASR模型选择
        self.asr_model_label = QLabel("ASR模型选择")
        self.scroll_layout.addWidget(self.asr_model_label)
        self.asr_model = RadioButtonGroup(['WhisperX', 'FunASR'], "ASR模型选择", 'WhisperX')
        self.scroll_layout.addWidget(self.asr_model)

        # WhisperX模型大小
        self.whisperx_size = RadioButtonGroup(['large', 'medium', 'small', 'base', 'tiny'], "WhisperX模型大小", 'large')
        self.scroll_layout.addWidget(self.whisperx_size)

        # 分离多个说话人
        self.separate_speakers = RadioButtonGroup([True, False], "分离多个说话人", True)
        self.scroll_layout.addWidget(self.separate_speakers)

        # 翻译配置
        self.scroll_layout.addWidget(QLabel("=== 翻译配置 ==="))
        # 翻译方式
        self.translation_method_label = QLabel("翻译方式")
        self.scroll_layout.addWidget(self.translation_method_label)
        self.translation_method = RadioButtonGroup(
            ['OpenAI', 'LLM', 'Google Translate', 'Bing Translate', 'Ernie', '火山引擎-deepseek', "deepseek-api",
             "阿里云-通义千问","Ollama"], "翻译方式", 'LLM')
        self.scroll_layout.addWidget(self.translation_method)

        # 目标语言 (翻译)
        self.target_language_translation_label = QLabel("目标语言 (翻译)")
        self.scroll_layout.addWidget(self.target_language_translation_label)
        self.target_language_translation = RadioButtonGroup(
            TARGET_LANGUAGES, "目标语言 (翻译)", '簡體中文')
        self.scroll_layout.addWidget(self.target_language_translation)

        # TTS配置
        self.scroll_layout.addWidget(QLabel("=== 语音合成配置 ==="))
        # AI语音生成方法
        self.tts_method_label = QLabel("AI语音生成方法")
        self.scroll_layout.addWidget(self.tts_method_label)
        self.tts_method = RadioButtonGroup(['xtts', 'cosyvoice', 'EdgeTTS'], "AI语音生成方法", 'EdgeTTS')
        self.scroll_layout.addWidget(self.tts_method)

        # 目标语言 (TTS)
        self.target_language_tts_label = QLabel("目标语言 (TTS)")
        self.scroll_layout.addWidget(self.target_language_tts_label)
        self.target_language_tts = RadioButtonGroup(
            ['中文', 'English', '粤语', 'Japanese', 'Korean', 'Spanish', 'French',
             'Vietnamese', 'Thai', 'Indonesian', 'Malay', 'Filipino'], "目标语言 (TTS)", '中文')
        self.scroll_layout.addWidget(self.target_language_tts)

        # EdgeTTS声音选择
        self.edge_tts_voice_label = QLabel("EdgeTTS声音选择")
        self.scroll_layout.addWidget(self.edge_tts_voice_label)
        self.edge_tts_voice = RadioButtonGroup(
            ['zh-CN-XiaoxiaoNeural', 'zh-CN-YunxiNeural', 'en-US-JennyNeural', 'ja-JP-NanamiNeural'],
            "EdgeTTS声音选择", 'zh-CN-XiaoxiaoNeural')
        self.scroll_layout.addWidget(self.edge_tts_voice)

        # 视频合成配置
        self.scroll_layout.addWidget(QLabel("=== 视频合成配置 ==="))
        # 添加字幕
        self.add_subtitles = RadioButtonGroup([True, False], "添加字幕", True)
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

        # 分辨率 (输出)
        self.output_resolution = ResolutionDropdown('输出分辨率', '1080p')
        self.scroll_layout.addWidget(self.output_resolution)

        # 高级配置
        self.scroll_layout.addWidget(QLabel("=== 高级配置 ==="))
        # Max Workers
        self.max_workers = CustomSlider(1, 100, 1, "Max Workers", 1)
        self.scroll_layout.addWidget(self.max_workers)

        # Max Retries
        self.max_retries = CustomSlider(1, 10, 1, "Max Retries", 3)
        self.scroll_layout.addWidget(self.max_retries)

    def add_label_value(self, value, tooltip=None):
        """为简单的文本值创建一个标签"""
        label = QLabel(value)
        if tooltip:
            label.setToolTip(tooltip)
        self.scroll_layout.addWidget(label)
        return label

    def get_config(self):
        """从UI控件中获取当前配置"""
        config = {
            "video_folder": self.video_folder.text(),
            "resolution": self.resolution.value(),
            "model": self.model.value(),
            "shifts": self.shifts.value(),
            "asr_model": self.asr_model.value(),
            "whisperx_size": self.whisperx_size.value(),
            "separate_speakers": self.separate_speakers.value(),
            "translation_method": self.translation_method.value(),
            "target_language_translation": self.target_language_translation.value(),
            "tts_method": self.tts_method.value(),
            "target_language_tts": self.target_language_tts.value(),
            "edge_tts_voice": self.edge_tts_voice.value(),
            "add_subtitles": self.add_subtitles.value(),
            "speed_factor": self.speed_factor.value(),
            "frame_rate": self.frame_rate.value(),
            "background_music": self.background_music.value(),
            "bg_music_volume": self.bg_music_volume.value(),
            "video_volume": self.video_volume.value(),
            "output_resolution": self.output_resolution.value(),
            "max_workers": self.max_workers.value(),
            "max_retries": self.max_retries.value()
        }
        return config

    def apply_config(self, config):
        """将配置应用到UI控件"""
        try:
            # 基本设置
            self.video_folder.setText(config.get("video_folder", "videos"))

            # 为每个单选按钮组应用更健壮的选择逻辑
            # 分辨率
            resolution_value = config.get("resolution", "1080p")
            self.resolution.setValue(resolution_value)

            # 模型
            model_value = config.get("model", "htdemucs_ft")
            self._set_radio_button(self.model.buttons, model_value, "htdemucs_ft")

            # 移位次数
            self.shifts.setValue(config.get("shifts", 5))

            # ASR模型
            asr_model_value = config.get("asr_model", "WhisperX")
            self._set_radio_button(self.asr_model.buttons, asr_model_value, "WhisperX")

            # WhisperX模型大小
            whisperx_size_value = config.get("whisperx_size", "large")
            self._set_radio_button(self.whisperx_size.buttons, whisperx_size_value, "large")

            # 分离多个说话人
            separate_speakers_value = config.get("separate_speakers", True)
            self._set_radio_button(self.separate_speakers.buttons, separate_speakers_value, True)

            # 翻译方式
            translation_method_value = config.get("translation_method", "LLM")
            self._set_radio_button(self.translation_method.buttons, translation_method_value, "LLM")

            # 目标语言 (翻译)
            target_lang_trans_value = config.get("target_language_translation", "簡體中文")
            self._set_language_radio(
                self.target_language_translation.buttons, target_lang_trans_value, "簡體中文")

            # TTS方法
            tts_method_value = config.get("tts_method", "EdgeTTS")
            self._set_radio_button(self.tts_method.buttons, tts_method_value, "EdgeTTS")

            # 目标语言 (TTS)
            target_lang_tts_value = config.get("target_language_tts", "中文")
            self._set_radio_button(self.target_language_tts.buttons, target_lang_tts_value, "中文")

            # EdgeTTS声音选择
            edge_tts_voice_value = config.get("edge_tts_voice", "zh-CN-XiaoxiaoNeural")
            self._set_radio_button(self.edge_tts_voice.buttons, edge_tts_voice_value, "zh-CN-XiaoxiaoNeural")

            # 添加字幕
            add_subtitles_value = config.get("add_subtitles", True)
            self._set_radio_button(self.add_subtitles.buttons, add_subtitles_value, True)

            # 加速倍数
            self.speed_factor.setValue(config.get("speed_factor", 1.00))

            # 帧率
            self.frame_rate.setValue(config.get("frame_rate", 30))

            # 背景音乐
            if config.get("background_music"):
                self.background_music.file_path.setText(config.get("background_music"))

            # 背景音乐音量
            self.bg_music_volume.setValue(config.get("bg_music_volume", 0.5))

            # 视频音量
            self.video_volume.setValue(config.get("video_volume", 1.0))

            # 输出分辨率
            output_resolution_value = config.get("output_resolution", "1080p")
            self.output_resolution.setValue(output_resolution_value)

            # 最大工作线程数
            self.max_workers.setValue(config.get("max_workers", 1))

            # 最大重试次数
            self.max_retries.setValue(config.get("max_retries", 3))

        except Exception as e:
            QMessageBox.warning(self, "配置加载错误", f"加载配置时出错: {str(e)}")

    def _set_radio_button(self, buttons, value, default_value):
        """辅助方法：安全地设置单选按钮值"""
        try:
            # 尝试找到匹配的选项并选中对应的按钮
            for option, button in buttons:
                if option == value:
                    button.setChecked(True)
                    return

            # 如果没找到，使用默认值
            for option, button in buttons:
                if option == default_value:
                    button.setChecked(True)
                    return
        except Exception:
            # 如果出现任何错误，尝试使用默认值
            for option, button in buttons:
                if option == default_value:
                    button.setChecked(True)
                    return

    def _set_language_radio(self, buttons, value, default_value):
        """Match translation labels even with （部分支援） or 简/簡 variants."""
        try:
            want = translation_language(value)
            for option, button in buttons:
                if option == value or translation_language(option) == want:
                    button.setChecked(True)
                    return
            for option, button in buttons:
                if option == default_value or translation_language(option) == translation_language(default_value):
                    button.setChecked(True)
                    return
        except Exception:
            self._set_radio_button(buttons, value, default_value)

    def save_config(self):
        """保存配置到JSON文件"""
        try:
            config = self.get_config()
            config_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(config_dir, "config.json")

            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=4)

            QMessageBox.information(self, "保存成功", f"配置已保存到 {config_path}")

            # 发送配置变更信号
            self.config_changed.emit(config)

        except Exception as e:
            QMessageBox.critical(self, "保存失败", f"保存配置时出错: {str(e)}")

    def load_config(self):
        """从JSON文件加载配置"""
        try:
            config_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(config_dir, "config.json")

            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                self.apply_config(config)
        except Exception as e:
            QMessageBox.warning(self, "加载配置失败", f"加载配置时出错: {str(e)}")

    def reset_config(self):
        """重置配置为默认值"""
        if QMessageBox.question(self, "确认重置", "确定要重置所有配置到默认值吗？") == QMessageBox.Yes:
            # 应用默认配置
            default_config = {
                "video_folder": "videos",
                "resolution": "1080p",
                "model": "htdemucs_ft",
                "shifts": 5,
                "asr_model": "WhisperX",
                "whisperx_size": "large",
                "separate_speakers": True,
                "translation_method": "LLM",
                "target_language_translation": "簡體中文",
                "tts_method": "EdgeTTS",
                "target_language_tts": "中文",
                "edge_tts_voice": "zh-CN-XiaoxiaoNeural",
                "add_subtitles": True,
                "speed_factor": 1.00,
                "frame_rate": 30,
                "background_music": None,
                "bg_music_volume": 0.5,
                "video_volume": 1.0,
                "output_resolution": "1080p",
                "max_workers": 1,
                "max_retries": 3
            }
            self.apply_config(default_config)
            QMessageBox.information(self, "重置成功", "所有配置已重置为默认值")