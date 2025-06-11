import os
import sys
import json
import time
import threading
import subprocess
import wave
import pyaudio
import keyboard
import requests
import numpy as np
import sounddevice as sd
from pydub import AudioSegment
import PyQt5
dirname = os.path.dirname(PyQt5.__file__)
qt_dir = os.path.join(dirname, 'Qt5', 'plugins', 'platforms')
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = qt_dir #
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTextEdit,
                             QVBoxLayout, QWidget, QPushButton,
                             QFileDialog, QLabel)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt5.QtGui import QPixmap, QColor


class Config:
    def __init__(self):
        self.wake_word = "恺撒"
        self.deepseek_url = "http://localhost:5000/v1/chat/completions"
        self.funasr_url = "http://localhost:8000/asr"
        self.history_size = 5
        self.audio_input_device = None
        self.audio_output_device = None
        self.sample_rate = 16000
        self.channels = 1
        self.chunk_size = 1024
        self.silence_threshold = 500
        self.silence_duration = 1.5
        self.current_bg = "default_bg.jpg"


class AIVoiceAssistant(QObject):
    new_message = pyqtSignal(str, str)
    listening_changed = pyqtSignal(bool)

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.conversation_history = []
        self.is_listening = False
        self.is_processing = False
        self.audio = pyaudio.PyAudio()
        self.engine = None
        self.init_tts()

    def init_tts(self):
        try:
            import pyttsx3
            self.engine = pyttsx3.init()
            voices = self.engine.getProperty('voices')
            if len(voices) > 1:
                self.engine.setProperty('voice', voices[1].id)
            self.engine.setProperty('rate', 150)
        except Exception as e:
            print(f"TTS初始化失败: {e}")
            self.engine = None

    def speak(self, text):
        self.new_message.emit(text, "assistant")
        if self.engine:
            self.engine.say(text)
            self.engine.runAndWait()

    def listen(self):
        if self.is_processing:
            return

        self.is_listening = True
        self.listening_changed.emit(True)
        self.new_message.emit("正在聆听...", "system")

        frames = self.record_audio()
        self.is_listening = False
        self.listening_changed.emit(False)

        if frames:
            temp_file = "temp_audio.wav"
            self.save_audio(frames, temp_file)
            text = self.transcribe_audio(temp_file)
            if text:
                self.new_message.emit(text, "user")
                self.process_query(text)

    def record_audio(self):
        stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=self.config.channels,
            rate=self.config.sample_rate,
            input=True,
            frames_per_buffer=self.config.chunk_size,
            input_device_index=self.config.audio_input_device
        )

        frames = []
        silent_frames = 0
        silence_limit = int(self.config.silence_duration * self.config.sample_rate / self.config.chunk_size)
        is_speaking = False

        while True:
            data = stream.read(self.config.chunk_size, exception_on_overflow=False)
            frames.append(data)

            if keyboard.is_pressed('esc'):
                break

            audio_data = np.frombuffer(data, dtype=np.int16)
            if np.abs(audio_data).mean() < self.config.silence_threshold:
                silent_frames += 1
                if is_speaking and silent_frames > silence_limit:
                    break
            else:
                silent_frames = 0
                is_speaking = True

        stream.stop_stream()
        stream.close()
        return frames

    def save_audio(self, frames, filename):
        wf = wave.open(filename, 'wb')
        wf.setnchannels(self.config.channels)
        wf.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(self.config.sample_rate)
        wf.writeframes(b''.join(frames))
        wf.close()

    def transcribe_audio(self, audio_file):
        try:
            with open(audio_file, 'rb') as f:
                files = {'audio_file': f}
                response = requests.post(self.config.funasr_url, files=files)

            if response.status_code == 200:
                return response.json().get('text', '').strip()
        except Exception as e:
            print(f"语音识别错误: {e}")
            return None

    def process_query(self, query):
        if not query:
            return

        self.is_processing = True

        if self.config.wake_word in query:
            query = query.replace(self.config.wake_word, "").strip()
            if not query:
                self.speak("我在，请说")
                self.is_processing = False
                return

        if self.handle_command(query):
            self.is_processing = False
            return

        self.add_to_history("user", query)
        response = self.call_deepseek_api(query)

        if response:
            self.add_to_history("assistant", response)
            self.speak(response)

        self.is_processing = False

    def handle_command(self, query):
        query_lower = query.lower()
        app_commands = {
            "打开记事本": "notepad",
            "打开计算器": "calc",
            "打开浏览器": "start chrome",
            "打开文件资源管理器": "explorer",
        }

        for cmd, app in app_commands.items():
            if cmd in query:
                try:
                    subprocess.Popen(app, shell=True)
                    self.speak(f"正在打开{cmd.replace('打开', '')}")
                    return True
                except Exception as e:
                    self.speak(f"无法打开应用程序: {e}")
                    return True

        if "停止监听" in query or "退出" in query:
            self.speak("再见")
            QApplication.quit()
            return True
        elif "清空历史" in query:
            self.conversation_history = []
            self.speak("已清空对话历史")
            return True

        return False

    def call_deepseek_api(self, query):
        headers = {"Content-Type": "application/json"}
        messages = [{"role": "system", "content": "你是一个有帮助的AI助手"}]
        for item in self.conversation_history[-self.config.history_size:]:
            messages.append({"role": item["role"], "content": item["content"]})

        data = {
            "model": "deepseek-chat",
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 2000
        }

        try:
            response = requests.post(
                self.config.deepseek_url,
                headers=headers,
                data=json.dumps(data)
            )

            if response.status_code == 200:
                return response.json()['choices']['message']['content']
        except Exception as e:
            print(f"API调用错误: {e}")
            return "服务暂时不可用"

    def add_to_history(self, role, content):
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": time.time()
        })


class VoiceAssistantUI(QMainWindow):
    def __init__(self, assistant):
        super().__init__()
        self.assistant = assistant
        self.init_ui()
        self.setup_connections()

    def init_ui(self):
        self.setWindowTitle("AI语音助手")
        self.setGeometry(100, 100, 800, 600)
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout()
        self.main_widget.setLayout(self.layout)

        self.background_label = QLabel()
        self.background_label.setAlignment(Qt.AlignCenter)
        self.background_label.setStyleSheet("background-color: #f0f0f0;")
        self.layout.addWidget(self.background_label)
        self.update_background(self.assistant.config.current_bg)

        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setStyleSheet("""
            QTextEdit {
                background-color: rgba(255, 255, 255, 0.7);
                border-radius: 10px;
                padding: 10px;
                font-size: 14px;
            }
        """)
        self.layout.addWidget(self.chat_display)

        self.button_layout = QVBoxLayout()
        self.listen_button = QPushButton("开始监听")
        self.listen_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 5px;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.button_layout.addWidget(self.listen_button)

        self.bg_button = QPushButton("更换背景")
        self.bg_button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 5px;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #0b7dda;
            }
        """)
        self.button_layout.addWidget(self.bg_button)
        self.layout.addLayout(self.button_layout)

    def setup_connections(self):
        self.listen_button.clicked.connect(self.toggle_listening)
        self.bg_button.clicked.connect(self.change_background)
        self.assistant.new_message.connect(self.display_message)
        self.assistant.listening_changed.connect(self.update_listen_button)

    def toggle_listening(self):
        if not self.assistant.is_processing:
            threading.Thread(target=self.assistant.listen, daemon=True).start()

    def update_listen_button(self, is_listening):
        if is_listening:
            self.listen_button.setText("停止监听")
            self.listen_button.setStyleSheet("background-color: #f44336; color: white;")
        else:
            self.listen_button.setText("开始监听")
            self.listen_button.setStyleSheet("background-color: #4CAF50; color: white;")

    def display_message(self, message, msg_type):
        if msg_type == "user":
            prefix = "你: "
            color = "#2e7d32"
        elif msg_type == "assistant":
            prefix = "AI: "
            color = "#1565c0"
        else:
            prefix = "系统: "
            color = "#616161"

        self.chat_display.append(f'<span style="color:{color};"><b>{prefix}</b>{message}</span>')

    def change_background(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "选择背景图片", "",
            "图片文件 (*.jpg *.jpeg *.png *.bmp)"
        )
        if file_name:
            self.assistant.config.current_bg = file_name
            self.update_background(file_name)

    def update_background(self, image_path):
        if os.path.exists(image_path):
            pixmap = QPixmap(image_path)
            scaled_pixmap = pixmap.scaled(
                self.size(),
                Qt.KeepAspectRatioByExpanding,
                Qt.SmoothTransformation
            )
            self.background_label.setPixmap(scaled_pixmap)


def main():
    config = Config()
    app = QApplication(sys.argv)
    assistant = AIVoiceAssistant(config)
    ui = VoiceAssistantUI(assistant)
    ui.show()
    keyboard.add_hotkey('ctrl+alt+a', lambda: threading.Thread(target=assistant.listen, daemon=True).start())
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
