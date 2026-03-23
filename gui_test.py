import os
import threading
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pyaudio
import wave
from array import array
from emotion_recognition import EmotionRecognizer
from utils import get_audio_config

# Constants for Recording
THRESHOLD = 500
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
RATE = 16000
SILENCE = 30

class EmotionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Speech Emotion Recognition - Pro Version")
        self.root.geometry("900x700")
        self.root.configure(bg="#2c3e50")

        # Initialize Recognizer
        self.emotions = ["sad", "neutral", "happy", "angry", "fear"]
        self.features = ["mfcc", "chroma", "mel"]
        self.detector = None
        
        self.setup_ui()
        self.init_detector_thread()

    def setup_ui(self):
        # Header
        header = tk.Label(self.root, text="Hệ thống Nhận dạng Cảm xúc Giọng nói", 
                         font=("Helvetica", 20, "bold"), fg="white", bg="#2c3e50", pady=20)
        header.pack()

        # Main Layout
        main_frame = tk.Frame(self.root, bg="#2c3e50")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Left Side - Controls & Result
        control_frame = tk.Frame(main_frame, bg="#34495e", bd=2, relief=tk.RIDGE)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        self.status_label = tk.Label(control_frame, text="Đang khởi tạo mô hình...", 
                                   fg="#f1c40f", bg="#34495e", font=("Helvetica", 12))
        self.status_label.pack(pady=20, padx=20)

        self.record_btn = tk.Button(control_frame, text="🔴 Ghi âm", command=self.start_recording,
                                  font=("Helvetica", 14), bg="#e74c3c", fg="white", 
                                  width=15, state=tk.DISABLED)
        self.record_btn.pack(pady=10, padx=20)

        self.result_title = tk.Label(control_frame, text="Kết quả dự đoán:", 
                                   fg="white", bg="#34495e", font=("Helvetica", 12))
        self.result_title.pack(pady=(30, 0))

        self.result_label = tk.Label(control_frame, text="---", 
                                   fg="#2ecc71", bg="#34495e", font=("Helvetica", 24, "bold"))
        self.result_label.pack(pady=10)

        # Info Frame
        info_frame = tk.Frame(control_frame, bg="#34495e")
        info_frame.pack(pady=20)
        
        tk.Label(info_frame, text="Mô hình: Random Forest", fg="#bdc3c7", bg="#34495e").pack()
        tk.Label(info_frame, text="Dataset: RAVDESS/TESS/EMO-DB", fg="#bdc3c7", bg="#34495e").pack()

        # Right Side - Visualization
        viz_frame = tk.Frame(main_frame, bg="#2c3e50")
        viz_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Matplotlib Figure
        self.fig, self.ax = plt.subplots(figsize=(5, 4), dpi=100)
        self.fig.patch.set_facecolor('#2c3e50')
        self.ax.set_facecolor('#34495e')
        self.ax.tick_params(colors='white')
        self.ax.set_title("Waveform (Âm thanh của bạn)", color='white')
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def init_detector_thread(self):
        def task():
            try:
                # Use a default model if grid fails
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(n_estimators=100)
                self.detector = EmotionRecognizer(model=model, emotions=self.emotions, 
                                                 features=self.features, verbose=0)
                self.detector.train()
                self.root.after(0, self.on_detector_ready)
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Lỗi", f"Không thể khởi tạo mô hình: {e}"))

        threading.Thread(target=task, daemon=True).start()

    def on_detector_ready(self):
        self.status_label.config(text="Hệ thống đã sẵn sàng!", fg="#2ecc71")
        self.record_btn.config(state=tk.NORMAL)

    def start_recording(self):
        self.record_btn.config(state=tk.DISABLED, text="⌛ Đang ghi âm...", bg="#95a5a6")
        self.status_label.config(text="Hãy nói đi, tôi đang lắng nghe...", fg="#3498db")
        self.result_label.config(text="---")
        
        threading.Thread(target=self.record_and_predict, daemon=True).start()

    def record_and_predict(self):
        filename = "gui_test.wav"
        try:
            sample_width, data = self.record()
            self.save_wave(filename, sample_width, data)
            
            # Predict
            result = self.detector.predict(filename)
            
            # Update UI
            self.root.after(0, lambda: self.update_result(result, data))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Lỗi Ghi âm", str(e)))
            self.root.after(0, self.on_detector_ready)

    def update_result(self, result, data):
        self.result_label.config(text=result.upper())
        self.on_detector_ready()
        
        # Update Plot
        self.ax.clear()
        self.ax.plot(data, color='#2ecc71')
        self.ax.set_title(f"Waveform - Cảm xúc: {result}", color='white')
        self.ax.set_facecolor('#34495e')
        self.ax.tick_params(colors='white')
        self.canvas.draw()

    def record(self):
        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK_SIZE)
        
        r = array('h')
        snd_started = False
        num_silent = 0

        while True:
            snd_data = array('h', stream.read(CHUNK_SIZE))
            r.extend(snd_data)
            
            silent = max(snd_data) < THRESHOLD
            if silent and snd_started:
                num_silent += 1
            elif not silent and not snd_started:
                snd_started = True

            if snd_started and num_silent > SILENCE:
                break
        
        sample_width = p.get_sample_size(FORMAT)
        stream.stop_stream()
        stream.close()
        p.terminate()
        return sample_width, r

    def save_wave(self, path, sample_width, data):
        import struct
        data_bytes = struct.pack('<' + ('h'*len(data)), *data)
        wf = wave.open(path, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(sample_width)
        wf.setframerate(RATE)
        wf.writeframes(data_bytes)
        wf.close()

if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionGUI(root)
    root.mainloop()
