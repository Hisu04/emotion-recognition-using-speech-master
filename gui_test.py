import os
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pyaudio
import wave
from array import array
from core.emotion_recognition import EmotionRecognizer
from core.utils import get_audio_config

# Constants for Recording
THRESHOLD = 500
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
RATE = 16000
SILENCE = 30

class EmotionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Speech Emotion Recognition - Modern UI")
        self.root.geometry("1000x650")
        self.root.configure(bg="#0f172a")

        # Initialize Recognizer
        self.emotions = ["sad", "neutral", "happy", "angry", "fear"]
        self.features = ["mfcc", "chroma", "mel"]
        self.detector = None
        
        self.setup_ui()
        self.init_detector_thread()

    def setup_ui(self):
        # Header
        header = tk.Label(self.root, text="HỆ THỐNG NHẬN DẠNG CẢM XÚC", 
                         font=("Segoe UI", 22, "bold"), fg="#f8fafc", bg="#0f172a", pady=25)
        header.pack()

        # Main Layout
        main_frame = tk.Frame(self.root, bg="#0f172a")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=30, pady=10)

        # Left Side - Controls & Result
        control_frame = tk.Frame(main_frame, bg="#1e293b", bd=0, highlightthickness=1, highlightbackground="#334155")
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=15, pady=10, ipadx=10)

        self.status_label = tk.Label(control_frame, text="Đang khởi tạo mô hình...", 
                                   fg="#38bdf8", bg="#1e293b", font=("Segoe UI", 12))
        self.status_label.pack(pady=25, padx=20)

        self.record_btn = tk.Button(control_frame, text="🔴 Bắt đầu Ghi âm", command=self.start_recording,
                                  font=("Segoe UI", 14, "bold"), bg="#ef4444", fg="white", 
                                  width=18, state=tk.DISABLED, relief=tk.FLAT, activebackground="#dc2626", activeforeground="white")
        self.record_btn.pack(pady=15, padx=20)

        self.file_btn = tk.Button(control_frame, text="📂 Tải file WAV", command=self.predict_from_file,
                                 font=("Segoe UI", 13), bg="#0ea5e9", fg="white",
                                 width=18, state=tk.DISABLED, relief=tk.FLAT, activebackground="#0284c7", activeforeground="white")
        self.file_btn.pack(pady=5, padx=20)

        self.result_title = tk.Label(control_frame, text="Kết quả dự đoán:", 
                                   fg="#94a3b8", bg="#1e293b", font=("Segoe UI", 12))
        self.result_title.pack(pady=(40, 5))

        self.result_label = tk.Label(control_frame, text="---", 
                                   fg="#10b981", bg="#1e293b", font=("Segoe UI", 28, "bold"))
        self.result_label.pack(pady=10)

        # Info Frame
        info_frame = tk.Frame(control_frame, bg="#1e293b")
        info_frame.pack(side=tk.BOTTOM, pady=25)
        
        tk.Label(info_frame, text="Mô hình: Random Forest", fg="#64748b", bg="#1e293b", font=("Segoe UI", 10)).pack()
        tk.Label(info_frame, text="Dataset: RAVDESS/TESS/EMO-DB", fg="#64748b", bg="#1e293b", font=("Segoe UI", 10)).pack()

        # Right Side - Visualization
        viz_frame = tk.Frame(main_frame, bg="#0f172a")
        viz_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))

        # Matplotlib Figure
        self.fig, self.ax = plt.subplots(figsize=(6, 4), dpi=100)
        self.fig.patch.set_facecolor('#0f172a')
        self.ax.set_facecolor('#1e293b')
        self.ax.tick_params(colors='#94a3b8')
        for spine in self.ax.spines.values():
            spine.set_color('#334155')
        self.ax.set_title("Biểu đồ Sóng âm (Waveform)", color='#f8fafc', fontfamily='sans-serif', fontsize=14, pad=15)
        
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
        self.status_label.config(text="Hệ thống đã sẵn sàng!", fg="#10b981")
        self.record_btn.config(state=tk.NORMAL, bg="#6366f1", activebackground="#4f46e5", text="🔴 Bắt đầu Ghi âm")
        self.file_btn.config(state=tk.NORMAL)

    def start_recording(self):
        self.record_btn.config(state=tk.DISABLED, text="⌛ Đang phân tích...", bg="#ef4444")
        self.file_btn.config(state=tk.DISABLED)
        self.status_label.config(text="Hãy nói đi, tôi đang lắng nghe...", fg="#38bdf8")
        self.result_label.config(text="---")
        
        threading.Thread(target=self.record_and_predict, daemon=True).start()

    def predict_from_file(self):
        file_path = filedialog.askopenfilename(
            title="Chọn file âm thanh WAV",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
        )
        if not file_path:
            return
        self.record_btn.config(state=tk.DISABLED)
        self.file_btn.config(state=tk.DISABLED, text="⌛ Đang phân tích...")
        self.status_label.config(text=f"Đang phân tích: {os.path.basename(file_path)}", fg="#38bdf8")
        self.result_label.config(text="---")
        threading.Thread(target=self._predict_file_thread, args=(file_path,), daemon=True).start()

    def _predict_file_thread(self, file_path):
        try:
            result = self.detector.predict(file_path)
            # Đọc waveform từ file để vẽ
            wf = wave.open(file_path, 'rb')
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)
            wf.close()
            import struct
            n_samples = len(raw) // 2
            data = array('h', struct.unpack('<' + 'h' * n_samples, raw))
            self.root.after(0, lambda: self.update_result(result, data, label=os.path.basename(file_path)))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Lỗi", str(e)))
            self.root.after(0, self.on_detector_ready)

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

    def update_result(self, result, data, label=None):
        self.result_label.config(text=result.upper())
        self.on_detector_ready()
        self.file_btn.config(text="📂 Tải file WAV")
        
        # Update Plot
        self.ax.clear()
        self.ax.plot(data, color='#38bdf8', linewidth=1.5)
        title_str = f"Waveform - Cảm xúc: {result}"
        if label:
            title_str += f"  [{label}]"
        self.ax.set_title(title_str, color='#f8fafc', fontfamily='sans-serif', fontsize=13, pad=15)
        self.ax.set_facecolor('#1e293b')
        self.ax.tick_params(colors='#94a3b8')
        for spine in self.ax.spines.values():
            spine.set_color('#334155')
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

        # Tiền xử lý: Tự động loại bỏ tiếng ồn/khoảng lặng dư thừa
        start_idx = 0
        end_idx = len(r) - 1
        
        for i in range(len(r)):
            if abs(r[i]) > THRESHOLD:
                start_idx = i
                break
                
        for i in range(len(r)-1, -1, -1):
            if abs(r[i]) > THRESHOLD:
                end_idx = i
                break
                
        # Chừa lại một biên độ khoảng 0.12s (2000 mảng/frames) để audio tự nhiên
        start_idx = max(0, start_idx - 2000)
        end_idx = min(len(r), end_idx + 2000)
        
        if start_idx < end_idx:
            r = r[start_idx:end_idx]

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
