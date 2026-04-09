# 🧠 Dự án Hoạt động Như Thế Nào?
### Hướng dẫn toàn diện — Dành cho người mới bắt đầu

> Tài liệu này giải thích **từng bước**, **từng luồng chạy** của hệ thống nhận dạng cảm xúc qua giọng nói, theo ngôn ngữ đơn giản nhất có thể. Bạn sẽ hiểu _tại sao_ mỗi dòng code tồn tại và _điều gì thực sự xảy ra_ bên trong máy tính.

---

## 📖 Mục lục

1. [Ý tưởng tổng thể — Dự án làm gì?](#1-ý-tưởng-tổng-thể--dự-án-làm-gì)
2. [Bức tranh toàn cảnh — Sơ đồ hệ thống](#2-bức-tranh-toàn-cảnh--sơ-đồ-hệ-thống)
3. [Luồng 1 — Chạy demo nhanh (demo_predict.py)](#3-luồng-1--chạy-demo-nhanh-demo_predictpy)
4. [Luồng 2 — Ghi âm realtime qua CLI (test.py)](#4-luồng-2--ghi-âm-realtime-qua-cli-testpy)
5. [Luồng 3 — Ứng dụng GUI (gui_test.py)](#5-luồng-3--ứng-dụng-gui-gui_testpy)
6. [Luồng cốt lõi — EmotionRecognizer hoạt động như nào?](#6-luồng-cốt-lõi--emotionrecognizer-hoạt-động-như-nào)
7. [Chi tiết bước 1 — Sinh file CSV metadata](#7-chi-tiết-bước-1--sinh-file-csv-metadata)
8. [Chi tiết bước 2 — Nạp dữ liệu và trích xuất đặc trưng](#8-chi-tiết-bước-2--nạp-dữ-liệu-và-trích-xuất-đặc-trưng)
9. [Chi tiết bước 3 — Huấn luyện mô hình](#9-chi-tiết-bước-3--huấn-luyện-mô-hình)
10. [Chi tiết bước 4 — Dự đoán cảm xúc từ file WAV](#10-chi-tiết-bước-4--dự-đoán-cảm-xúc-từ-file-wav)
11. [Luồng ghi âm — Microphone hoạt động như nào?](#11-luồng-ghi-âm--microphone-hoạt-động-như-nào)
12. [Hệ thống cache — Tại sao lần 2 lại nhanh hơn?](#12-hệ-thống-cache--tại-sao-lần-2-lại-nhanh-hơn)
13. [Các mô hình ML được dùng](#13-các-mô-hình-ml-được-dùng)
14. [Câu hỏi thường gặp của người mới](#14-câu-hỏi-thường-gặp-của-người-mới)

---

## 1. Ý tưởng tổng thể — Dự án làm gì?

Hãy tưởng tượng bạn nói một câu vào microphone. Máy tính sẽ:

```
Giọng nói của bạn  →  Phân tích âm thanh  →  "happy" / "sad" / "angry"...
```

**Nguyên lý:** Khi con người nói với các cảm xúc khác nhau, giọng nói sẽ có đặc điểm _vật lý_ khác nhau — cao độ, tốc độ, năng lượng... Mô hình AI học từ hàng nghìn mẫu giọng nói đã được gán nhãn cảm xúc, sau đó dự đoán cảm xúc của giọng nói mới.

### 3 cách dùng hệ thống:

| Cách | File | Dùng khi |
|---|---|---|
| 🔬 Demo tự động | `demo_predict.py` | Muốn test pipeline nhanh, không cần mic |
| 🎤 Ghi âm CLI | `test.py` | Muốn nói chuyện qua terminal |
| 🖥️ Giao diện GUI | `gui_test.py` | Muốn giao diện đẹp, trực quan |

---

## 2. Bức tranh toàn cảnh — Sơ đồ hệ thống

```
╔══════════════════════════════════════════════════════════════════╗
║                    TỔNG QUAN HỆ THỐNG                           ║
╚══════════════════════════════════════════════════════════════════╝

   📂 DỮ LIỆU THÔ                    🔧 XỬ LÝ                    🎯 KẾT QUẢ
   ─────────────                    ──────────                    ──────────
   
   data/training/        ──┐
   data/validation/      ──┤   create_csv.py        metadata/*.csv
   data/emodb/wav/       ──┘──────────────►     (danh sách path+nhãn)
   data/train-custom/    ──┘                              │
                                                          │
                                               data_extractor.py
                                               (đọc CSV, nạp WAV)
                                                          │
                                                          ▼
                                                   utils.py
                                              extract_feature()
                                                          │
                                         ┌────────────────────────┐
                                         │  MFCC (40D)            │
                                         │  Chroma (12D)          │──► vector 180D
                                         │  Mel Spectrogram(128D) │
                                         └────────────────────────┘
                                                          │
                                               features/*.npy (cache)
                                                          │
                                              ┌───────────┴──────────┐
                                              │                      │
                                     sklearn classifier        Keras LSTM/GRU
                                     (svm, mlp, rf...)        deep_emotion...
                                              │                      │
                                              └───────────┬──────────┘
                                                          │
                                                    .predict()
                                                          │
                                    ┌─────────────────────┴──────────────────────┐
                                    │                                            │
                              GUI (gui_test.py)                         CLI (test.py)
                         "HAPPY" hiện lên màn hình               print("happy")
```

---

## 3. Luồng 1 — Chạy demo nhanh (`demo_predict.py`)

**Dùng khi:** Bạn chưa có microphone hoặc chỉ muốn xem pipeline hoạt động.

### Chạy lệnh:
```bash
python demo_predict.py
```

### Từng bước xảy ra:

```
╔══════════════════════════════════════════════════════╗
║         LUỒNG CHẠY demo_predict.py                  ║
╚══════════════════════════════════════════════════════╝

BƯỚC 1: Chọn ngẫu nhiên 1 file WAV từ data/emodb/wav/
        │
        │  wav_files = os.listdir("data/emodb/wav/")
        │  sample_file = random.choice(wav_files)
        │  → Ví dụ: "03a04Wc.wav"
        │
        ▼
BƯỚC 2: Khởi tạo mô hình MLPClassifier
        │
        │  model = MLPClassifier(alpha=0.01, hidden_layer_sizes=(300,), ...)
        │  → Đây chỉ là "bộ khung xương" chưa được huấn luyện
        │
        ▼
BƯỚC 3: Khởi tạo EmotionRecognizer
        │
        │  detector = EmotionRecognizer(
        │      model=model,
        │      emotions=["sad","neutral","happy","angry"],
        │      features=["mfcc","chroma","mel"]
        │  )
        │
        │  ⚙️ Bên trong __init__():
        │      ├─ Kiểm tra nhãn cảm xúc hợp lệ
        │      ├─ Gọi write_csv() → sinh metadata/*.csv
        │      └─ Gán self.model = model
        │
        ▼
BƯỚC 4: Huấn luyện mô hình
        │
        │  detector.train()
        │
        │  ⚙️ Bên trong train():
        │      ├─ Gọi load_data() nếu chưa có dữ liệu
        │      │     └─ Đọc CSV → nạp WAV → trích xuất đặc trưng
        │      └─ model.fit(X_train, y_train)
        │          (khoảng 1-5 phút lần đầu)
        │
        ▼
BƯỚC 5: Đánh giá mô hình
        │
        │  detector.test_score() → ví dụ: 0.8750 = 87.5%
        │
        ▼
BƯỚC 6: Dự đoán cảm xúc file WAV đã chọn
        │
        │  result = detector.predict("data/emodb/wav/03a04Wc.wav")
        │
        │  ⚙️ Bên trong predict():
        │      ├─ Đọc file WAV
        │      ├─ Trích xuất vector 180D
        │      └─ model.predict(vector) → "angry"
        │
        ▼
OUTPUT: "Result: angry"
```

---

## 4. Luồng 2 — Ghi âm realtime qua CLI (`test.py`)

**Dùng khi:** Bạn muốn nói chuyện với máy tính qua terminal.

### Chạy lệnh:
```bash
# Mặc định: 3 cảm xúc, BaggingClassifier
python test.py

# Tuỳ chỉnh:
python test.py --emotions "sad,neutral,happy,angry" --model "MLPClassifier"
```

### Từng bước xảy ra:

```
╔══════════════════════════════════════════════════════╗
║         LUỒNG CHẠY test.py                          ║
╚══════════════════════════════════════════════════════╝

BƯỚC 1: Phân tích tham số dòng lệnh (argparse)
        │
        │  -e / --emotions  → ["sad", "neutral", "happy"]
        │  -m / --model     → "BaggingClassifier"
        │
        ▼
BƯỚC 2: Nạp mô hình tốt nhất từ grid/
        │
        │  estimators = get_best_estimators(classification=True)
        │  → Đọc file grid/best_classifiers.pickle
        │  → Nếu không có: dùng BaggingClassifier mặc định
        │
        ▼
BƯỚC 3: Khởi tạo + Huấn luyện EmotionRecognizer
        │
        │  detector = EmotionRecognizer(model, emotions=..., features=...)
        │  detector.train()
        │  → In ra: "Test accuracy score: 87.50%"
        │
        ▼
BƯỚC 4: Chờ người dùng nói
        │
        │  print("Please talk")
        │  record_to_file("test.wav")
        │
        │  ⚙️ Bên trong record():
        │      ├─ Mở kết nối PyAudio (microphone)
        │      ├─ Đọc từng chunk âm thanh 1024 mẫu
        │      ├─ Phát hiện: có tiếng nói chưa? (so với ngưỡng 500)
        │      ├─ Khi phát hiện im lặng kéo dài 30 chunk → dừng
        │      ├─ normalize() → cân bằng âm lượng
        │      ├─ trim()      → cắt khoảng lặng đầu/cuối
        │      └─ add_silence() → thêm 0.5s yên lặng cả hai đầu
        │
        ▼
BƯỚC 5: Lưu file WAV tạm
        │
        │  Ghi vào "test.wav" (16000Hz, 1 kênh, PCM 16-bit)
        │
        ▼
BƯỚC 6: Dự đoán cảm xúc
        │
        │  result = detector.predict("test.wav")
        │
        ▼
OUTPUT: In ra màn hình: "happy" / "sad" / "angry"...
```

---

## 5. Luồng 3 — Ứng dụng GUI (`gui_test.py`)

**Dùng khi:** Bạn muốn giao diện đồ hoạ trực quan, hiện sóng âm.

### Chạy lệnh:
```bash
python gui_test.py
```

### Từng bước xảy ra:

```
╔══════════════════════════════════════════════════════════════════╗
║         LUỒNG CHẠY gui_test.py                                  ║
╚══════════════════════════════════════════════════════════════════╝

KHỞI ĐỘNG:
    root = tk.Tk()
    app = EmotionGUI(root)
    root.mainloop()
    │
    ▼
GIAI ĐOẠN 1: Xây dựng giao diện (setup_ui)
    │
    ├─ Tạo cửa sổ 1000×650px, nền tối #0f172a
    ├─ Panel trái: nút ghi âm, nhãn trạng thái, kết quả
    └─ Panel phải: biểu đồ Matplotlib (waveform)
    │
    ▼
GIAI ĐOẠN 2: Khởi tạo mô hình trong luồng nền (daemon thread)
    │
    │  threading.Thread(target=task, daemon=True).start()
    │
    │  ⚠️ Tại sao dùng thread? → Nếu huấn luyện trên main thread,
    │     giao diện sẽ bị đóng băng (freeze) trong 30-60 giây.
    │     Dùng thread nền → giao diện vẫn phản hồi bình thường.
    │
    │  ⚙️ Bên trong task():
    │      model = RandomForestClassifier(n_estimators=100)
    │      self.detector = EmotionRecognizer(model=model,
    │          emotions=["sad","neutral","happy","angry","fear"])
    │      self.detector.train()   ← ~30-60 giây
    │      → Báo về main thread: on_detector_ready()
    │
    ▼
GIAI ĐOẠN 3: Mô hình sẵn sàng
    │
    │  Nút "Ghi âm" chuyển từ DISABLED → ENABLED
    │  Màu nút: xám → tím (#6366f1)
    │  Trạng thái: "Hệ thống đã sẵn sàng!"
    │
    ▼
GIAI ĐOẠN 4: Người dùng nhấn "Bắt đầu Ghi âm"
    │
    │  threading.Thread(target=record_and_predict, daemon=True).start()
    │  → Lại dùng thread nền để GUI không bị đóng băng lúc ghi âm
    │
    │  ⚙️ Bên trong record_and_predict():
    │      sample_width, data = self.record()
    │      self.save_wave("gui_test.wav", sample_width, data)
    │      result = self.detector.predict("gui_test.wav")
    │      self.root.after(0, lambda: self.update_result(result, data))
    │      └─ root.after() → gửi kết quả về main thread một cách an toàn
    │
    ▼
GIAI ĐOẠN 5: Cập nhật giao diện (update_result)
    │
    │  self.result_label.config(text="HAPPY")  ← hiện to, màu xanh lá
    │  self.ax.plot(data, ...)                 ← vẽ sóng âm
    │  self.canvas.draw()                      ← refresh biểu đồ
    │
    ▼
Trở về GIAI ĐOẠN 3 (sẵn sàng ghi âm lần tiếp theo)
```

---

## 6. Luồng cốt lõi — `EmotionRecognizer` hoạt động như nào?

`EmotionRecognizer` là "trái tim" của hệ thống. Mọi luồng trên đều sử dụng nó.

```
╔══════════════════════════════════════════════════════════════════╗
║      VÒNG ĐỜI CỦA EmotionRecognizer                            ║
╚══════════════════════════════════════════════════════════════════╝

  EmotionRecognizer(model, emotions, features)
         │
         ▼ __init__()
  ┌─────────────────────────────────────────────┐
  │  1. _verify_emotions()                      │
  │     → Kiểm tra xem nhãn trong "emotions"    │
  │       có nằm trong AVAILABLE_EMOTIONS không │
  │                                             │
  │  2. get_audio_config(features)              │
  │     → Chuyển ["mfcc","chroma","mel"] thành  │
  │       {"mfcc":True, "chroma":True, "mel":   │
  │        True, "contrast":False, ...}         │
  │                                             │
  │  3. _set_metadata_filenames()               │
  │     → Xác định sẽ đọc file CSV nào          │
  │       train_tess_ravdess.csv, train_emodb... │
  │                                             │
  │  4. write_csv()                             │
  │     → Sinh file CSV nếu chưa có             │
  │                                             │
  │  5. self.model = model                      │
  │     (hoặc determine_best_model())           │
  └─────────────────────────────────────────────┘
         │
         ▼ .train()
  ┌─────────────────────────────────────────────┐
  │  1. load_data() → X_train, X_test,          │
  │                   y_train, y_test           │
  │  2. model.fit(X_train, y_train)             │
  └─────────────────────────────────────────────┘
         │
         ▼ .predict(audio_path)
  ┌─────────────────────────────────────────────┐
  │  1. extract_feature(audio_path)             │
  │     → vector 180D                           │
  │  2. model.predict(vector.reshape(1,-1))     │
  │     → "happy"                               │
  └─────────────────────────────────────────────┘
```

---

## 7. Chi tiết bước 1 — Sinh file CSV metadata

**File:** `core/create_csv.py`

### Tại sao cần file CSV?

Thay vì "nhớ" từng file WAV ở đâu, hệ thống lưu tất cả đường dẫn + nhãn vào file CSV. Như một cuốn danh sách điểm danh.

### Cách script quét thư mục và trích xuất nhãn:

```
Với TESS/RAVDESS (data/training/Actor_*/):
─────────────────────────────────────────
Tên file: "03-02-05-01-01-01-01_angry.wav"
                                 ↑
                          Lấy phần sau dấu "_" → "angry"

Tên file: "25_01_01_01_far_sad.wav"
                             ↑
                      Lấy phần sau dấu "_" cuối → "sad"

Với EMO-DB (data/emodb/wav/):
──────────────────────────────
Tên file: "03a04Wc.wav"
               ↑
          Ký tự thứ 5 → "W"
Ánh xạ: W→angry, N→neutral, F→happy, T→sad, A→fear, E→disgust, L→boredom

→ Kết quả ghi vào CSV:
   ,path,emotion
   0,data/emodb/wav/03a04Wc.wav,angry
```

### Output (các file CSV được tạo):

```
metadata/
├── train_tess_ravdess.csv  ← ~3.833 dòng
├── test_tess_ravdess.csv   ← ~600 dòng
├── train_emodb.csv         ← 327 dòng
├── test_emodb.csv          ← ~80 dòng
├── train_custom.csv        ← 99 dòng (nếu có)
└── test_custom.csv         ← ~50 dòng (nếu có)
```

---

## 8. Chi tiết bước 2 — Nạp dữ liệu và trích xuất đặc trưng

**File:** `core/data_extractor.py` + `core/utils.py`

### Bước này tốn nhiều thời gian nhất (5–15 phút lần đầu)

```
╔══════════════════════════════════════════════════════════════════╗
║      QUÁ TRÌNH TRÍCH XUẤT ĐẶC TRƯNG (extract_feature)          ║
╚══════════════════════════════════════════════════════════════════╝

1. ĐỌC FILE WAV
   ─────────────
   soundfile.SoundFile("data/training/Actor_01/01_01_02_01_kids-talking_happy.wav")
         │
         ▼
   X = [0.0012, -0.0034, 0.0056, ...]  ← mảng số float (sóng âm thô)
   sample_rate = 22050 Hz               ← 22050 mẫu mỗi giây

2. TRÍCH XUẤT MFCC (40 số)
   ─────────────────────────
   librosa.feature.mfcc(y=X, sr=22050, n_mfcc=40)
         │
         ▼
   Ma trận (40, T) → Lấy trung bình theo thời gian → vector 40D
   
   💡 MFCC là gì?
   MFCC = cách máy tính "mô tả hình dạng" của giọng nói.
   Giống như bạn mô tả một người: "cao 1m70, nặng 60kg, tóc đen..."
   MFCC mô tả âm thanh theo 40 con số đặc trưng.

3. TRÍCH XUẤT CHROMA (12 số)
   ─────────────────────────
   stft = np.abs(librosa.stft(y=X))
   librosa.feature.chroma_stft(S=stft, sr=22050)
         │
         ▼
   Ma trận (12, T) → Lấy trung bình → vector 12D
   
   💡 Chroma là gì?
   Biểu diễn 12 nốt nhạc (C, C#, D, D#, E, F, F#, G, G#, A, A#, B).
   Nắm bắt cảm giác "lên xuống" giọng điệu — người buồn nói giọng
   thấp hơn người vui.

4. TRÍCH XUẤT MEL SPECTROGRAM (128 số)
   ─────────────────────────────────────
   librosa.feature.melspectrogram(y=X, sr=22050)
         │
         ▼
   Ma trận (128, T) → Lấy trung bình → vector 128D
   
   💡 Mel Spectrogram là gì?
   Bản "bản đồ nhiệt" của âm thanh — tần số nào có năng lượng cao?
   Người tức giận nói to hơn, có nhiều năng lượng ở tần số cao.

5. GHÉP NỐI (CONCATENATE)
   ─────────────────────
   result = [mfcc_40D] + [chroma_12D] + [mel_128D]
          = vector 180 chiều
   
   [0.23, -1.4, ...(40 số)..., 0.12, 0.09, ...(12 số)..., 0.001, ...(128 số)...]
    ←─────── MFCC ──────────→ ←── Chroma ──→ ←────── Mel Spectrogram ────────→
```

### Kết quả cuối cùng của load_data():

```python
X_train.shape = (3000, 180)  # 3000 mẫu train, mỗi mẫu 180 số
X_test.shape  = (500, 180)   # 500 mẫu test
y_train = ["happy", "sad", "neutral", "angry", ...]  # nhãn tương ứng
y_test  = ["fear", "happy", "sad", ...]
```

---

## 9. Chi tiết bước 3 — Huấn luyện mô hình

**File:** `core/emotion_recognition.py` → `EmotionRecognizer.train()`

```
╔══════════════════════════════════════════════════════════════════╗
║      QUÁ TRÌNH HUẤN LUYỆN MODEL                                ║
╚══════════════════════════════════════════════════════════════════╝

model.fit(X_train, y_train)
     │
     │  Ví dụ với RandomForestClassifier:
     │  → Xây dựng 100 cây quyết định
     │  → Mỗi cây học từ một tập con ngẫu nhiên của dữ liệu
     │  → Khi dự đoán: 100 cây "bỏ phiếu" → lấy kết quả đa số
     │
     │  Với MLPClassifier (Mạng nơ-ron):
     │  → Xây dựng mạng: 180 → 300 → N_emotions
     │  → Học qua nhiều vòng (epoch) bằng gradient descent
     │
     │  Với SVC (Support Vector Machine):
     │  → Tìm siêu phẳng chia không gian 180 chiều
     │
     ▼
model.model_trained = True   ← Đánh dấu đã train xong
```

### So sánh các mô hình:

| Mô hình | Thời gian train | Độ chính xác (3 cảm xúc) | Tốc độ dự đoán |
|---|---|---|---|
| **MLPClassifier** ⭐ | 1–5 phút | ~89.6% | Nhanh |
| RandomForestClassifier | 30 giây | ~93.5% | Nhanh |
| BaggingClassifier | 1 phút | ~85% | Nhanh |
| SVC | 2–5 phút | ~81.5% | Chậm hơn |
| GradientBoostingClassifier | 3–8 phút | ~87% | Trung bình |

> 💡 **Gợi ý cho người mới:** Dùng `RandomForestClassifier` để có kết quả tốt + train nhanh.

---

## 10. Chi tiết bước 4 — Dự đoán cảm xúc từ file WAV

```
╔══════════════════════════════════════════════════════════════════╗
║      QUÁ TRÌNH DỰ ĐOÁN (predict)                               ║
╚══════════════════════════════════════════════════════════════════╝

detector.predict("gui_test.wav")
         │
         ▼
1. Đọc file WAV → mảng số float
         │
         ▼
2. extract_feature("gui_test.wav") → vector 180D
   [0.15, -0.43, ...(40 MFCC)..., 0.21, ...(12 Chroma)..., 0.003, ...(128 Mel)...]
         │
         ▼
3. vector.reshape(1, -1) → shape (1, 180)
   (mô hình cần input là ma trận 2D, dù chỉ có 1 mẫu)
         │
         ▼
4. model.predict([[0.15, -0.43, ...]])
         │
         ▼
5. Trả về: "happy"
```

### Trực quan hoá dự đoán (predict_proba):

```python
detector.predict_proba("gui_test.wav")
# → {'angry': 0.05, 'fear': 0.02, 'happy': 0.85, 'neutral': 0.07, 'sad': 0.01}
#                                            ↑
#                                  Mô hình 85% chắc chắn là "happy"
```

---

## 11. Luồng ghi âm — Microphone hoạt động như nào?

Đây là phần "ma thuật" dễ bị hiểu nhầm nhất. Hệ thống ghi âm **tự động dừng khi phát hiện yên lặng**.

```
╔══════════════════════════════════════════════════════════════════╗
║      LUỒNG GHI ÂM (record function)                            ║
╚══════════════════════════════════════════════════════════════════╝

KHỞI TẠO:
   PyAudio → mở stream từ microphone
   Format: PCM 16-bit
   Channels: 1 (mono)
   Rate: 16000 Hz (16000 mẫu/giây)
   Chunk: 1024 mẫu/lần đọc
          │
          ▼
    ┌──────────────────────────────────────────────────────┐
    │              VÒNG LẶP ĐỌC CHUNK                     │
    │                                                      │
    │  while True:                                         │
    │      chunk = stream.read(1024)  ← đọc 1/15.6 giây   │
    │      max_amplitude = max(abs(chunk))                 │
    │                                                      │
    │      Nếu max_amplitude < 500 (THRESHOLD):            │
    │          → Im lặng                                   │
    │          Nếu đang nói: num_silent += 1               │
    │                                                      │
    │      Nếu max_amplitude >= 500:                       │
    │          → Có tiếng nói                              │
    │          snd_started = True                          │
    │          num_silent = 0                              │
    │                                                      │
    │      Nếu snd_started AND num_silent > 30:            │
    │          → Im lặng 30 chunk = ~2 giây → DỪNG LẠI    │
    │          break                                       │
    └──────────────────────────────────────────────────────┘
          │
          ▼
XỬ LÝ SAU GHI:
   normalize()     → cân bằng âm lượng (tránh quá to/quá nhỏ)
   trim()          → cắt khoảng yên lặng đầu và cuối
   add_silence()   → thêm 0.5s yên lặng cả hai đầu
          │
          ▼
XUẤT FILE: test.wav / gui_test.wav
   16000 Hz, mono, PCM 16-bit

═══════════════════════════════════════════════════════════
💡 MINH HOẠ TRỰC QUAN:

Thời gian → 
─────────────────────────────────────────────────────────
Yên lặng │ /\/\/\/\/\/\/\/\│ Yên lặng ─── ─── ─── ─── │
  (dưới   │  (tiếng nói)   │  (đếm 30 chunk im lặng)   │
  ngưỡng) │                │                             │ DỪNG
─────────────────────────────────────────────────────────
          ↑                ↑                             ↑
     snd_started=True  num_silent++                    break
═══════════════════════════════════════════════════════════
```

---

## 12. Hệ thống cache — Tại sao lần 2 lại nhanh hơn?

Trích xuất đặc trưng từ ~4000 file WAV tốn **5–15 phút**. Nếu phải làm lại mỗi lần chạy thì quá lâu. Hệ thống có cơ chế **cache** để giải quyết vấn đề này.

```
╔══════════════════════════════════════════════════════════════════╗
║      CƠ CHẾ CACHE (.npy files)                                 ║
╚══════════════════════════════════════════════════════════════════╝

LẦN ĐẦU CHẠY:
──────────────
   features/*.npy chưa tồn tại
         │
         ▼
   Trích xuất đặc trưng từ ~4000 file WAV (5-15 phút)
         │
         ▼
   Lưu vào file cache:
   features/train_mfcc-chroma-mel_ANH.npy    ← X_train (ma trận đặc trưng)
   features/train_mfcc-chroma-mel_ANH_labels.npy  ← y_train (nhãn)
   features/test_mfcc-chroma-mel_ANH.npy
   features/test_mfcc-chroma-mel_ANH_labels.npy
         │
         ▼
   Tiếp tục huấn luyện...

LẦN SAU CHẠY:
──────────────
   features/*.npy đã tồn tại
         │
         ▼
   numpy.load("features/train_mfcc-chroma-mel_ANH.npy")  < 5 giây!
         │
         ▼
   Bỏ qua toàn bộ bước trích xuất → Huấn luyện ngay lập tức

═══════════════════════════════════════════════════════════
💡 TÊN FILE CACHE được xây dựng từ:
   - Tập train/test
   - Các đặc trưng đang dùng: mfcc-chroma-mel
   - Chữ cái đầu của các cảm xúc: A=Angry, N=Neutral, H=Happy...
   
Khi bạn THAY ĐỔI emotions hoặc features → tên file cache khác
→ Tự động trích xuất lại từ đầu (đúng!)
═══════════════════════════════════════════════════════════

⚠️ KHI NÀO XOÁ CACHE?
   Xoá features/*.npy khi:
   ✗ Bạn thêm file WAV mới vào thư mục data/
   ✗ Bạn thay đổi cấu hình audio (sample rate, mono/stereo...)
   ✗ Cache cũ bị lỗi
   
   Lệnh xoá:
   del features\*.npy          (Windows)
   rm features/*.npy           (Linux/Mac)
```

---

## 13. Các mô hình ML được dùng

### Phân loại (Classification)

```
ĐẦU VÀO: vector 180 số (đặc trưng âm thanh)
ĐẦU RA:  nhãn cảm xúc (string: "happy", "sad"...)
```

#### RandomForestClassifier — Rừng ngẫu nhiên
```
180D ──► Cây 1 → "happy"  ┐
         Cây 2 → "happy"  │── Bỏ phiếu → "happy" (70%)
         Cây 3 → "sad"    │
         ...100 cây...    ┘
```

#### MLPClassifier — Mạng nơ-ron nhiều lớp
```
Input(180) → Hidden(300) → Output(N cảm xúc)
   ↑               ↑              ↑
 180 nơ-ron    300 nơ-ron    N nơ-ron (softmax)
```

#### SVC — Support Vector Machine
```
Tìm đường thẳng (siêu phẳng) trong không gian 180 chiều
để chia các điểm dữ liệu thành các nhóm cảm xúc
```

#### BaggingClassifier — Tổng hợp nhiều mô hình
```
Tương tự RandomForest nhưng có thể dùng bất kỳ mô hình cơ sở nào
```

### Deep Learning (DeepEmotionRecognizer)

```
Input (180D) → LSTM(128) → LSTM(128) → Dense(128) → Dense(128) → Softmax(N)
                 ↑
           Nhớ thứ tự thời gian trong chuỗi âm thanh
```

---

## 14. Câu hỏi thường gặp của người mới

### ❓ "Lần đầu chạy thấy rất chậm, có bình thường không?"
> **Bình thường hoàn toàn.** Lần đầu phải trích xuất đặc trưng từ ~4000 file WAV. Mất 5–15 phút. Từ lần 2 trở đi sẽ < 5 giây nhờ cache.

---

### ❓ "Tại sao dự đoán sai?"
> Một số nguyên nhân phổ biến:
> - Chất lượng microphone kém / nhiều tiếng ồn xung quanh
> - Giọng nói không đủ rõ ràng để phát hiện (dưới ngưỡng 500)
> - Dataset huấn luyện là tiếng Anh → dự đoán giọng Việt có thể kém chính xác hơn
> - Số lượng cảm xúc quá nhiều (9 cảm xúc khó hơn nhiều so với 3)

---

### ❓ "MFCC, Chroma, Mel là gì? Tại sao lại dùng 3 cái này?"
> 3 đặc trưng này bổ sung cho nhau:
> - **MFCC (40D):** nắm bắt "hình dạng" tổng thể của giọng nói (content)
> - **Chroma (12D):** nắm bắt cao độ, điệu (pitch) — người buồn nói giọng thấp hơn
> - **Mel Spec (128D):** nắm bắt phân bố năng lượng theo tần số — người tức giận nói to hơn

---

### ❓ "Thêm dữ liệu của tôi vào như nào?"
```
1. Đặt tên file: {bất_kỳ}_{cảm_xúc}.wav
   Ví dụ: myvoice_happy.wav, recording001_sad.wav

2. Copy vào: data/train-custom/

3. Khi chạy, bật tuỳ chọn custom_db=True:
   detector = EmotionRecognizer(model, custom_db=True, emotions=[...])

4. Xoá cache cũ (nếu có):
   del features\*.npy
```

---

### ❓ "File pickle trong thư mục grid/ là gì?"
> Kết quả sau khi chạy `python tools/grid_search.py` (mất 2–8 giờ!). Chứa các mô hình đã được tối ưu tham số sẵn (best hyperparameters). Khi gọi `determine_best_model()`, hệ thống đọc file này thay vì Grid Search lại từ đầu.

---

### ❓ "Sơ đồ module phụ thuộc nhau như thế nào?"

```
gui_test.py ──────────────────────────┐
test.py ──────────────────────────────┤
demo_predict.py ──────────────────────┤
                                      ▼
                          core/emotion_recognition.py
                                    │
               ┌────────────────────┼────────────────────┐
               ▼                    ▼                    ▼
       core/create_csv.py   core/data_extractor.py   core/utils.py
               │                    │                    │
               ├── os.walk()        ├── pandas           ├── librosa
               ├── glob             ├── numpy            ├── soundfile
               └─→ metadata/*.csv   └─→ features/*.npy   └─→ extract_feature()
```

---

### ❓ "Tổng kết: Chạy dự án lần đầu theo thứ tự nào?"

```
╔══════════════════════════════════════════════════════════════╗
║        THỨ TỰ KHỞI ĐỘNG NHANH (Quick Start)                ║
╚══════════════════════════════════════════════════════════════╝

Bước 1: Cài thư viện
        pip install -r requirements.txt

Bước 2: Kiểm tra dữ liệu có đầy đủ chưa
        Kiểm tra thư mục data/training/ có chứa Actor_01...Actor_26 không
        Kiểm tra data/emodb/wav/ có chứa file .wav không

Bước 3: Thử demo không cần mic (kiểm tra pipeline)
        python demo_predict.py
        → Kết quả: "Result: angry" (hoặc nhãn khác)

Bước 4: Thử ghi âm qua terminal
        python test.py
        → Đợi "Please talk", nói câu gì đó, chờ kết quả

Bước 5: Dùng giao diện GUI
        python gui_test.py
        → Đợi "Hệ thống đã sẵn sàng!", nhấn nút ghi âm

════════════════════════════════════════════════════════════════
💡 MẸO: Lần đầu chạy bước 3 sẽ tạo cache. Bước 4 và 5 sẽ NHANH hơn nhiều!
```

---

## Tài liệu liên quan

| File | Đọc khi |
|---|---|
| [`README.md`](README.md) | Muốn cài đặt nhanh và xem ví dụ code |
| [`DATA_ANALYSIS.md`](DATA_ANALYSIS.md) | Muốn hiểu chi tiết cấu trúc dữ liệu |
| [`PROJECT_GUIDE.md`](PROJECT_GUIDE.md) | Muốn hướng dẫn vận hành đầy đủ + sơ đồ |

---

*Tài liệu được viết cho người mới học Machine Learning.*  
*Cập nhật lần cuối: 2026-04-09*
