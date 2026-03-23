# 🎤 Hướng dẫn chi tiết Dự án Nhận dạng Cảm xúc qua Giọng nói

Chào mừng bạn đến với dự án **Speech Emotion Recognition**! Đây là tài liệu hướng dẫn từ A-Z dành cho người mới bắt đầu học lập trình âm thanh và trí tuệ nhân tạo.

---

## 🛠 1. Công nghệ & Thư viện sử dụng

Dự án sử dụng các thư viện Python mạnh mẽ sau:

- **Xử lý âm thanh**:
    - `librosa`: Dùng để trích xuất đặc trưng âm thanh tinh vi (MFCC, Chroma, Mel).
    - `pyaudio`: Kết nối với Microphone để thu âm trực tiếp.
    - `soundfile`: Đọc/ghi các file âm thanh định dạng `.wav`.
- **Học máy (Machine Learning)**:
    - `scikit-learn`: Cung cấp các thuật toán chính như **Random Forest**, **SVM**, **MLP**.
- **Giao diện & Trực quan hóa**:
    - `tkinter`: Tạo cửa sổ giao diện người dùng (GUI).
    - `matplotlib`: Vẽ biểu đồ sóng âm (Waveform) và biểu đồ phân bổ dữ liệu.

---

## 📚 2. Các thuật ngữ "Lập trình âm thanh" cần biết

Dự án này không chỉ "nghe" âm thanh, mà nó chuyển âm thanh thành các thông số kỹ thuật:
1.  **MFCC (Mel-frequency cepstral coefficients)**: Giống như "vân tay" của giọng nói, giúp máy phân biệt các đặc điểm âm sắc của từng cảm xúc.
2.  **Chroma**: Đại diện cho các cao độ trong âm nhạc, giúp nhận diện tông giọng.
3.  **Mel Spectrogram**: Biểu thị năng lượng của âm thanh theo cách tai người cảm âm.

---

## 🚀 3. Các bước chạy dự án (Thứ tự ưu tiên)

### Bước 1: Cài đặt toàn bộ thư viện
Mở Terminal/PowerShell tại thư mục dự án và chạy:
```bash
python -m pip install -r requirements.txt
```
*Lưu ý: Nếu bị lỗi `pyaudio`, hãy sử dụng `pip install pipwin` rồi `pipwin install pyaudio`.*

### Bước 2: Dọn dẹp mô hình cũ (Quan trọng)
Do các mô hình đi kèm dự án được huấn luyện ở phiên bản cũ không tương thích, hãy chạy script dọn dẹp tôi đã tạo:
```bash
python clean_models.py
```

### Bước 3: Chạy giao diện GUI (Khuyên dùng cho người mới)
Đây là cách dễ dàng nhất để bạn trải nghiệm dự án một cách trực quan:
```bash
python gui_test.py
```
- Click **Ghi âm**, nói một câu thoải mái.
- Hệ thống sẽ hiển thị **Sóng âm (Waveform)** và dự đoán cảm xúc của bạn ngay lập tức!

### Bước 4: Phân tích dữ liệu (Dành cho báo cáo/bảo vệ)
Để xem tập dữ liệu bạn đang dùng "mạnh" hay "yếu" ở đâu:
```bash
python analyze_dataset.py
```
Nó sẽ tạo ra file `dataset_distribution.png` hiển thị số lượng mẫu của từng cảm xúc.

---

## 📁 4. Giải thích cấu trúc file
- `gui_test.py`: Giao diện người dùng hiện đại với biểu đồ (Tôi đã tạo thêm).
- `analyze_dataset.py`: Công cụ phân tích dữ liệu (Tôi đã tạo thêm).
- `emotion_recognition.py`: Chứa "linh hồn" của dự án - các hàm xử lý và huấn luyện ML.
- `data/`: Chứa các bộ dữ liệu nổi tiếng thế giới (RAVDESS, TESS, EMO-DB).

---

## 💡 5. Mẹo để đạt điểm cao khi bảo vệ dự án
1.  **Chỉ ra sự mất cân bằng**: Sử dụng hình ảnh từ `analyze_dataset.py` để thuyết trình rằng tập dữ liệu có lớp nhiều, lớp ít và bạn đã nhận diện được điều đó.
2.  **Giải thích về đặc trưng**: MFCC là từ khóa quan trọng nhất, hãy nhớ nó khi được hỏi về cách máy tính "nghe".
3.  **Trình diễn bằng GUI**: Sử dụng bản `gui_test.py` sẽ gây ấn tượng mạnh hơn nhiều so với việc chạy dòng lệnh đen trắng.

**Chúc bạn học tập và hoàn thành dự án xuất sắc!**
