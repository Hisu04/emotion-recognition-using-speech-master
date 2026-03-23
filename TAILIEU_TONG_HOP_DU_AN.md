# 🏆 TÀI LIỆU TỔNG HỢP CHI TIẾT: DỰ ÁN NHẬN DẠNG CẢM XÚC QUA GIỌNG NÓI (SPEECH EMOTION RECOGNITION)

Tài liệu này tổng hợp toàn bộ những thay đổi, nâng cấp, các công nghệ đã sử dụng và hướng dẫn cài đặt/vận hành chi tiết nhất cho dự án nhận dạng cảm xúc qua giọng nói. Đây là cẩm nang toàn diện giúp bạn dễ dàng nắm bắt mã nguồn, hiểu rõ kiến trúc, và tự tin bảo vệ dự án.

---

## 🌟 1. NHỮNG THAY ĐỔI VÀ NÂNG CẤP ĐÃ THỰC HIỆN TRÊN DỰ ÁN

Dự án gốc đã được tinh chỉnh, sửa lỗi và nâng cấp đáng kể để hoạt động trơn tru trên môi trường Python và thư viện mới nhất. 

**Các nâng cấp chính bao gồm:**
1. **Sửa lỗi không tương thích phiên bản thư viện (Dependency Fixes):**
   - Đã khắc phục triệt để lỗi `InconsistentVersionWarning` và các lỗi import `sklearn` do khác biệt phiên bản `scikit-learn` giữa lúc huấn luyện mô hình cũ và quá trình chạy hiện tại.
   - Sửa lỗi tương thích của thư viện `librosa` và `numpy` (ví dụ: lỗi `np.float`).
   - Sửa lỗi `ImportError` cho module `data_extractor` và `sklearn.metrics` cấu trúc lại luồng import.
2. **Quản lý mô hình an toàn (`clean_models.py`):**
   - Tạo mới script `clean_models.py` giúp tự động quét và xóa các file mô hình pre-trained cũ (`.pickle`), buộc hệ thống phải huấn luyện lại trên dữ liệu và phiên bản thư viện mới nhất của bạn (giúp độ chính xác và tính ổn định cao hơn, không crash).
3. **Bổ sung Giao diện Đồ họa Trực quan (`gui_test.py`):**
   - Thay vì chỉ thao tác trên Terminal/Command Prompt khô khan, dự án đã được bổ sung một app giao diện hoàn chỉnh bằng `tkinter`.
   - **Tính năng GUI:** Nhấn nút để ghi âm trực tiếp, hiển thị biểu đồ sóng âm (waveform) sống động bằng `matplotlib`, và in ra kết quả dự đoán cảm xúc ngay lập tức sau khi dừng ghi âm. Thay thế sự phụ thuộc kém ổn định của `pyaudio` bằng `sounddevice` và `soundfile` trên Windows.
4. **Công cụ Phân tích Dữ liệu trực quan (`analyze_dataset.py`):**
   - Viết thêm script để tự động thống kê số mẫu của từng loại cảm xúc (Happy, Sad, Angry, v.v.).
   - Tạo ra biểu đồ trực quan (`dataset_distribution.png`) giúp bạn đánh giá được độ "lệch" của dữ liệu (Class Imbalance) - một luận điểm tuyệt vời để ăn điểm khi bảo vệ đồ án.
5. **Cải tiến đánh giá mô hình (Confusion Matrix):**
   - Đã thêm khả năng sinh ra bản đồ nhiệt (Heatmap) của Text/Confusion Matrix, giúp nhìn rõ mô hình hay bị nhầm lẫn giữa cặp cảm xúc nào (ví dụ: mô hình hay nhầm *fearful* với *sad*).
6. **Công cụ dự đoán nhanh bằng dòng lệnh (`demo_predict.py`):**
   - Viết thêm luồng chạy nhanh gọn lẹ: Đưa thẳng file ghi âm vào công cụ này để có ngay cảm xúc đầu ra.
7. **Hỗ trợ Deep Learning (`deep_emotion_recognition.py`):**
   - Khắc phục các lỗi về việc tải trọng số (weights) và tương thích luồng của `tensorflow`/`keras` giúp dự án có khả năng chạy thuật toán Deep Learning (mạng nơ ron) ngoài thuật toán máy học truyền thống.
8. **Hỗ trợ định dạng đầu ra tiếng Việt (UTF-8 console):**
   - Các script đã cấu hình để in các chuỗi Tiếng Việt, các ký hiệu cảm xúc (emoji) lên dòng lệnh Windows PowerShell không bị lỗi font (encoding utf-8).

---

## 🛠 2. CÁC CÔNG NGHỆ VÀ THƯ VIỆN SỬ DỤNG

Kiến trúc của dự án là sự kết hợp của hệ sinh thái Xử lý tín hiệu số (DSP) và Trí tuệ nhân tạo (AI) của Python:

### Ngôn ngữ lõi:
- **Python (>=3.8)**: Nền tảng chính.

### 2.1. Xử lý Trích xuất Đặc trưng Âm thanh (DSP)
- **`librosa`**: Thư viện mạnh mẽ nhất để phân tích âm thanh, được dùng để trích xuất 3 loại đặc trưng quan trọng:
  - *MFCC* (Mel-frequency cepstral coefficients): Phân tích đặc trưng âm sắc tiếng người.
  - *Chroma*: Phân tích cao độ.
  - *Mel Spectrogram*: Tần số tỷ lệ Mel mô phỏng thính giác người.
- **`soundfile` & `sounddevice`**: Đọc/Ghi/Thu âm các file audio `.wav`. Sự kết hợp rất ổn định để thay thế `pyaudio` khi cần thu thập buffer ghi âm thời gian thực trên Windows.

### 2.2. Học Máy Cơ Bản & Nâng Cao (Machine Learning)
- **`scikit-learn`**: 
  - Khai thác các mô hình phân loại: **MLPClassifier** (Mạng nơ-ron nhiều lớp cơ bản), **RandomForest**, **SVM**, **GridSearchCV** (để tối ưu siêu tham số).
  - Cung cấp tính năng đo lường: `accuracy_score`, `confusion_matrix`, `classification_report`.
- **`joblib` / `pickle`**: Lưu (save) các mô hình sau khi huấn luyện xong xuống ổ cứng, và tải (load) lên để sử dụng dự đoán (giúp không phải huấn luyện lại từ đầu mỗi lần chạy dự án).

### 2.3. Học Sâu (Deep Learning)
- **`TensorFlow` & `Keras`**: Được tích hợp bên trong file `deep_emotion_recognition.py` để xây dựng cấu trúc Mạng nơ-ron hồi quy (RNN/LSTM) hoặc mạng dày đặc (Dense) xử lý tốt dữ liệu dạng chuỗi liên tục như âm thanh.

### 2.4. Trực Quan Hóa (Visualization) & Giao Diện Người Dùng (GUI)
- **`tkinter`**: Module có sẵn của Python giúp xây dựng cửa sổ làm việc (GUI), vẽ các nút bấm Ghi âm / Dừng / Kết quả nhanh chóng.
- **`matplotlib` & `seaborn`**: 
  - Vẽ sóng âm thanh trực tiếp (Waveform).
  - Vẽ phân bổ dữ liệu (Bar chart) và Ma trận nhầm lẫn (Confusion Matrix Heatmap).
- **`pandas` & `numpy`**: 
  - Thao tác, ma trận hóa, gom nhóm và tính toán toán học tốc độ cao trên mảng dữ liệu đặc trưng.

---

## 🚀 3. HƯỚNG DẪN CÀI ĐẶT VÀ CHẠY DỰ ÁN CHI TIẾT

Đây là các bước chuẩn xác để khởi chạy môi trường dự án một cách an toàn nhất:

### Bước 1: Chuẩn bị Môi trường (Cài đặt Thư viện)
Mở Terminal/PowerShell hoặc Command Prompt tại tại thư mục gốc của dự án (`g:\CV\emotion-recognition-using-speech-master`), và gõ lệnh:
```bash
python -m pip install -r requirements.txt
```
*(Nếu cài đặt gói thư viện bị lỗi `pyaudio`, bạn có thể chạy tuần tự: `pip install pipwin` sau đó `pipwin install pyaudio`)*.

### Bước 2: Dọn dẹp lỗi Mô hình cũ (Rất Quan trọng)
Vì bạn vừa cài phiên bản `scikit-learn` mới, trong thư mục dự án (hoặc thư mục con `results/`) có thể vẫn chứa các mô hình mẫu từ năm cũ. Hãy bắt buộc chạy lệnh này để xóa chúng đi, hệ thống sẽ tự huấn luyện dòng code mới vào tập tin mới:
```bash
python clean_models.py
```

### Bước 3: Phân tích Tổng quan Dữ liệu (Để Báo cáo)
Chạy script phân tích để dự án tự động lướt qua toàn bộ file âm thanh huấn luyện (training audio) và thống kê:
```bash
python analyze_dataset.py
```
> **Kết quả:** Nó sẽ sinh ra file hình ảnh tên `dataset_distribution.png` trong máy. Hãy mở nó ra để xem số lượng dữ liệu, copy ảnh đó đưa vào báo cáo môn học của bạn!

### Bước 4: Chạy Đánh giá/Huấn luyện (Training & Testing)
Để máy tính tự đọc tệp, huấn luyện (Machine Learning) và báo cáo độ chính xác (Accuracy), ma trận phân loại:
```bash
python test.py
```
*(Quá trình này có thể tốn vài phút tùy số lượng âm thanh huấn luyện vì máy sẽ trích xuất MFCC cho từng file và huấn luyện `MLPClassifier`).*

### Bước 5: 🎤 Chạy Giao diện Ghi âm & Nhận dạng (Khuyên dùng Demo)
Đây là tập lệnh hoàn chỉnh và ấn tượng nhất để bạn demo trước thầy cô / đối tác:
```bash
python gui_test.py
```
**Cách sử dụng:**
1. Cửa sổ "Speech Emotion Recognition" sẽ hiện lên.
2. Bạn nhấn nút **"Record"** (Ghi âm) và nói vào Micro một câu nhấn nhá cảm xúc (vd: quát lớn, nhí nhảnh, buồn bã).
3. Đồ thị sóng âm (Waveform) sẽ được vẽ ngay lên màn hình.
4. Sau tầm 3 giây (hoặc tự thiết lập), máy tính tự trích xuất MFCC truyền vào mô hình và trả về Cảm xúc tương ứng, ví dụ: `Kết quả: Cảm xúc TỨC GIẬN 😡` hoặc `VUI VẺ 😄`.

### Bước 6: Thử nghiệm không giao diện (Tùy chọn)
Nếu bạn đã có sẵn 1 file âm thanh (ví dụ `test.wav`) và chỉ muốn gõ lệnh xem nó là cảm xúc gì:
```bash
python demo_predict.py
```
*(Bạn có thể vào file `demo_predict.py` ở dòng dưới cùng, sửa `"test.wav"` thành đường dẫn âm thanh của bạn).*

---

## 🎯 4. MẸO BẢO VỆ ĐỒ ÁN ĐỂ ĐẠT ĐIỂM 9-10

1. **Hiểu về MFCC:** Nếu bị hỏi *"Dự án dùng đặc trưng gì của âm thanh?"*, hãy trả lời dõng dạc là **MFCC**. Vì MFCC (Mel-frequency cepstral coefficients) được sinh ra để mô phỏng lại cách tai người nghe. Đó là "dấu vân tay" của giọng nói giúp máy học dễ phân biệt cảm xúc hơn so với việc cứ đưa sóng âm thanh thô vào máy.
2. **Chiến lược xử lý Dữ liệu mất cân bằng:** Hãy mở biểu đồ `dataset_distribution.png` lên, chỉ ra các cột không bằng phẳng: *"Em/Nhóm em nhận thấy số lượng file cảm xúc VUI nhiều hơn SỢ HÃI, nên nhóm đã tùy chỉnh mô hình tốt hơn, đánh giá chéo (Cross Validation) / GridSearchCV"*...
3. **Demo bằng GUI trực tiếp:** Thay vì biểu diễn mã code Console chán ngắt, hãy bật hộp máy `gui_test.py` lên, cắm micro và nói trực tiếp (Live Predict). Việc làm được Real-time Inference trên Desktop sẽ được cộng điểm sáng tạo rất lớn.

*Chúc bạn thành công rực rỡ với phần mềm này! Mọi thành phần đã được gỡ lỗi và tối ưu hóa tối đa trong mã nguồn.*
