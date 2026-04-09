# Phân tích Dữ liệu — Emotion Recognition Using Speech

> Tài liệu phân tích chi tiết toàn bộ dữ liệu âm thanh và metadata được sử dụng trong hệ thống nhận dạng cảm xúc qua giọng nói.

[![Datasets](https://img.shields.io/badge/Datasets-3%20nguồn%20chính-blue)](#)
[![Audio Files](https://img.shields.io/badge/File%20WAV-~5000%2B-green)](#)
[![Emotions](https://img.shields.io/badge/Cảm%20xúc-9%20nhãn-orange)](#)
[![Format](https://img.shields.io/badge/Format-CSV%20%2B%20WAV-purple)](#)

---

## Mục lục

1. [Tổng quan dữ liệu](#1-tổng-quan-dữ-liệu)
2. [Cấu trúc thư mục](#2-cấu-trúc-thư-mục)
3. [Thư mục `data/` — Dữ liệu âm thanh](#3-thư-mục-data--dữ-liệu-âm-thanh)
   - [RAVDESS — training/ & validation/](#31-ravdess--tess----training--validation)
   - [EMO-DB — emodb/wav/](#32-emo-db----emodбwav)
   - [Custom — train-custom/ & test-custom/](#33-custom----train-custom--test-custom)
4. [Thư mục `metadata/` — File CSV mô tả](#4-thư-mục-metadata--file-csv-mô-tả)
   - [Cấu trúc CSV](#41-cấu-trúc-chung-của-tất-cả-file-csv)
   - [train_tess_ravdess.csv](#42-train_tess_ravdescsv)
   - [test_tess_ravdess.csv](#43-test_tess_ravdessscsv)
   - [train_emodb.csv](#44-train_emodбcsv)
   - [test_emodb.csv](#45-test_emodbcsv)
   - [train_custom.csv](#46-train_customcsv)
   - [test_custom.csv](#47-test_customcsv)
5. [Quy ước đặt tên file](#5-quy-ước-đặt-tên-file)
   - [Quy ước TESS](#51-quy-ước-đặt-tên-tess)
   - [Quy ước RAVDESS](#52-quy-ước-đặt-tên-ravdess)
   - [Quy ước EMO-DB](#53-quy-ước-đặt-tên-emo-db)
   - [Quy ước Custom](#54-quy-ước-đặt-tên-custom)
6. [Phân phối nhãn cảm xúc](#6-phân-phối-nhãn-cảm-xúc)
7. [Luồng xử lý dữ liệu](#7-luồng-xử-lý-dữ-liệu)
8. [Cơ chế sinh metadata — create_csv.py](#8-cơ-chế-sinh-metadata--create_csvpy)
9. [Trích xuất đặc trưng — AudioExtractor](#9-trích-xuất-đặc-trưng--audioextractor)
10. [Tóm tắt số liệu](#10-tóm-tắt-số-liệu)

---

## 1. Tổng quan dữ liệu

Dự án tổng hợp **3 bộ dữ liệu chuẩn quốc tế** và **1 bộ dữ liệu tuỳ chỉnh** để huấn luyện mô hình nhận dạng cảm xúc:

| Nguồn dữ liệu | Thư mục âm thanh | File metadata | Ngôn ngữ | Người nói |
|---|---|---|---|---|
| **RAVDESS** | `data/training/`, `data/validation/` | `train_tess_ravdess.csv`, `test_tess_ravdess.csv` | Tiếng Anh | 24 diễn viên |
| **TESS** | `data/training/`, `data/validation/` | `train_tess_ravdess.csv`, `test_tess_ravdess.csv` | Tiếng Anh | 2 nữ diễn viên |
| **EMO-DB** | `data/emodb/wav/` | `train_emodb.csv`, `test_emodb.csv` | Tiếng Đức | 10 người nói |
| **Custom** | `data/train-custom/`, `data/test-custom/` | `train_custom.csv`, `test_custom.csv` | Tuỳ ý | Người dùng cung cấp |

> **Lưu ý:** RAVDESS và TESS được lưu chung trong cùng thư mục `training/` và `validation/`, phân biệt qua **cấu trúc tên file**. File CSV `tess_ravdess` ánh xạ toàn bộ hai nguồn này.

---

## 2. Cấu trúc thư mục

```
emotion-recognition-using-speech-master/
│
├── 📂 data/                          ← Toàn bộ file âm thanh WAV
│   ├── 📂 training/                  ← Dữ liệu huấn luyện (RAVDESS + TESS)
│   │   ├── Actor_01/  (138 file)     ← Diễn viên 01: TESS + RAVDESS
│   │   ├── Actor_02/                 ← Diễn viên 02
│   │   ├── ...
│   │   ├── Actor_24/                 ← Diễn viên 24 (RAVDESS gốc)
│   │   ├── Actor_25/  (~200+ file)   ← Diễn viên 25 (TESS — từ đơn)
│   │   └── Actor_26/  (~200+ file)   ← Diễn viên 26 (TESS — từ đơn)
│   │
│   ├── 📂 validation/                ← Dữ liệu kiểm thử (RAVDESS + TESS)
│   │   ├── Actor_07/
│   │   ├── Actor_08/ ... Actor_26/   ← 14 actors được chọn làm tập test
│   │
│   ├── 📂 emodb/                     ← Bộ dữ liệu tiếng Đức EMO-DB
│   │   └── wav/  (535 file)          ← File âm thanh WAV gốc
│   │
│   ├── 📂 train-custom/              ← Dữ liệu huấn luyện do người dùng cung cấp
│   └── 📂 test-custom/               ← Dữ liệu kiểm thử do người dùng cung cấp
│
└── 📂 metadata/                      ← File CSV mô tả dataset (được sinh tự động)
    ├── train_tess_ravdess.csv         ← 3.833 dòng (train RAVDESS+TESS)
    ├── test_tess_ravdess.csv          ← ~600 dòng (test RAVDESS+TESS)
    ├── train_emodb.csv                ← 327 dòng (train EMO-DB)
    ├── test_emodb.csv                 ← ~80 dòng (test EMO-DB)
    ├── train_custom.csv               ← 99 dòng (train custom)
    └── test_custom.csv                ← ~50 dòng (test custom)
```

---

## 3. Thư mục `data/` — Dữ liệu âm thanh

### 3.1 RAVDESS + TESS — `training/` & `validation/`

#### Tổng quan
- **Định dạng:** WAV (PCM, 16-bit, 22050 Hz hoặc 48000 Hz)
- **Cấu trúc:** Mỗi diễn viên có một thư mục riêng `Actor_XX/`
- **Phân chia train/test:** Actor được phân chia cố định — `training/` có **26 actors**, `validation/` chỉ giữ lại **14 actors** được chọn ngẫu nhiên

#### Phân tách Actor theo nguồn gốc

| Actor ID | Nguồn | Giới tính | Thư mục |
|---|---|---|---|
| Actor_01 → Actor_24 | **RAVDESS** | Nam chẵn, Nữ lẻ | `training/` + `validation/` |
| Actor_25 | **TESS** | Nữ (26 tuổi) | Chỉ `training/` |
| Actor_26 | **TESS** | Nữ (64 tuổi) | Chỉ `training/` |

#### Tập training (`data/training/`)
- **26 thư mục Actor** (Actor_01 → Actor_26)
- Mỗi Actor_01→24 có khoảng **138–150 file** WAV
- Actor_25, Actor_26 (TESS) có khoảng **200+ file** (từ đơn)
- **Tổng ước tính:** ~3.800+ file WAV trong `training/`

#### Tập validation (`data/validation/`)
- **14 thư mục Actor** được chọn: Actor_07, 08, 09, 10, 14, 15, 16, 17, 20, 21, 22, 23, 25, 26
- **Tổng ước tính:** ~800 file WAV trong `validation/`

---

### 3.2 EMO-DB — `emodb/wav/`

#### Tổng quan
| Thuộc tính | Chi tiết |
|---|---|
| **Tên đầy đủ** | Berlin Database of Emotional Speech |
| **Nguồn gốc** | Đại học Kỹ thuật Berlin (Technische Universität Berlin) |
| **Ngôn ngữ** | Tiếng Đức |
| **Môi trường** | Thu âm trong phòng cách âm chuyên dụng |
| **Số file trong `wav/`** | **535 file WAV** |
| **Người nói** | 10 diễn viên (5 nam, 5 nữ, mã số: 03, 08, 09, 10, 11, 12, 13, 14, 15, 16) |
| **Kích thước file** | ~40 KB → 290 KB (độ dài âm thanh ngắn hơn RAVDESS) |

#### Cảm xúc trong EMO-DB (7 nhãn)
```
angry  (tức giận)  · neutral (trung tính) · happy  (vui)
sad    (buồn)      · fear    (sợ hãi)    · disgust (ghê tởm)
boredom (chán nản)
```

> **Lưu ý:** EMO-DB không có nhãn `calm` và `ps`. Khi hệ thống nạp dữ liệu, các nhãn EMO-DB được ánh xạ sang không gian nhãn thống nhất.

#### Phân chia train/test EMO-DB
Dựa trên mã người nói (2 ký tự đầu của tên file):

| Tập dữ liệu | Mã người nói | Số file (ước tính) |
|---|---|---|
| **train_emodb** | 03, 08, 09, 10, 11, 12, 13, 14, 15 | **327 file** |
| **test_emodb** | 16 (và một phần 15) | **~80 file** |

---

### 3.3 Custom — `train-custom/` & `test-custom/`

- **Hiện trạng:** `test-custom/` hiện **rỗng** (không có file)
- **`train-custom/`:** Chứa file WAV do người dùng tự thu âm (`hamed`, `nadjib`, `oumaima`, `rockikz`, `soumaya`)
- **Quy ước đặt tên:** `{người_nói}_{stt}_{cảm_xúc}.wav` (ví dụ: `hamed2_neutral.wav`)
- **Nhãn hiện có:** `neutral`, `happy`
- **Số bản ghi:** ~99 file (theo metadata)

---

## 4. Thư mục `metadata/` — File CSV mô tả

### 4.1 Cấu trúc chung của tất cả file CSV

Tất cả 6 file CSV trong `metadata/` đều dùng **cùng một schema**:

```csv
,path,emotion
0,data/training\Actor_01\01_01_01_01_dogs-sitting_sad.wav,sad
1,data/emodb/wav\03a01Fa.wav,happy
2,data/train-custom\hamed2_neutral.wav,neutral
```

| Cột | Kiểu dữ liệu | Ý nghĩa |
|---|---|---|
| *(index)* | `int` | Chỉ số hàng tự động (Pandas default index) |
| `path` | `str` | Đường dẫn tương đối đến file WAV từ thư mục gốc dự án |
| `emotion` | `str` | Nhãn cảm xúc bằng tiếng Anh thường (lowercase) |

> File CSV được sinh tự động bởi `core/create_csv.py` và được `core/data_extractor.py` nạp vào Pandas DataFrame để cấp dữ liệu cho quá trình huấn luyện.

---

### 4.2 `train_tess_ravdess.csv`

| Thuộc tính | Giá trị |
|---|---|
| **Kích thước file** | ~248 KB |
| **Số dòng dữ liệu** | **3.833 dòng** |
| **Nguồn dữ liệu** | RAVDESS (Actor_01→24) + TESS (Actor_25, Actor_26) |
| **Thư mục tham chiếu** | `data/training/Actor_*/` |
| **Chứa nhãn** | `neutral`, `calm`, `happy`, `sad`, `angry`, `fear`, `disgust`, `ps` |

**Mẫu dữ liệu:**
```csv
,path,emotion
0,data/training\Actor_01\01_01_01_01_dogs-sitting_sad.wav,sad
16,data/training\Actor_01\03-02-04-01-01-01-01_sad.wav,sad
464,data/training\Actor_25\25_01_01_01_far_sad.wav,sad
```

**Đặc điểm nổi bật:**
- Dữ liệu từ TESS (Actor_25–26) chứa các file **từ đơn** (far, fat, fit, five, food...) theo từng cảm xúc
- Dữ liệu từ RAVDESS (Actor_01–24) chứa các câu nói với **nội dung cố định** ("dogs sitting", "kids talking") kết hợp cảm xúc
- Một Actor có thể có cả hai loại file (TESS-style + RAVDESS-style)

---

### 4.3 `test_tess_ravdess.csv`

| Thuộc tính | Giá trị |
|---|---|
| **Kích thước file** | ~45 KB |
| **Số dòng dữ liệu** | **~600 dòng** |
| **Nguồn dữ liệu** | RAVDESS + TESS (14 actors được chọn làm tập test) |
| **Thư mục tham chiếu** | `data/validation/Actor_*/` |
| **Chứa nhãn** | Giống `train_tess_ravdess.csv` |

**Actors trong tập validation:**
```
Actor_07, Actor_08, Actor_09, Actor_10, Actor_14, Actor_15,
Actor_16, Actor_17, Actor_20, Actor_21, Actor_22, Actor_23,
Actor_25, Actor_26
```

---

### 4.4 `train_emodb.csv`

| Thuộc tính | Giá trị |
|---|---|
| **Kích thước file** | ~12 KB |
| **Số dòng dữ liệu** | **327 dòng** |
| **Nguồn dữ liệu** | EMO-DB (người nói 03, 08→15) |
| **Thư mục tham chiếu** | `data/emodb/wav/` |
| **Chứa nhãn** | `angry`, `neutral`, `happy`, `sad`, `fear` |

**Mẫu dữ liệu:**
```csv
,path,emotion
0,data/emodb/wav\03a01Fa.wav,happy
1,data/emodb/wav\03a01Nc.wav,neutral
2,data/emodb/wav\03a01Wa.wav,angry
8,data/emodb/wav\03a04Ad.wav,fear
```

**Phân phối cảm xúc ước tính trong EMO-DB:**

| Cảm xúc | Ký hiệu EMO-DB | Số lượng (train) |
|---|---|---|
| `angry` | `W` (Wut) | ~130 |
| `neutral` | `N` (Neutral) | ~60 |
| `happy` | `F` (Freude) | ~55 |
| `sad` | `T` (Trauer) | ~45 |
| `fear` | `A` / `F` (Angst) | ~37 |

> EMO-DB có **mất cân bằng lớp** rõ rệt — `angry` chiếm áp đảo. Đây là lý do nên bật `balance=True` khi dùng dataset này.

---

### 4.5 `test_emodb.csv`

| Thuộc tính | Giá trị |
|---|---|
| **Kích thước file** | ~3 KB |
| **Số dòng dữ liệu** | **~80 dòng** |
| **Nguồn dữ liệu** | EMO-DB (người nói 15→16) |
| **Thư mục tham chiếu** | `data/emodb/wav/` |

---

### 4.6 `train_custom.csv`

| Thuộc tính | Giá trị |
|---|---|
| **Kích thước file** | ~4.7 KB |
| **Số dòng dữ liệu** | **99 dòng** |
| **Nguồn dữ liệu** | Người dùng tự thu âm |
| **Thư mục tham chiếu** | `data/train-custom/` |
| **Chứa nhãn** | `neutral`, `happy` |

**Người nói (speakers) trong custom dataset:**
```
hamed    · nadjib    · oumaima
rockikz  · soumaya
```

**Mẫu dữ liệu:**
```csv
,path,emotion
1,data/train-custom\hamed2_neutral.wav,neutral
50,data/train-custom\hamed1_happy.wav,happy
74,data/train-custom\oumaima1_happy.wav,happy
```

---

### 4.7 `test_custom.csv`

| Thuộc tính | Giá trị |
|---|---|
| **Kích thước file** | ~2.7 KB |
| **Số dòng dữ liệu** | **~50 dòng** |
| **Nguồn dữ liệu** | Người dùng tự thu âm (phần kiểm thử) |

---

## 5. Quy ước đặt tên file

### 5.1 Quy ước đặt tên TESS

**Định dạng:** `{actor_id}_{gender}_{modality}_{channel}_{word}_{emotion}.wav`

```
Ví dụ: 25_01_01_01_dogs-sitting_sad.wav
        │    │    │    │     │             │
        │    │    │    │     │             └── Cảm xúc (sad, happy, fear...)
        │    │    │    │     └──────────────── Từ/câu nói ("dogs-sitting", "far", "food"...)
        │    │    │    └────────────────────── Kênh âm thanh (01 = stereo left...)
        │    │    └─────────────────────────── Cách diễn đạt (01 = normal, 02 = strong)
        │    └──────────────────────────────── Giới tính (01 = nữ, 02 = nam — TESS)
        └───────────────────────────────────── ID Actor (25 hoặc 26)
```

**Ví dụ thực tế từ Actor_25 (TESS):**
```
25_01_01_01_far_sad.wav       ← từ "far" phát âm buồn
25_01_01_01_dogs-sitting_angry.wav  ← câu "dogs sitting" với tức giận
```

---

### 5.2 Quy ước đặt tên RAVDESS

**Định dạng:** `{modality}-{vocal-channel}-{emotion}-{intensity}-{statement}-{repetition}-{actor}.wav`

```
Ví dụ: 03-02-05-02-01-01-01_angry.wav
        │   │   │   │   │   │   │
        │   │   │   │   │   │   └── Actor ID (01–24)
        │   │   │   │   │   └────── Repetition (01, 02)
        │   │   │   │   └────────── Statement (01 = "Kids talking...", 02 = "Dogs sitting...")
        │   │   │   └────────────── Intensity (01 = normal, 02 = strong)
        │   │   └────────────────── Emotion (01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fear, 07=disgust, 08=ps)
        │   └────────────────────── Vocal channel (01 = speech, 02 = song)
        └────────────────────────── Modality (03 = audio-only)
```

**Bảng mã cảm xúc RAVDESS:**
| Mã số | Nhãn trong project |
|---|---|
| `01` | `neutral` |
| `02` | `calm` |
| `03` | `happy` |
| `04` | `sad` |
| `05` | `angry` |
| `06` | `fear` |
| `07` | `disgust` |
| `08` | `ps` (pleasant surprise) |

---

### 5.3 Quy ước đặt tên EMO-DB

**Định dạng:** `{speaker}{sentence}{emotion}{version}.wav`

```
Ví dụ: 03a04Fa.wav
        │  │  │ │
        │  │  │ └── Phiên bản/lặp lại (a, b, c...)
        │  │  └──── Mã cảm xúc (xem bảng bên dưới)
        │  └──────── Mã câu nói (a01–a07, b01–b10, b09, b10)
        └──────────── Mã người nói (03, 08, 09, 10, 11, 12, 13, 14, 15, 16)
```

**Bảng mã cảm xúc EMO-DB:**
| Mã | Tiếng Đức | Nhãn ánh xạ |
|---|---|---|
| `W` | Wut | `angry` |
| `L` | Langeweile | `boredom` |
| `E` | Ekel | `disgust` |
| `A` | Angst | `fear` |
| `F` | Freude | `happy` |
| `T` | Trauer | `sad` |
| `N` | Neutral | `neutral` |

---

### 5.4 Quy ước đặt tên Custom

**Định dạng tự do:** `{tên_người}_{stt}_{cảm_xúc}.wav`

```
Ví dụ: hamed2_neutral.wav
         │      │  │
         │      │  └── Nhãn cảm xúc (neutral, happy, sad, angry...)
         │      └────── Số thứ tự bản ghi
         └───────────── Tên người nói
```

> Khi thêm dữ liệu custom mới, **phải giữ đúng định dạng** `*_{emotion}.wav` để `create_csv.py` có thể tự động trích xuất nhãn.

---

## 6. Phân phối nhãn cảm xúc

### Nhãn có mặt trong từng dataset

| Cảm xúc | RAVDESS | TESS | EMO-DB | Custom |
|---|:---:|:---:|:---:|:---:|
| `neutral` (trung tính) | ✅ | ✅ | ✅ | ✅ |
| `calm` (bình tĩnh) | ✅ | ❌ | ❌ | ❌ |
| `happy` (vui) | ✅ | ✅ | ✅ | ✅ |
| `sad` (buồn) | ✅ | ✅ | ✅ | ❌ |
| `angry` (tức giận) | ✅ | ✅ | ✅ | ❌ |
| `fear` (sợ hãi) | ✅ | ✅ | ✅ | ❌ |
| `disgust` (ghê tởm) | ✅ | ✅ | ✅ | ❌ |
| `ps` (ngạc nhiên dễ chịu) | ✅ | ❌ | ❌ | ❌ |
| `boredom` (chán nản) | ❌ | ❌ | ✅ | ❌ |

### Số mẫu ước tính theo cảm xúc (tập huấn luyện tổng hợp)

```
angry   ████████████████████████████████ ~910
neutral ████████████████████  ~650
sad     ███████████████████████████████ ~860
happy   ████████████████████████  ~720
fear    ████████████████████  ~620
calm    ██████████████  ~450
disgust █████████████████████  ~650
ps      ████████████████  ~500
```

> Số liệu ước tính từ kết quả `rec.get_samples_by_class()` được ghi nhận trong README.md. Gọi hàm này để lấy số liệu chính xác.

---

## 7. Luồng xử lý dữ liệu

Dưới đây là cách dữ liệu từ `data/` và `metadata/` được hệ thống sử dụng:

```
┌─────────────────────────────────────────────────────────────┐
│                      LUỒNG DỮ LIỆU                         │
└─────────────────────────────────────────────────────────────┘

  [data/training/]   [data/emodb/wav/]   [data/train-custom/]
        │                   │                      │
        └───────────────────┴──────────────────────┘
                            │
                    core/create_csv.py
                    (quét tên file → trích nhãn)
                            │
                            ▼
  ┌────────────────────────────────────────────────────┐
  │               metadata/ (6 file CSV)               │
  │  train_tess_ravdess.csv  │  test_tess_ravdess.csv  │
  │  train_emodb.csv         │  test_emodb.csv          │
  │  train_custom.csv        │  test_custom.csv         │
  └────────────────────────────────────────────────────┘
                            │
                 core/data_extractor.py
                 (AudioExtractor.load_data())
                 Đọc CSV → dùng path nạp WAV
                            │
                            ▼
             librosa.load(path) → waveform
                            │
                    core/utils.py
                    extract_feature()
                            │
             ┌─────────────┼──────────────┐
             │             │              │
           MFCC (40D)  Chroma (12D)  Mel Spec (128D)
             │             │              │
             └─────────────┴──────────────┘
                           │
                   concat → vector 180D
                           │
                   Lưu cache vào features/*.npy
                           │
             ┌─────────────┴──────────────┐
             │                            │
    EmotionRecognizer            DeepEmotionRecognizer
    (sklearn classifier)         (Keras LSTM/GRU)
```

---

## 8. Cơ chế sinh metadata — `create_csv.py`

File `core/create_csv.py` thực hiện **tự động quét và lập danh sách** dữ liệu:

### Cách hoạt động

```python
# Giả lập quy trình trong create_csv.py
for actor_dir in data/training/Actor_*/:
    for wav_file in actor_dir:
        label = extract_from_filename(wav_file)
        # TESS:   "25_01_01_01_far_sad.wav"     → "sad"
        # RAVDESS: "03-02-05-01-01-01_angry.wav" → "angry"
        rows.append({"path": relative_path, "emotion": label})

save_to_csv("metadata/train_tess_ravdess.csv", rows)
```

### Khi nào cần chạy lại

```bash
python core/create_csv.py
```

Chạy lại khi:
- Thêm file WAV mới vào `data/training/` hoặc `data/emodb/wav/`
- Thêm dữ liệu custom vào `data/train-custom/`
- Thay đổi phân chia train/test

---

## 9. Trích xuất đặc trưng — AudioExtractor

File `core/data_extractor.py` sử dụng `AudioExtractor` để:

1. **Đọc CSV metadata** → lấy danh sách `(path, emotion)`
2. **Nạp file WAV** qua `librosa.load()` với sample rate mặc định
3. **Cắt im lặng** (Silence Trimming) — tiền xử lý giảm nhiễu
4. **Gọi `extract_feature()`** từ `core/utils.py`

### Vector đặc trưng 180D

| Đặc trưng | Chiều | Mô tả kỹ thuật |
|---|---|---|
| **MFCC** | 40D | Mel-Frequency Cepstral Coefficients — biểu diễn hình bao phổ âm thanh theo thang mel |
| **Chroma** | 12D | Chromagram pitch classes — nắm bắt nội dung thanh điệu và hoà âm |
| **Mel Spectrogram** | 128D | Năng lượng phổ tần số ánh xạ theo thang mel |

### Cache `.npy`

```
features/
├── train_ravdess_angry_sad_happy.npy   ← Cache X (đặc trưng)
└── train_ravdess_angry_sad_happy_labels.npy  ← Cache y (nhãn)
```

- File cache được đặt tên theo **tập dữ liệu + danh sách cảm xúc** đang dùng
- Nếu tồn tại cache, hệ thống **bỏ qua** bước trích xuất để tiết kiệm thời gian (5–15 phút → < 5 giây)
- Xoá `features/*.npy` khi cần tái trích xuất

---

## 10. Tóm tắt số liệu

### Số liệu file metadata

| File CSV | Kích thước | Số dòng | Bộ dữ liệu tham chiếu |
|---|---|---|---|
| `train_tess_ravdess.csv` | 248 KB | **~3.833** | RAVDESS + TESS (training) |
| `test_tess_ravdess.csv` | 45 KB | **~600** | RAVDESS + TESS (validation) |
| `train_emodb.csv` | 12 KB | **327** | EMO-DB (train) |
| `test_emodb.csv` | 3 KB | **~80** | EMO-DB (test) |
| `train_custom.csv` | 4.7 KB | **99** | Custom (train) |
| `test_custom.csv` | 2.7 KB | **~50** | Custom (test) |
| **Tổng** | **~315 KB** | **~4.989 dòng** | — |

### Số liệu thư mục âm thanh

| Thư mục | Số file WAV | Nguồn |
|---|---|---|
| `data/training/` (26 actors) | **~3.800+** | RAVDESS + TESS |
| `data/validation/` (14 actors) | **~800** | RAVDESS + TESS |
| `data/emodb/wav/` | **535** | EMO-DB |
| `data/train-custom/` | **~99** | Custom |
| `data/test-custom/` | **0 (rỗng)** | Custom |
| **TỔNG** | **~5.234+ file WAV** | — |

### Tổng quan nhanh

```
📦 Dung lượng tổng bộ dữ liệu:   ~2.0 – 2.5 GB (ước tính)
🎵 Tổng số file WAV:              ~5.234+ file
📋 Tổng số dòng metadata CSV:     ~4.989 bản ghi
😠 Số nhãn cảm xúc hỗ trợ:       9 nhãn (tối đa)
🌍 Ngôn ngữ:                      Tiếng Anh (RAVDESS, TESS) + Tiếng Đức (EMO-DB)
👥 Tổng số người nói:             36+ người (24 RAVDESS + 2 TESS + 10 EMO-DB + custom)
```

---

## Tài liệu liên quan

| File | Nội dung |
|---|---|
| [`README.md`](README.md) | Tổng quan dự án, hướng dẫn cài đặt và sử dụng |
| [`PROJECT_GUIDE.md`](PROJECT_GUIDE.md) | Hướng dẫn vận hành đầy đủ, sơ đồ kiến trúc, sequence diagram |
| [`core/create_csv.py`](core/create_csv.py) | Script sinh file metadata CSV từ thư mục dữ liệu |
| [`core/data_extractor.py`](core/data_extractor.py) | Module nạp dữ liệu từ CSV và trích xuất đặc trưng |
| [`core/utils.py`](core/utils.py) | Hàm `extract_feature()` — trích xuất MFCC, Chroma, Mel Spectrogram |
| [`tools/analyze_dataset.py`](tools/analyze_dataset.py) | Script phân tích và trực quan hoá phân phối dataset |

---

*Tài liệu được tạo tự động dựa trên phân tích cấu trúc thư mục `data/` và `metadata/` của dự án.*  
*Cập nhật lần cuối: 2026-04-09*
