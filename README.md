# Video Action Recognition - Kinetics Transfer Learning

Dự án nhận diện hành động trong video sử dụng 3D CNN (X3D), pretrain trên Kinetics 5% và transfer learning sang NSARPMD Sports Dataset.

## 📋 Tổng quan

**Pipeline:**
1. **Pretrain/Fine-tune** trên Kinetics 5% (Kaggle/Colab) → weights `x3d_kinetics_subset_best.pth`
2. **Transfer learning** sang NSARPMD sports dataset → weights `x3d_nsar_best.pth`
3. **Deploy** trên máy cá nhân (VS Code) với inference script

**Datasets:**
- **Kinetics 5%**: Chọn 10-20 classes từ Kinetics-400
- **NSARPMD**: National Sports Action Recognition Dataset (124 videos HD)

**Framework:**
- PyTorch + TorchVision + PyTorchVideo
- Model: X3D (Facebook Research)

---

## 🏗️ Cấu trúc thư mục

```
video-action-kinetics-transfer/
├── src/
│   ├── datasets/
│   │   ├── kinetics_subset.py      # DataLoader cho Kinetics 5%
│   │   └── nsar_sports.py          # DataLoader cho NSARPMD
│   ├── models/
│   │   └── x3d_wrapper.py          # X3D model wrapper
│   ├── train_kinetics.py           # Script train Kinetics subset
│   ├── train_nsar.py               # Script transfer learning NSAR
│   └── inference.py                # Script inference local
├── configs/
│   ├── kinetics_subset.yaml        # Config Kinetics training
│   └── nsar_transfer.yaml          # Config NSAR transfer
├── weights/                        # Model weights (tải từ Kaggle)
├── scripts/                        # Shell scripts tiện ích
├── requirements.txt
└── README.md
```

---

## 🚀 Hướng dẫn sử dụng

### 1. Setup môi trường local (VS Code)

```bash
# Tạo virtual environment
python -m venv .venv

# Activate (Windows)
.\.venv\Scripts\activate

# Activate (Linux/Mac)
source .venv/bin/activate

# Cài đặt dependencies
pip install -r requirements.txt
```

### 2. Pretrain trên Kinetics 5% (Kaggle)

**Trên Kaggle Notebook:**

1. Tạo notebook mới, thêm dataset **Kinetics 5%**
2. Clone repo này:
   ```python
   !git clone https://github.com/<your-username>/video-action-kinetics-transfer.git
   %cd video-action-kinetics-transfer
   !pip install -r requirements.txt
   ```
3. Chỉnh sửa `configs/kinetics_subset.yaml`:
   - `data_root`: đường dẫn dataset Kinetics trên Kaggle
   - `selected_classes`: chọn 10-20 classes
4. Chạy training:
   ```python
   !python src/train_kinetics.py --config configs/kinetics_subset.yaml
   ```
5. Tải weights về máy:
   - File: `/kaggle/working/weights/x3d_kinetics_subset_best.pth`
   - Download từ Kaggle Output hoặc:
     ```python
     from IPython.display import FileLink
     FileLink('weights/x3d_kinetics_subset_best.pth')
     ```

### 3. Transfer learning trên NSARPMD (Kaggle/Colab)

**Trên Kaggle Notebook:**

1. Thêm dataset:
   - **NSARPMD** dataset
   - **Kinetics weights** (upload file `.pth` làm private dataset)
2. Clone repo, cài dependencies
3. Chỉnh sửa `configs/nsar_transfer.yaml`:
   - `data_root`: đường dẫn NSARPMD
   - `kinetics_weights`: đường dẫn file weight Kinetics
4. Chạy training:
   ```python
   !python src/train_nsar.py --config configs/nsar_transfer.yaml
   ```
5. Tải weights về: `x3d_nsar_best.pth`

### 4. Inference trên máy local

**Sau khi tải weights về:**

1. Đặt weights vào thư mục `weights/`:
   ```
   weights/
   ├── x3d_kinetics_subset_best.pth
   └── x3d_nsar_best.pth
   ```

2. Chạy inference:
   ```bash
   python src/inference.py \
       --video path/to/video.mp4 \
       --model weights/x3d_nsar_best.pth \
       --classes "basketball,soccer,tennis,volleyball,badminton,cricket,hockey,swimming" \
       --device cpu
   ```

**Ví dụ output:**
```
=== Predictions for video.mp4 ===
1. basketball: 0.8523 (85.23%)
2. volleyball: 0.0921 (9.21%)
3. tennis: 0.0356 (3.56%)
```

---

## 🐳 Docker Deployment (Optional)

**Dockerfile:**
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "src/inference.py"]
```

**Build & Run:**
```bash
docker build -t action-recognition .
docker run -v $(pwd)/videos:/app/videos action-recognition \
    --video /app/videos/test.mp4 \
    --model weights/x3d_nsar_best.pth \
    --classes "basketball,soccer,..."
```

---

## 📊 Kết quả mong đợi

| Stage | Dataset | Accuracy |
|-------|---------|----------|
| Pretrain | Kinetics 10 classes | ~70-80% |
| Transfer | NSARPMD 8 sports | ~80-90% |

---

## 🔧 Cấu hình training

**Kinetics:**
- Model: X3D-XS (pretrained Kinetics-400)
- Batch size: 8
- Learning rate: 0.001
- Epochs: 20

**NSARPMD (Transfer):**
- Model: X3D-XS + Kinetics weights
- Freeze backbone: ✓
- Batch size: 4
- Learning rate: 0.0001
- Epochs: 30

---

## 📚 Tài liệu tham khảo

- [PyTorchVideo](https://github.com/facebookresearch/pytorchvideo)
- [X3D Paper](https://arxiv.org/abs/2004.04730)
- [Kinetics Dataset](https://github.com/cvdfoundation/kinetics-dataset)

---

## 📝 License

MIT License

---

## 👤 Author

Reimei2009 - [GitHub](https://github.com/Ki1Nam4)
