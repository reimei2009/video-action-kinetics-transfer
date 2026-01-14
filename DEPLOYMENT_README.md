# 🎥 Video Action Recognition API - Deployment Guide

Hướng dẫn triển khai FastAPI + Gradio cho X3D model.

---

## 📦 Installation

### 1. Cài đặt dependencies

```bash
# Install API requirements (bao gồm cả training deps)
pip install -r requirements-api.txt
```

**Lưu ý:** File `requirements-api.txt` đã bao gồm tất cả dependencies cần thiết:
- FastAPI + Uvicorn (API server)
- Gradio (Web UI)
- PyTorch + TorchVision + PyTorchVideo (đã có từ training)

---

## 🚀 Quick Start

### Cách 1: Chạy với script (Khuyến nghị)

```bash
python run_api.py
```

Server sẽ start tại:
- **API**: http://localhost:8000
- **Swagger Docs**: http://localhost:8000/docs
- **Gradio UI**: http://localhost:8000/demo

### Cách 2: Chạy trực tiếp với uvicorn

```bash
# API only
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Với auto-reload (development)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Cách 3: Chạy Gradio standalone

```bash
python -m app.frontend.gradio_app
```

Gradio UI sẽ chạy tại: http://localhost:7860

---

## 📖 API Documentation

### 1. POST /api/v1/predict

Upload video và nhận predictions.

**Request:**
```bash
curl -X POST "http://localhost:8000/api/v1/predict?top_k=5" \
  -F "file=@test_video.mp4"
```

**Response:**
```json
{
  "success": true,
  "message": "Prediction completed successfully",
  "predictions": [
    {"label": "PlayingPiano", "confidence": 0.2100, "rank": 1},
    {"label": "Archery", "confidence": 0.0886, "rank": 2},
    {"label": "PlayingGuitar", "confidence": 0.0885, "rank": 3}
  ],
  "model_name": "ucf101",
  "processing_time": 1.234,
  "video_metadata": {
    "filename": "test_video.mp4",
    "duration": 12.88,
    "fps": 25,
    "size_mb": 2.5
  }
}
```

### 2. GET /api/v1/health

Health check endpoint.

```bash
curl http://localhost:8000/api/v1/health
```

### 3. GET /api/v1/models

Lấy thông tin model hiện tại.

```bash
curl http://localhost:8000/api/v1/models
```

### 4. POST /api/v1/models/switch?model_name=kinetics

Switch giữa Kinetics và UCF101 models.

```bash
curl -X POST "http://localhost:8000/api/v1/models/switch?model_name=kinetics"
```

---

## 🎨 Gradio UI Usage

1. Mở trình duyệt: http://localhost:8000/demo
2. Upload video (drag-and-drop hoặc click)
3. Chọn số lượng predictions (slider)
4. Click **"Predict Action"**
5. Xem kết quả:
   - Top-K predictions với confidence bars
   - Video preview
   - Processing time & metadata

---

## ⚙️ Configuration

### Thay đổi model (Kinetics ↔ UCF101)

Edit file `app/core/config.py`:

```python
MODEL_NAME: str = "ucf101"  # Hoặc "kinetics"
```

### Thay đổi server port

```bash
python run_api.py --port 8080
```

### Thay đổi video settings

Edit `app/core/config.py`:
```python
VIDEO_CLIP_DURATION: float = 2.0    # Clip duration (seconds)
VIDEO_NUM_FRAMES: int = 16          # Number of frames
TOP_K: int = 5                      # Top-K predictions
```

---

## 🏗️ Project Structure

```
video-action-kinetics-transfer/
├── app/
│   ├── __init__.py
│   ├── main.py                    # FastAPI app entry
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py              # API endpoints
│   ├── core/
│   │   ├── __init__.py
│   │   └── config.py              # Settings (model, paths, etc.)
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py             # Pydantic schemas
│   ├── services/
│   │   ├── __init__.py
│   │   ├── model_service.py       # Load X3D model, inference
│   │   └── video_service.py       # Video preprocessing
│   ├── utils/
│   │   ├── __init__.py
│   │   └── logger.py
│   └── frontend/
│       ├── __init__.py
│       └── gradio_app.py          # Gradio UI
├── src/                           # Training code (giữ nguyên)
├── weights/
│   ├── x3d_kinetics_subset_best.pth
│   └── x3d_ucf101_best.pth
├── requirements-api.txt           # API dependencies
└── run_api.py                     # Script chạy server
```

---

## 🧪 Testing

### Test với Python requests

```python
import requests

# Upload video
url = "http://localhost:8000/api/v1/predict"
files = {"file": open("test_video.mp4", "rb")}
params = {"top_k": 5}

response = requests.post(url, files=files, params=params)
print(response.json())
```

### Test với cURL

```bash
# Predict
curl -X POST "http://localhost:8000/api/v1/predict?top_k=5" \
  -F "file=@4088191-hd_1920_1080_25fps.mp4"

# Health check
curl http://localhost:8000/api/v1/health

# Models info
curl http://localhost:8000/api/v1/models
```

---

## 🐛 Troubleshooting

### Lỗi: "Weights file not found"

**Giải pháp:** Đảm bảo file weights tồn tại:
```bash
ls -l weights/
# Phải có: x3d_kinetics_subset_best.pth, x3d_ucf101_best.pth
```

### Lỗi: "Module not found"

**Giải pháp:** Cài đặt dependencies:
```bash
pip install -r requirements-api.txt
```

### Lỗi: "Address already in use"

**Giải pháp:** Port 8000 đã được sử dụng, đổi port:
```bash
python run_api.py --port 8080
```

### Lỗi: Gradio UI không hiển thị

**Giải pháp:** Kiểm tra Gradio đã cài đặt:
```bash
pip install gradio==4.13.0
```

---

## 📊 Performance

### Inference Time

- **CPU**: ~1-2 seconds/video
- **CUDA GPU**: ~0.3-0.5 seconds/video

### Upload Limits

- **Max file size**: 100 MB (config trong `app/core/config.py`)
- **Supported formats**: MP4, AVI, MOV, MKV, WEBM, FLV

---

## 🚀 Next Steps (Deploy to Cloud)

### 1. Docker

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY . /app

RUN pip install -r requirements-api.txt

CMD ["python", "run_api.py", "--host", "0.0.0.0", "--port", "8000"]
```

Build & Run:
```bash
docker build -t video-action-api .
docker run -p 8000:8000 video-action-api
```

### 2. Azure App Service

```bash
# Deploy với Azure CLI
az webapp up --name video-action-api --runtime "PYTHON:3.10"
```

### 3. AWS EC2

```bash
# SSH vào EC2 instance
git clone <repo>
pip install -r requirements-api.txt
python run_api.py --host 0.0.0.0 --port 8000
```

---

## 📝 Notes

- Model được load **1 lần duy nhất** khi start server (Singleton pattern)
- Mỗi request tạo temp file, tự động cleanup sau khi xử lý
- Hỗ trợ CORS cho phép frontend gọi API từ domain khác
- API docs tự động gen bởi FastAPI (Swagger UI)

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first.

---

## 📄 License

MIT License - Free to use for personal and commercial projects.

---

## 👨‍💻 Author

Developed for Ki1Nam4 Project - Video Action Recognition with X3D

**Tech Stack:**
- FastAPI (API framework)
- Gradio (Web UI)
- PyTorch + PyTorchVideo (Deep Learning)
- X3D Model (Facebook Research)
- UCF101 & Kinetics-400 datasets
