# 🚀 Quick Start Guide

## 📦 Cài Đặt

```bash
pip install -r requirements-api.txt
```

---

## ▶️ Chạy Server

```bash
python run_api.py
```

Server sẽ chạy tại:
- **API**: http://127.0.0.1:8000
- **Swagger Docs**: http://127.0.0.1:8000/docs
- **Gradio UI**: http://127.0.0.1:8000/demo

### 🛑 Dừng Server

**Cách 1: Trong terminal đang chạy**
```
Ctrl + C
```

**Cách 2: Kill process**
```bash
# Tìm process đang chạy trên port 8000
netstat -ano | findstr :8000

# Kill process theo PID (thay <PID> bằng số thực tế)
taskkill /PID <PID> /F

# Hoặc kill tất cả Python processes (cẩn thận!)
taskkill /IM python.exe /F
```

---

## 🧪 Test API

### Cách 1: Gradio UI (Dễ nhất)

1. Mở trình duyệt: **http://127.0.0.1:8000/demo**
2. Upload video (kéo thả hoặc click)
3. Chọn số lượng predictions (slider)
4. Click **"Predict Action"**
5. Xem kết quả với confidence bars

### Cách 2: Swagger UI

1. Mở: **http://127.0.0.1:8000/docs**
2. Click **POST /api/v1/predict**
3. Click **"Try it out"**
4. Upload video file
5. Set `top_k = 5`
6. Click **"Execute"**

### Cách 3: cURL

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/predict?top_k=5" \
  -F "file=@4088191-hd_1920_1080_25fps.mp4;type=video/mp4"
```

### Cách 4: Python Script

```python
import requests

url = "http://127.0.0.1:8000/api/v1/predict"
files = {"file": ("video.mp4", open("4088191-hd_1920_1080_25fps.mp4", "rb"), "video/mp4")}
params = {"top_k": 5}

response = requests.post(url, files=files, params=params)
print(response.json())
```

---

## 📹 Video Test

**Video mẫu**: `4088191-hd_1920_1080_25fps.mp4` (trong thư mục gốc)

**Kết quả mong đợi:**
```json
{
  "success": true,
  "predictions": [
    {"label": "PlayingPiano", "confidence": 0.91, "rank": 1},
    {"label": "PlayingGuitar", "confidence": 0.02, "rank": 2},
    {"label": "Archery", "confidence": 0.02, "rank": 3}
  ],
  "processing_time": 5.69
}
```

---

## 🔧 Troubleshooting

**Server không chạy?**
```bash
# Check port 8000 có bị chiếm không
netstat -ano | findstr :8000

# Đổi port khác
python run_api.py --port 8080
```

**Lỗi kết nối?**
- Đảm bảo server đang chạy
- Dùng `127.0.0.1` thay vì `0.0.0.0` khi test

---

## 📚 Tài Liệu Chi Tiết

Xem [DEPLOYMENT_README.md](DEPLOYMENT_README.md) để biết thêm chi tiết.
