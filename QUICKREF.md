# Quick Reference Card

## 🚀 Commands Cheat Sheet

### Test Structure
```bash
python test_structure.py
```

### Git Operations
```bash
git status                          # Xem trạng thái
git add .                           # Add tất cả files
git commit -m "message"             # Commit
git push origin main                # Push lên GitHub
```

### Training (sẽ chạy trên Kaggle)
```bash
# Kinetics pretrain
python src/train_kinetics.py --config configs/kinetics_subset.yaml

# NSAR transfer learning
python src/train_nsar.py --config configs/nsar_transfer.yaml
```

### Inference (local)
```bash
python src/inference.py \
  --video path/to/video.mp4 \
  --model weights/x3d_nsar_best.pth \
  --classes "basketball,soccer,tennis" \
  --device cpu
```

### Docker
```bash
docker build -t action-recognition .
docker-compose up
```

---

## 📋 File Locations

| Mục đích | File |
|----------|------|
| Kinetics config | `configs/kinetics_subset.yaml` |
| NSAR config | `configs/nsar_transfer.yaml` |
| X3D model | `src/models/x3d_wrapper.py` |
| Kinetics loader | `src/datasets/kinetics_subset.py` |
| NSAR loader | `src/datasets/nsar_sports.py` |
| Train Kinetics | `src/train_kinetics.py` |
| Train NSAR | `src/train_nsar.py` |
| Inference | `src/inference.py` |
| Kaggle scripts | `scripts/*.sh` |
| Model weights | `weights/` (empty, sẽ có sau training) |

---

## 🎯 Pipeline Overview

```
1. [Local] Setup structure → Push to GitHub
                ↓
2. [Kaggle] Clone repo → Train Kinetics → Save weights
                ↓
3. [Kaggle] Load Kinetics weights → Transfer NSAR → Save weights
                ↓
4. [Local] Download weights → Implement inference → Test
                ↓
5. [Local] Docker deployment
```

---

## ⚠️ Important Notes

1. **Weights không được commit** - File .gitignore đã ignore `weights/*.pth`
2. **Dataset không commit** - Dùng Kaggle dataset trực tiếp
3. **Virtual env không commit** - `.venv/` đã được ignore
4. **Skeleton code** - Các script hiện tại chỉ là khung, chưa train thật

---

## ✅ Checklist Bước 1

- [x] Tạo cấu trúc thư mục
- [x] Viết skeleton code cho training scripts
- [x] Viết skeleton code cho inference
- [x] Tạo configs YAML
- [x] Tạo requirements.txt
- [x] Tạo Dockerfile
- [x] Tạo .gitignore
- [x] Tạo README.md đầy đủ
- [x] Test structure (all passed)
- [ ] Push to GitHub ← **NEXT STEP**

---

## 📞 Key Functions

### `src/models/x3d_wrapper.py`
- `build_x3d(num_classes, ...)` - Tạo X3D model

### `src/train_kinetics.py`
- `train_one_epoch(...)` - Training loop
- `evaluate(...)` - Validation
- `main(config_path)` - Entrypoint

### `src/train_nsar.py`
- Tương tự train_kinetics.py
- Thêm load Kinetics weights

### `src/inference.py`
- `predict_video(video, weights, classes, device)` - Dự đoán video
