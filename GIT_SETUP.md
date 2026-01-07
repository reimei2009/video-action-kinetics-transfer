# Git Setup Guide

## 1. Khởi tạo Git repository

```bash
# Kiểm tra status
git status

# Add all files
git add .

# Commit
git commit -m "feat: Initial project structure with skeleton code

- Setup project structure for video action recognition
- Add skeleton code for Kinetics training (src/train_kinetics.py)
- Add skeleton code for NSAR transfer learning (src/train_nsar.py)
- Add skeleton code for inference (src/inference.py)
- Add X3D model wrapper (src/models/x3d_wrapper.py)
- Add dataset loaders for Kinetics and NSARPMD
- Add config files for training
- Add Dockerfile and docker-compose for deployment
- Add test script to validate structure

Ready for Kaggle training implementation."
```

## 2. Push lên GitHub

```bash
# Tạo repository trên GitHub trước
# Sau đó:

git remote add origin https://github.com/<your-username>/video-action-kinetics-transfer.git
git branch -M main
git push -u origin main
```

## 3. Tạo branch cho development (optional)

```bash
git checkout -b develop
git push -u origin develop
```

## 4. Workflow tiếp theo

### Trên Kaggle:
1. Clone repo
2. Implement training logic trong `train_kinetics.py`
3. Commit changes back (hoặc copy code về local)
4. Push weights lên GitHub Release hoặc download về local

### Local:
1. Pull changes từ GitHub
2. Implement inference logic
3. Test với weights từ Kaggle
4. Deploy với Docker

---

## Git Ignore Summary

File `.gitignore` đã được cấu hình để ignore:
- ✓ Virtual environment (`.venv/`)
- ✓ Python cache (`__pycache__/`)
- ✓ Model weights (`weights/*.pth`)
- ✓ Video files (`*.mp4`, `*.avi`)
- ✓ Dataset directories (`data/`, `datasets/`)
- ✓ Logs (`logs/`, `*.log`)

Chỉ commit code, không commit weights/data!
