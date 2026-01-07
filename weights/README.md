# Weights Directory

Thư mục này chứa model weights tải từ Kaggle/Colab.

## Files cần có:

1. **x3d_kinetics_subset_best.pth**
   - Pretrained weights từ Kinetics 5%
   - Tải từ Kaggle sau khi chạy `train_kinetics.py`

2. **x3d_nsar_best.pth**
   - Transfer learning weights từ NSARPMD
   - Tải từ Kaggle sau khi chạy `train_nsar.py`

## Lưu ý:

- Không commit file `.pth` lên GitHub (đã có trong `.gitignore`)
- File weights thường có dung lượng 10-50 MB
