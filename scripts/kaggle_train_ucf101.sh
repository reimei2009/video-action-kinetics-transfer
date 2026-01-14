#!/bin/bash
# Kaggle Training Script - NSAR Transfer Learning
# Run this in Kaggle Notebook cell

set -e

echo "=== Cloning repository ==="
git clone https://github.com/<YOUR_USERNAME>/video-action-kinetics-transfer.git
cd video-action-kinetics-transfer

echo "=== Installing dependencies ==="
pip install -q -r requirements.txt

echo "=== Starting transfer learning ==="
python src/train_nsar.py --config configs/ucf101_transfer.yaml

echo "=== Training completed ==="
echo "Weights saved to: /kaggle/working/weights/x3d_ucf101_best.pth"
echo "Download from Kaggle Output tab"
