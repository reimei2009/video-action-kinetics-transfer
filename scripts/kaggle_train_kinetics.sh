#!/bin/bash
# Kaggle Training Script - Kinetics Subset
# Run this in Kaggle Notebook cell

set -e

echo "=== Cloning repository ==="
git clone https://github.com/<YOUR_USERNAME>/video-action-kinetics-transfer.git
cd video-action-kinetics-transfer

echo "=== Installing dependencies ==="
pip install -q -r requirements.txt

echo "=== Starting training ==="
python src/train_kinetics.py --config configs/kinetics_subset.yaml

echo "=== Training completed ==="
echo "Weights saved to: /kaggle/working/weights/x3d_kinetics_subset_best.pth"
echo "Download from Kaggle Output tab"
