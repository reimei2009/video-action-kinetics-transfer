# Scripts tiện ích

## Kaggle training script

**kaggle_train_kinetics.sh** - Script chạy trên Kaggle Notebook:

```bash
#!/bin/bash
# Clone repo
git clone https://github.com/<your-username>/video-action-kinetics-transfer.git
cd video-action-kinetics-transfer

# Install dependencies
pip install -r requirements.txt

# Train
python src/train_kinetics.py --config configs/kinetics_subset.yaml

# Show output
echo "Training completed! Check /kaggle/working/weights/"
```

## Local inference script

**inference_example.sh** - Ví dụ chạy inference local:

```bash
#!/bin/bash
python src/inference.py \
    --video test_video.mp4 \
    --model weights/x3d_nsar_best.pth \
    --classes "basketball,soccer,tennis,volleyball,badminton,cricket,hockey,swimming" \
    --device cpu \
    --top_k 3
```

## Docker build script

**docker_build.sh**:

```bash
#!/bin/bash
docker build -t action-recognition:latest .
docker tag action-recognition:latest action-recognition:v1.0
echo "Build completed!"
```
