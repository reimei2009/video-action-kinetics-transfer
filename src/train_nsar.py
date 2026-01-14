"""
Training script cho NSARPMD Sports Dataset
Transfer learning từ Kinetics pretrained weights
"""

import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from pathlib import Path
import sys
import os

# Add src to path
src_dir = Path(__file__).parent.resolve()
sys.path.insert(0, str(src_dir))

from datasets.nsar_sports import NSARSportsDataset, get_nsar_transforms
from models.x3d_wrapper import build_x3d


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train model trong 1 epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')
    for videos, labels in pbar:
        videos = videos.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(videos)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # Metrics
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({
            'loss': f"{running_loss / (pbar.n + 1):.4f}",
            'acc': f"{100. * correct / total:.2f}%"
        })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device):
    """Evaluate model trên validation set"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    if len(dataloader) == 0:
        return 0.0, 0.0
    
    with torch.no_grad():
        for videos, labels in tqdm(dataloader, desc='[Val]'):
            videos = videos.to(device)
            labels = labels.to(device)
            
            outputs = model(videos)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_loss = running_loss / len(dataloader)
    val_acc = 100. * correct / total
    return val_loss, val_acc


def main(config_path):
    print(f"\n=== NSAR Transfer Learning Script ===")
    print(f"Config: {config_path}")
    
    # Load config
    if Path(config_path).exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"✓ Config loaded")
        print(f"  - Model: {config.get('model_name', 'x3d_xs')}")
        print(f"  - Freeze backbone: {config.get('freeze_backbone', True)}")
        print(f"  - Kinetics weights: {config.get('kinetics_weights', 'N/A')}")
    else:
        print(f"⚠ Config file not found: {config_path}")
        config = {'epochs': 2}
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✓ Device: {device}")
    
    # Load NSAR dataset
    print(f"\n=== Loading NSAR Dataset ===")
    data_root = config.get('data_root', '/kaggle/input/nsar-sports/Dataset')
    
    transform = get_nsar_transforms(
        num_frames=config.get('num_frames', 16),
        crop_size=config.get('crop_size', 224)
    )
    
    full_dataset = NSARSportsDataset(
        data_root=data_root,
        annotation_file=config.get('train_annotation'),
        clip_duration=config.get('clip_duration', 2.0),
        num_frames=config.get('num_frames', 16),
        split='train',
        transform=transform
    )
    
    print(f"✓ Total samples: {len(full_dataset)}")
    print(f"✓ Classes: {full_dataset.sports_classes}")
    
    if len(full_dataset) == 0:
        print(f"\n❌ ERROR: No videos found in dataset!")
        print(f"   data_root: {data_root}")
        print(f"\n   Please check the dataset path on Kaggle:")
        print(f"   1. List available datasets:")
        print(f"      !ls -la /kaggle/input/")
        print(f"   2. Check dataset structure:")
        print(f"      !ls -la /kaggle/input/YOUR_DATASET_NAME/")
        print(f"   3. Update data_root in configs/nsar_transfer.yaml")
        return
    
    # Split train/val
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('batch_size', 4),
        shuffle=True,
        num_workers=config.get('num_workers', 2)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get('batch_size', 4),
        shuffle=False,
        num_workers=config.get('num_workers', 2)
    )
    
    print(f"✓ Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Build model
    print(f"\n=== Building Model ===")
    num_classes = len(full_dataset.sports_classes)
    
    model = build_x3d(
        num_classes=num_classes,
        model_name=config.get('model_name', 'x3d_xs'),
        pretrained=False,
        freeze_backbone=config.get('freeze_backbone', True)
    )
    
    # Load Kinetics weights
    kinetics_weights = config.get('kinetics_weights')
    if kinetics_weights and os.path.exists(kinetics_weights):
        print(f"✓ Loading Kinetics weights: {kinetics_weights}")
        checkpoint = torch.load(kinetics_weights, map_location='cpu')
        
        # Load state dict (skip classifier layer vì khác num_classes)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Remove classifier weights
        state_dict = {k: v for k, v in state_dict.items() if 'blocks.5.proj' not in k}
        model.load_state_dict(state_dict, strict=False)
        print(f"  ✓ Transferred {len(state_dict)} layers from Kinetics")
    
    model = model.to(device)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.get('learning_rate', 0.0001),
        weight_decay=config.get('weight_decay', 0.0001)
    )
    
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.get('scheduler_step', 15),
        gamma=config.get('scheduler_gamma', 0.1)
    )
    
    # Training loop
    print(f"\n=== Starting Transfer Learning ===")
    best_val_acc = 0.0
    save_dir = Path(config.get('save_dir', 'weights'))
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(1, config.get('epochs', 30) + 1):
        print(f"\nEpoch {epoch}/{config.get('epochs', 30)}")
        
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        print(f"  Train: loss={train_loss:.4f}, acc={train_acc:.2f}%")
        print(f"  Val:   loss={val_loss:.4f}, acc={val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = save_dir / 'x3d_nsar_best.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_acc': train_acc,
                'val_acc': val_acc,
                'config': config
            }, save_path)
            print(f"  ✓ Saved best model: {save_path} (val_acc={val_acc:.2f}%)")
    
    print(f"\n✓ Transfer learning completed!")
    print(f"  Best val acc: {best_val_acc:.2f}%")
    print(f"  Weights saved to: {save_dir / 'x3d_nsar_best.pth'}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transfer learning to NSAR Sports')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/nsar_transfer.yaml',
        help='Path to config file'
    )
    args = parser.parse_args()
    
    main(args.config)
