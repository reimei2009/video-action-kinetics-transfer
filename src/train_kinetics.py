"""
Training script cho Kinetics Subset
Chạy trên Kaggle Notebook để pretrain/fine-tune trên Kinetics 5%
"""

import os
import yaml
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

from datasets.kinetics_subset import KineticsSubsetDataset, get_kinetics_transforms
from models.x3d_wrapper import create_x3d_model


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train model trong 1 epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for videos, labels in pbar:
        videos = videos.to(device)
        labels = labels.to(device)
        
        # Forward
        optimizer.zero_grad()
        outputs = model(videos)
        loss = criterion(outputs, labels)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        # Metrics
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': running_loss / (pbar.n + 1),
            'acc': 100. * correct / total
        })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for videos, labels in tqdm(dataloader, desc='Validating'):
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
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Dataset
    print('Loading datasets...')
    transform = get_kinetics_transforms(
        num_frames=config['num_frames'],
        crop_size=config['crop_size']
    )
    
    train_dataset = KineticsSubsetDataset(
        data_root=config['data_root'],
        selected_classes=config.get('selected_classes', None),
        clip_duration=config['clip_duration'],
        num_frames=config['num_frames'],
        split='train',
        transform=transform
    )
    
    val_dataset = KineticsSubsetDataset(
        data_root=config['data_root'],
        selected_classes=config.get('selected_classes', None),
        clip_duration=config['clip_duration'],
        num_frames=config['num_frames'],
        split='val',
        transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config.get('num_workers', 2)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config.get('num_workers', 2)
    )
    
    print(f'Train samples: {len(train_dataset)}')
    print(f'Val samples: {len(val_dataset)}')
    print(f'Num classes: {len(train_dataset.selected_classes)}')
    
    # Model
    print('Creating model...')
    model = create_x3d_model(
        model_name=config['model_name'],
        num_classes=len(train_dataset.selected_classes),
        pretrained=config['pretrained'],
        freeze_backbone=config.get('freeze_backbone', False)
    )
    model = model.to(device)
    
    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config.get('weight_decay', 1e-4)
    )
    
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.get('scheduler_step', 10),
        gamma=config.get('scheduler_gamma', 0.1)
    )
    
    # Training loop
    best_val_acc = 0.0
    save_dir = Path(config.get('save_dir', 'weights'))
    save_dir.mkdir(exist_ok=True)
    
    for epoch in range(1, config['epochs'] + 1):
        print(f'\n=== Epoch {epoch}/{config["epochs"]} ===')
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = save_dir / 'x3d_kinetics_subset_best.pth'
            torch.save(model.state_dict(), save_path)
            print(f'✓ Saved best model to {save_path}')
        
        # Step scheduler
        scheduler.step()
    
    print(f'\nTraining completed! Best Val Acc: {best_val_acc:.2f}%')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=str,
        default='configs/kinetics_subset.yaml',
        help='Path to config file'
    )
    args = parser.parse_args()
    
    main(args.config)
