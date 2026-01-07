"""
Training script cho Kinetics Subset
Chạy trên Kaggle Notebook để pretrain/fine-tune trên Kinetics 5%
"""

import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from pathlib import Path
import sys
import os

# Add src directory to Python path (for Kaggle import)
# train_kinetics.py is in src/, so __file__.parent is src/
src_dir = Path(__file__).parent.absolute()
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from datasets.kinetics_subset import KineticsSubsetDataset, get_kinetics_transforms
from models.x3d_wrapper import build_x3d


def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device, epoch, config):
    """Train model trong 1 epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch_idx, (videos, labels) in enumerate(pbar):
        videos = videos.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        use_amp = config.get('use_amp', False)
        if use_amp:
            with autocast():
                outputs = model(videos)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            
            # Gradient clipping
            if config.get('gradient_clip'):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
            
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(videos)
            loss = criterion(outputs, labels)
            loss.backward()
            
            if config.get('gradient_clip'):
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
            
            optimizer.step()
        
        # Metrics
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        if (batch_idx + 1) % config.get('log_freq', 10) == 0:
            pbar.set_postfix({
                'loss': f"{running_loss / (batch_idx + 1):.4f}",
                'acc': f"{100. * correct / total:.2f}%"
            })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device):
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
    print(f"\n{'='*60}")
    print(f"=== Kinetics Training Script ===")
    print(f"{'='*60}\n")
    
    # Load config
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print(f"✓ Config loaded from: {config_path}\n")
    
    # Print config
    print(f"--- Configuration ---")
    print(f"  Dataset: {config['dataset_root']}")
    print(f"  Classes: {len(config['selected_classes'])}")
    print(f"  Model: {config['model_name']}")
    print(f"  Epochs: {config['max_epochs']}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Learning rate: {config['learning_rate']}")
    print(f"  Debug mode: {config.get('debug_mode', False)}")
    print(f"  Mixed precision: {config.get('use_amp', False)}\n")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✓ Device: {device}\n")
    
    # Create output directory
    output_dir = Path(config.get('output_dir', config.get('save_dir', 'weights')))
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Output directory: {output_dir}\n")
    
    # Load dataset
    print("Loading datasets...")
    transform = get_kinetics_transforms(
        num_frames=config['num_frames'],
        crop_size=config['crop_size']
    )
    
    train_dataset = KineticsSubsetDataset(
        data_root=config['dataset_root'],
        selected_classes=config['selected_classes'],
        clip_duration=config['clip_duration'],
        num_frames=config['num_frames'],
        split='train',
        transform=transform
    )
    
    val_dataset = KineticsSubsetDataset(
        data_root=config['dataset_root'],
        selected_classes=config['selected_classes'],
        clip_duration=config['clip_duration'],
        num_frames=config['num_frames'],
        split='val',
        transform=transform
    )
    
    # Debug mode - limit samples
    if config.get('debug_mode', False):
        max_samples = config.get('max_debug_samples', 50)
        train_dataset.samples = train_dataset.samples[:max_samples]
        val_dataset.samples = val_dataset.samples[:min(max_samples//5, len(val_dataset.samples))]
        print(f"⚠ DEBUG MODE: Limited to {len(train_dataset)} train, {len(val_dataset)} val samples\n")
    
    print(f"✓ Train samples: {len(train_dataset)}")
    print(f"✓ Val samples: {len(val_dataset)}")
    print(f"✓ Num classes: {len(train_dataset.selected_classes)}\n")
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config.get('num_workers', 2),
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config.get('num_workers', 2),
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Model
    print("Creating model...")
    model = build_x3d(
        num_classes=len(train_dataset.selected_classes),
        model_name=config['model_name'],
        pretrained=config['pretrained'],
        freeze_backbone=config.get('freeze_backbone', False)
    )
    model = model.to(device)
    print(f"✓ Model: {config['model_name']} with {len(train_dataset.selected_classes)} classes\n")
    
    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config.get('weight_decay', 1e-4)
    )
    
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.get('scheduler_step', 5),
        gamma=config.get('scheduler_gamma', 0.1)
    )
    
    # Mixed precision scaler
    scaler = GradScaler() if config.get('use_amp', False) else None
    
    # Training loop
    print(f"{'='*60}")
    print(f"=== Starting Training ===")
    print(f"{'='*60}\n")
    
    best_val_acc = 0.0
    
    for epoch in range(1, config['max_epochs'] + 1):
        print(f"Epoch {epoch}/{config['max_epochs']}")
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch, config
        )
        print(f"Train → Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        
        # Validate
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"Val   → Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = output_dir / 'x3d_kinetics_subset_best.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'config': config
            }, save_path)
            print(f"✓ Saved best model: {save_path} (Val Acc: {val_acc:.2f}%)")
        
        # Save checkpoint periodically
        if epoch % config.get('checkpoint_freq', 1) == 0:
            checkpoint_path = output_dir / f'checkpoint_epoch_{epoch}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"✓ Saved checkpoint: {checkpoint_path}")
        
        # Step scheduler
        scheduler.step()
        print(f"Learning rate: {scheduler.get_last_lr()[0]:.6f}\n")
    
    print(f"{'='*60}")
    print(f"✓ Training completed!")
    print(f"  Best Val Acc: {best_val_acc:.2f}%")
    print(f"  Weights saved to: {output_dir}/x3d_kinetics_subset_best.pth")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train X3D on Kinetics Subset (Kaggle)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default config
  python src/train_kinetics.py
  
  # Train with custom config
  python src/train_kinetics.py --config configs/kinetics_subset.yaml
  
  # Debug mode (fast test)
  python src/train_kinetics.py --debug
  
  # On Kaggle
  !python src/train_kinetics.py --config configs/kinetics_subset.yaml
        """
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/kinetics_subset.yaml',
        help='Path to config YAML file'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode (override config)'
    )
    
    args = parser.parse_args()
    
    # Override config if debug flag
    if args.debug:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        config['debug_mode'] = True
        config['max_epochs'] = 2
        
        # Save temp config
        temp_config = 'configs/temp_debug.yaml'
        with open(temp_config, 'w') as f:
            yaml.dump(config, f)
        args.config = temp_config
        print("⚠ DEBUG MODE ENABLED\n")
    
    main(args.config)
