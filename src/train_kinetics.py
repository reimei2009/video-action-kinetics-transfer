"""
Training script cho Kinetics Subset
Chạy trên Kaggle Notebook để pretrain/fine-tune trên Kinetics 5%
"""

import argparse
import yaml
import torch
from pathlib import Path


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train model trong 1 epoch"""
    print(f"  [train_one_epoch] Epoch {epoch} - training...")
    # TODO: Implement training loop
    return 0.5, 75.0  # dummy loss, acc


def evaluate(model, dataloader, criterion, device):
    """Evaluate model trên validation set"""
    print(f"  [evaluate] Validating...")
    # TODO: Implement evaluation loop
    return 0.6, 70.0  # dummy val_loss, val_acc


def main(config_path):
    print(f"\n{'='*60}")
    print(f"=== Kinetics Training Script ===")
    print(f"{'='*60}\n")
    print(f"Config file: {config_path}")
    
    # Load config
    if Path(config_path).exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"✓ Config loaded successfully\n")
        
        # Print key configs
        print(f"--- Dataset Configuration ---")
        print(f"  Dataset root: {config.get('dataset_root', 'N/A')}")
        print(f"  Selected classes ({len(config.get('selected_classes', []))}): {config.get('selected_classes', [])[:3]}...")
        print(f"  Num frames: {config.get('num_frames', 16)}")
        print(f"  Batch size: {config.get('batch_size', 8)}")
        
        print(f"\n--- Model Configuration ---")
        print(f"  Model: {config.get('model_name', 'x3d_xs')}")
        print(f"  Pretrained: {config.get('pretrained', True)}")
        print(f"  Freeze backbone: {config.get('freeze_backbone', False)}")
        
        print(f"\n--- Training Configuration ---")
        print(f"  Max epochs: {config.get('max_epochs', config.get('epochs', 10))}")
        print(f"  Learning rate: {config.get('learning_rate', 0.0001)}")
        print(f"  Output dir: {config.get('output_dir', config.get('save_dir', '/kaggle/working'))}")
        
    else:
        print(f"⚠ Config file not found: {config_path}")
        print(f"  Using default parameters\n")
        config = {
            'max_epochs': 2,
            'selected_classes': ['playing_guitar', 'cooking']
        }
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n✓ Device: {device}")
    
    # TODO: Load dataset
    print(f"✓ Dataset: [will load from {config.get('dataset_root', 'Kaggle')}]")
    
    # TODO: Create model
    print(f"✓ Model: [will build {config.get('model_name', 'X3D')} model]")
    
    # Training loop skeleton
    max_epochs = config.get('max_epochs', config.get('epochs', 2))
    print(f"\n{'='*60}")
    print(f"=== Training Loop (Skeleton) ===")
    print(f"{'='*60}\n")
    
    for epoch in range(1, max_epochs + 1):
        print(f"Epoch {epoch}/{max_epochs}")
        train_loss, train_acc = train_one_epoch(None, None, None, None, device, epoch)
        val_loss, val_acc = evaluate(None, None, None, device)
        print(f"  Train: loss={train_loss:.4f}, acc={train_acc:.2f}%")
        print(f"  Val:   loss={val_loss:.4f}, acc={val_acc:.2f}%\n")
    
    print(f"{'='*60}")
    print(f"✓ Training completed!")
    print(f"  Weights will be saved to: {config.get('output_dir', config.get('save_dir', 'weights'))}/x3d_kinetics_subset_best.pth")
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
  
  # On Kaggle
  !python src/train_kinetics.py --config configs/kinetics_subset.yaml
        """
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/kinetics_subset.yaml',
        help='Path to config YAML file (default: configs/kinetics_subset.yaml)'
    )
    args = parser.parse_args()
    
    main(args.config)
