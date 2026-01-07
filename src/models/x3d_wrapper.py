"""
X3D Model Wrapper
Hỗ trợ load pretrained X3D từ PyTorchVideo và thay đổi số lớp output
"""

import torch
import torch.nn as nn


def build_x3d(
    num_classes=10,
    model_name='x3d_xs',
    pretrained=True,
    freeze_backbone=False
):
    """
    Tạo X3D model với số lớp tùy chỉnh
    
    Args:
        num_classes: số lớp output
        model_name: 'x3d_xs', 'x3d_s', 'x3d_m'
        pretrained: load pretrained weights từ Kinetics-400
        freeze_backbone: freeze các layer backbone (chỉ train classifier)
    
    Returns:
        model: X3D model
    """
    
    # Load pretrained model từ PyTorchVideo hub
    if pretrained:
        model = torch.hub.load(
            'facebookresearch/pytorchvideo',
            model_name,
            pretrained=True
        )
    else:
        model = torch.hub.load(
            'facebookresearch/pytorchvideo',
            model_name,
            pretrained=False
        )
    
    # Thay fully connected layer cuối
    # X3D có blocks.5.proj (projection layer)
    in_features = model.blocks[5].proj.in_features
    model.blocks[5].proj = nn.Linear(in_features, num_classes)
    
    # Freeze backbone nếu cần (transfer learning)
    if freeze_backbone:
        for name, param in model.named_parameters():
            if 'blocks.5.proj' not in name:  # Không freeze classifier
                param.requires_grad = False
    
    return model


def create_x3d_model(
    model_name='x3d_xs',
    num_classes=10,
    pretrained=True,
    freeze_backbone=False
):
    """
    Alias for build_x3d (backward compatibility)
    """
    return build_x3d(num_classes, model_name, pretrained, freeze_backbone)


def create_r3d_model(num_classes=10, pretrained=True):
    """
    Tạo R3D-18 model (alternative cho X3D)
    
    Args:
        num_classes: số lớp output
        pretrained: load pretrained weights
    
    Returns:
        model: R3D model
    """
    from torchvision.models.video import r3d_18, R3D_18_Weights
    
    if pretrained:
        weights = R3D_18_Weights.KINETICS400_V1
        model = r3d_18(weights=weights)
    else:
        model = r3d_18(weights=None)
    
    # Thay FC layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    
    return model


if __name__ == '__main__':
    # Test model creation
    model = build_x3d(num_classes=10, pretrained=False)
    print(f"Model created: {model.__class__.__name__}")
    print(f"Output classes: 10")
    
    # Test forward pass
    dummy_input = torch.randn(1, 3, 16, 224, 224)  # (B, C, T, H, W)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")  # Should be (1, 10)
