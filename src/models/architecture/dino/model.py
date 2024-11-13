from pathlib import Path
from typing import Tuple
import torch
import torch.nn as nn
from torchvision import transforms
from ....utils.config import ModelConfig
from .vit import (
    dino_vits16, dino_vits8, 
    dino_vitb16, dino_vitb8,
    dino_vitb_cifar10, dino_resnet50
)

def create_dino_model(config: ModelConfig) -> Tuple[nn.Module, transforms.Compose]:
    """Create DINO model and its transform
    
    Args:
        config: Model configuration containing architecture and other settings
        
    Returns:
        tuple: (model, transform)
        - model: The DINO model
        - transform: Preprocessing transform for images
        
    Raises:
        ValueError: If architecture is not supported
    """
    model_mapping = {
        'vit_small16': dino_vits16,
        'vit_small8': dino_vits8,
        'vit_base16': dino_vitb16,
        'vit_base8': dino_vitb8,
        'vit_base_cifar': dino_vitb_cifar10,
        'resnet50': dino_resnet50
    }
    
    if config.arch not in model_mapping:
        raise ValueError(f'Architecture {config.arch} not supported for DINO. '
                        f'Choose from: {list(model_mapping.keys())}')
        
    model = model_mapping[config.arch](pretrained=True)
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225]),
    ])
    
    return model, transform