import torch
import torch.nn as nn
from torchvision import transforms
from typing import Tuple

from models.factory import ModelConfig

def create_sscd_model(config: ModelConfig) -> Tuple[nn.Module, transforms.Compose]:
    """Create SSCD model
    
    Args:
        config: Model configuration
        
    Returns:
        model: SSCD model
        transform: Preprocessing transform
    """
    if config.arch == 'resnet50':
        model = torch.jit.load("./pretrainedmodels/sscd_disc_mixup.torchscript.pt")
    elif config.arch == 'resnet50_disc':
        model = torch.jit.load("./pretrainedmodels/sscd_disc_large.torchscript.pt")
    else:
        raise NotImplementedError('This model type does not exist/supported for SSCD')
        
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    
    return model, transform