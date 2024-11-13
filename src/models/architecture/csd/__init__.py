from pathlib import Path
from typing import Tuple
import torch
import torch.nn as nn
from torchvision import transforms

from .model import CSD_CLIP
from .transforms import create_transforms
from ....utils.config import ModelConfig
from src.utils.model import convert_state_dict

def has_batchnorms(model):
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.SyncBatchNorm):
            return True
    return False

def create_csd_model(
    config: ModelConfig,
    pretrained: bool = True
) -> Tuple[nn.Module, transforms.Compose]:
    """Create CSD model and its transform pipeline"""
    model = CSD_CLIP(
        name=config.arch,
        content_proj_head=config.content_proj_head,
        eval_embed=config.eval_embed
    )

    if pretrained:
        assert config.pretrained_path is not None, "Model path missing for CSD model"
        checkpoint = torch.load(config.pretrained_path, map_location="cpu")
        state_dict = convert_state_dict(checkpoint['model_state_dict'])
        msg = model.load_state_dict(state_dict, strict=False)
        print(f"=> loaded checkpoint with msg {msg}")

    transforms_b0, _, _ = create_transforms(
        size=config.image_size
    )
    
    return model, transforms_b0

__all__ = [
    'CSD_CLIP',
    'create_csd_model',
    'create_transforms'
]