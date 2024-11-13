from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import vgg16, VGG16_Weights

from .architecture.dino import create_dino_model
from .architecture.moco import create_moco_model
from .architecture.clip import load as load_clip
from .architecture.csd import create_csd_model
from .architecture.sscd import create_sscd_model

@dataclass
class ModelConfig:
    backbone: str
    arch: str
    pt_style: str
    embedding_dim: int
    num_heads: int
    content_proj_head: str = "default"
    eval_embed: str = "head"
    pretrained_path: Optional[Path] = None
    device: str = "cuda"
    distributed: bool = False
    feattype: str = "normal"  # For SSCD: ['otprojected', 'weighted', 'concated', 'gram', 'normal']
    image_size: int = 224
    layer: int = 1
    gram_dims: int = 1024
    multiscale: bool = False

class ModelFactory:
    """Factory for creating different embedding models"""
    
    def create_model(self, config: ModelConfig) -> Tuple[nn.Module, transforms.Compose]:
        """Create model and its preprocessing transform"""
        if config.pt_style == "dino":
            return create_dino_model(config)
        elif config.pt_style == "moco":
            return create_moco_model(config)
        elif config.pt_style == "clip":
            return self.create_clip_model(config)
        elif config.pt_style == "vgg":
            return self._create_vgg(config)
        elif config.pt_style == "sscd":
            return create_sscd_model(config)
        elif config.pt_style.startswith("csd"):
            return create_csd_model(config)
        else:
            raise ValueError(f"Unknown model type: {config.pt_style}")

    def _create_vgg(self, config: ModelConfig) -> Tuple[nn.Module, transforms.Compose]:
        model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
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
    
    def create_clip_model(self, config: ModelConfig) -> Tuple[nn.Module, transforms.Compose]:
        """Create CLIP model and transform pipeline"""
        model, transform = load_clip(
            name=config.arch,
            device=config.device
        )
        return model, transform