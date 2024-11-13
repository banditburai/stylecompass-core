from pathlib import Path
from typing import Tuple
import torch
import torch.nn as nn
from torchvision import transforms
from ....utils.config import ModelConfig
from timm.models.vision_transformer import _cfg
from .vit import VisionTransformerMoCo
from .stem import ConvStem
from functools import partial

def create_moco_model(config: ModelConfig) -> Tuple[nn.Module, transforms.Compose]:
    """Create MoCo model and its transform"""
    model_mapping = {
        'vit_small': vit_small,
        'vit_base': vit_base,
        'vit_conv_small': vit_conv_small,
        'vit_conv_base': vit_conv_base
    }
    
    if config.arch not in model_mapping:
        raise ValueError(f'Architecture {config.arch} not supported for MoCo. '
                        f'Choose from: {list(model_mapping.keys())}')
    
    model = model_mapping[config.arch](
        stop_grad_conv1=True,
        num_classes=0,
        global_pool=''
    )
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    return model, transform

def vit_small(**kwargs):
    model = VisionTransformerMoCo(
        patch_size=16, embed_dim=384, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

def vit_base(**kwargs):
    model = VisionTransformerMoCo(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

def vit_conv_small(**kwargs):
    # minus one ViT block
    model = VisionTransformerMoCo(
        patch_size=16, embed_dim=384, depth=11, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), embed_layer=ConvStem, **kwargs)
    model.default_cfg = _cfg()
    return model

def vit_conv_base(**kwargs):
    # minus one ViT block
    model = VisionTransformerMoCo(
        patch_size=16, embed_dim=768, depth=11, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), embed_layer=ConvStem, **kwargs)
    model.default_cfg = _cfg()
    return model
