from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import clip
import copy

from .layers import ProjectionHead, ReverseLayerF, init_weights
from src.utils.model import convert_weights_float

@dataclass
class CSDOutput:
    features: torch.Tensor
    content_output: torch.Tensor
    style_output: torch.Tensor

class CSD_CLIP(nn.Module):
    """CLIP-based Content-Style Disentanglement model"""
    def __init__(
        self, 
        name: str = 'vit_large',
        content_proj_head: str = 'default',
        feat_dim: Optional[int] = None
    ):
        super().__init__()
        self.content_proj_head = content_proj_head
        self.backbone, self.embedding_dim = self._init_backbone(name)
        self.feat_dim = feat_dim or self.embedding_dim
        
        convert_weights_float(self.backbone)
        self._init_projection_heads()

    def _init_backbone(self, name: str) -> Tuple[nn.Module, int]:
        if name == 'vit_large':
            clipmodel, _ = clip.load("ViT-L/14")
            return clipmodel.visual, 1024
        elif name == 'vit_base':
            clipmodel, _ = clip.load("ViT-B/16")
            return clipmodel.visual, 768
        raise ValueError(f'Model {name} not implemented')

    def _init_projection_heads(self) -> None:
        self.last_layer_style = copy.deepcopy(self.backbone.proj)
        
        if self.content_proj_head == 'custom':
            self.last_layer_content = ProjectionHead(
                self.embedding_dim,
                self.feat_dim
            )
            self.last_layer_content.apply(init_weights)
        else:
            self.last_layer_content = copy.deepcopy(self.backbone.proj)
        
        self.backbone.proj = None

    def forward(
        self, 
        input_data: torch.Tensor,
        alpha: Optional[float] = None
    ) -> CSDOutput:
        feature = self.backbone(input_data)
        
        reverse_feature = (
            ReverseLayerF.apply(feature, alpha) 
            if alpha is not None else feature
        )

        style_output = feature @ self.last_layer_style
        style_output = nn.functional.normalize(style_output, dim=1, p=2)

        content_output = (
            self.last_layer_content(reverse_feature)
            if self.content_proj_head == 'custom'
            else reverse_feature @ self.last_layer_content
        )
        content_output = nn.functional.normalize(content_output, dim=1, p=2)

        return CSDOutput(
            features=feature,
            content_output=content_output,
            style_output=style_output
        )