from typing import Dict, Optional, Tuple, List
import sys
import math
import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.cuda.amp import GradScaler

from .model import CSD_CLIP
from src.utils.model import get_params_groups, clip_gradients
from src.utils.training import (
    split_reshape,
    cosine_scheduler,
    constant_with_warmup_scheduler,
    cancel_gradients_last_layer
)
from src.utils.distributed import GatherLayer

class CSDTrainer:
    def __init__(
        self,
        model: CSD_CLIP,
        loss_content: nn.Module,
        loss_style: nn.Module,
        config: Dict
    ):
        self.model = model
        self.loss_content = loss_content
        self.loss_style = loss_style
        self.config = config
        self.device = next(model.parameters()).device
        
        # Setup optimizers and schedulers
        self.opt_bb = self._setup_backbone_optimizer()
        self.opt_proj = self._setup_projection_optimizer()
        self.lr_schedule_bb, self.lr_schedule_proj = self._setup_schedulers()
        self.fp16_scaler = GradScaler() if config.use_fp16 else None

    def _setup_backbone_optimizer(self) -> SGD:
        params_groups = get_params_groups(self.model.backbone)
        return SGD(
            params_groups, 
            lr=0,  # lr is set by scheduler
            momentum=0.9,
            weight_decay=self.config.weight_decay
        )

    def _setup_projection_optimizer(self) -> SGD:
        if self.config.content_proj_head != 'default':
            params = [
                {'params': self.model.last_layer_style},
                {'params': self.model.last_layer_content.parameters()},
            ]
        else:
            params = [
                self.model.last_layer_style,
                self.model.last_layer_content
            ]
        
        return SGD(
            params,
            lr=0,  # lr is set by scheduler
            momentum=0.9,
            weight_decay=0  # no weight decay for projection
        )

    def _setup_schedulers(self) -> Tuple[List[float], List[float]]:
        """Setup learning rate schedulers"""
        world_size = self.config.get('world_size', 1)
        batch_size = self.config.batch_size_per_gpu
        base_lr_bb = self.config.lr_bb * (batch_size * world_size) / 256.  # linear scaling rule
        base_lr_proj = self.config.lr * (batch_size * world_size) / 256.  # linear scaling rule
        
        if self.config.lr_scheduler_type == 'cosine':
            lr_schedule_bb = cosine_scheduler(
                base_value=base_lr_bb,
                final_value=min(self.config.min_lr, self.config.lr_bb),
                iters=max(self.config.iters, 1),
                warmup_iters=min(self.config.warmup_iters, self.config.iters)
            )
            lr_schedule_proj = cosine_scheduler(
                base_value=base_lr_proj,
                final_value=min(self.config.min_lr, self.config.lr),
                iters=max(self.config.iters, 1),
                warmup_iters=min(self.config.warmup_iters, self.config.iters)
            )
        else:  # constant_with_warmup or constant
            warmup_iters = (
                min(self.config.warmup_iters, self.config.iters)
                if self.config.lr_scheduler_type == 'constant_with_warmup'
                else 0
            )
            lr_schedule_bb = constant_with_warmup_scheduler(
                base_value=base_lr_bb,
                iters=max(self.config.iters, 1),
                warmup_iters=warmup_iters
            )
            lr_schedule_proj = constant_with_warmup_scheduler(
                base_value=base_lr_proj,
                iters=max(self.config.iters, 1),
                warmup_iters=warmup_iters
            )
        
        return lr_schedule_bb, lr_schedule_proj

    def training_step(
        self, 
        images: torch.Tensor,
        artists: torch.Tensor,
        iter_num: int,
        total_iters: int
    ) -> Dict[str, float]:
        """Single training step implementation"""
        self.model.train()
        
        # Calculate alpha for gradient reversal
        if self.config.non_adv_train:
            alpha = None
        else:
            p = float(iter_num) / total_iters
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

        # Move data to GPU
        images = images.to(self.device, non_blocking=True)
        artists = artists.to(self.device, non_blocking=True).float()

        # Forward pass with automatic mixed precision
        with torch.cuda.amp.autocast(self.fp16_scaler is not None):
            _, content_output, style_output = self.model(images, alpha)

            # Normalize outputs
            content_output = nn.functional.normalize(content_output, dim=1, p=2)
            style_output = nn.functional.normalize(style_output, dim=1, p=2)

            # Reshape outputs for contrastive learning
            style_output = split_reshape(style_output, self.config.batch_size, [0, 1])
            content_output = split_reshape(content_output, self.config.batch_size, [0, -1])

            # Gather from all GPUs if using distributed training
            if self.config.use_distributed_loss:
                style_output = torch.cat(GatherLayer.apply(style_output), dim=0)
                content_output = torch.cat(GatherLayer.apply(content_output), dim=0)
                artists = torch.cat(GatherLayer.apply(artists), dim=0)

            # Calculate content loss
            loss_c = self.loss_content(content_output)
            if self.config.clamp_content_loss is not None:
                loss_c = loss_c.clamp(max=self.config.clamp_content_loss)
                if self.config.non_adv_train:
                    loss_c = -1 * loss_c

            # Calculate style loss
            label_mask = artists @ artists.t()
            if self.config.style_loss_type == 'SimClr':
                loss_s_ssl = self.loss_style(style_output)
                loss_s_sup = torch.tensor(0.).to(self.device)
            elif self.config.style_loss_type == 'OnlySup':
                loss_s_ssl = torch.tensor(0.).to(self.device)
                loss_s_sup = self.loss_style(style_output[:, 0:1, :], mask=label_mask)
            else:
                loss_s_sup = self.loss_style(style_output[:, 0:1, :], mask=label_mask)
                loss_s_ssl = self.loss_style(style_output)

            loss_s = self.config.lam_sup * loss_s_sup + loss_s_ssl
            loss = self.config.lambda_c * loss_c + self.config.lambda_s * loss_s

        if not math.isfinite(loss.item()):
            print(f"Loss is {loss.item()}, stopping training")
            sys.exit(1)

        # Optimization step
        self.opt_bb.zero_grad()
        self.opt_proj.zero_grad()
        
        if self.fp16_scaler is None:
            loss.backward()
            if self.config.clip_grad:
                self._clip_gradients()
            self.opt_bb.step()
            self.opt_proj.step()
        else:
            self.fp16_scaler.scale(loss).backward()
            if self.config.clip_grad:
                self.fp16_scaler.unscale_(self.opt_bb)
                self.fp16_scaler.unscale_(self.opt_proj)
                self._clip_gradients()
            self.fp16_scaler.step(self.opt_bb)
            self.fp16_scaler.step(self.opt_proj)
            self.fp16_scaler.update()

        return {
            'loss': loss.item(),
            'content_loss': loss_c.item(),
            'style_loss': loss_s.item(),
            'style_loss_sup': loss_s_sup.item(),
            'style_loss_ssl': loss_s_ssl.item(),
            'lr_bb': self.opt_bb.param_groups[0]["lr"],
            'lr_proj': self.opt_proj.param_groups[0]["lr"]
        }

    def _clip_gradients(self):
        """Clips gradients by global norm"""
        return clip_gradients(self.model, self.config.clip_grad)