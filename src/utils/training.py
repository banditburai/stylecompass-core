from typing import List, Optional
import numpy as np
import torch
from torch import nn
from collections import OrderedDict

def cancel_gradients_last_layer(epoch: int, model: nn.Module, freeze_last_layer: int) -> None:
    """Cancel gradients for last layer before specified epoch"""
    if epoch >= freeze_last_layer:
        return
    for n, p in model.named_parameters():
        if "last_layer" in n:
            p.grad = None

def split_reshape(x: torch.Tensor, bs: int, combination: Optional[List[int]] = None) -> torch.Tensor:
    """Split and reshape tensor for contrastive learning"""
    if combination is None:
        combination = [0, 1]
    n_views = len(combination)
    return x.view(bs, n_views, -1)

def cosine_scheduler(
    base_value: float,
    final_value: float,
    iters: int,
    warmup_iters: int = 0,
    start_warmup_value: float = 0
) -> List[float]:
    """Cosine learning rate scheduler with warmup"""
    warmup_schedule = np.array([])
    if warmup_iters > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(iters - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == iters
    return schedule.tolist()

def constant_with_warmup_scheduler(
    base_value: float,
    iters: int,
    warmup_iters: int = 0,
    start_warmup_value: float = 0
) -> List[float]:
    """Constant learning rate scheduler with warmup"""
    warmup_schedule = np.array([])
    if warmup_iters > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    schedule = np.array([base_value] * (iters - warmup_iters))
    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == iters
    return schedule.tolist()
