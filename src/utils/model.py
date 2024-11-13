import os
from collections import OrderedDict
from typing import Dict, List, Optional
import torch
import torch.nn as nn

def get_params_groups(model: nn.Module) -> List[Dict]:
    """Get parameter groups for optimizer"""
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]


def clip_gradients(model: nn.Module, clip: float) -> List[float]:
    """Clip gradients by global norm"""
    norms = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            norms.append(param_norm.item())
            clip_coef = clip / (param_norm + 1e-6)
            if clip_coef < 1:
                p.grad.data.mul_(clip_coef)
    return norms

def convert_state_dict(state_dict: OrderedDict) -> OrderedDict:
    """Convert state dict from parallel to single model"""
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k.replace("module.", "")
        new_state_dict[k] = v
    return new_state_dict

def restart_from_checkpoint(
    ckp_path: str,
    run_variables: Optional[Dict] = None,
    **kwargs
):
    """Restart training from checkpoint"""
    if not os.path.isfile(ckp_path):
        return
    print(f"Found checkpoint at {ckp_path}")
    
    checkpoint = torch.load(ckp_path, map_location="cpu")
    
    for key, value in kwargs.items():
        if key in checkpoint and value is not None:
            try:
                value.load_state_dict(checkpoint[key])
            except:
                msg = f"Error loading {key} from checkpoint"
                print(msg)
    
    if run_variables is not None:
        for var_name in run_variables:
            if var_name in checkpoint:
                run_variables[var_name] = checkpoint[var_name]

def has_batchnorms(model: nn.Module) -> bool:
    """Check if model contains batch normalization layers"""
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
    for name, module in model.named_modules():
        if isinstance(module, bn_types):
            return True
    return False

def convert_weights_float(model: nn.Module) -> None:
    """Convert applicable model parameters to fp32"""
    def _convert_weights_to_fp32(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.float()
            if l.bias is not None:
                l.bias.data = l.bias.data.float()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.float()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.float()

    model.apply(_convert_weights_to_fp32)

