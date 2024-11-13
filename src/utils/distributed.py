import os
import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import warnings
from dataclasses import dataclass
from typing import Optional
import random
import torch.multiprocessing as mp
import sys
from torch.autograd import Function

@dataclass
class DistributedConfig:
    world_size: int = -1
    rank: int = -1
    dist_url: str = "tcp://224.66.41.62:23456"
    dist_backend: str = "nccl"
    gpu: Optional[int] = None
    multiprocessing_distributed: bool = False
    seed: Optional[int] = None

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def setup_for_distributed(is_master):
    """Disables printing when not in master process"""
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def init_distributed_mode(args):
    """Initialize distributed training settings"""
    args.distributed = True
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    elif torch.cuda.is_available():
        print('Will run the code on one GPU.')
        args.rank, args.gpu, args.world_size = 0, 0, 1
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
    else:
        print('Does not support training without GPU.')
        sys.exit(1)

    dist.init_process_group(
        backend="nccl",
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )

def setup_distributed_training(config: DistributedConfig):
    """Main entry point for distributed training setup"""
    if config.seed is not None:
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        cudnn.deterministic = True
        warnings.warn('Using deterministic training - performance may be impacted')

    if config.dist_url == "env://" and config.world_size == -1:
        config.world_size = int(os.environ["WORLD_SIZE"])

    config.distributed = config.world_size > 1 or config.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()

    if config.multiprocessing_distributed:
        config.world_size = ngpus_per_node * config.world_size
        mp.spawn(
            _distributed_worker,
            nprocs=ngpus_per_node,
            args=(ngpus_per_node, config)
        )
        return True
    return False

def _distributed_worker(gpu: int, ngpus_per_node: int, config: DistributedConfig):
    """Worker function for distributed training"""
    config.gpu = gpu

    if config.gpu is not None:
        print(f"Using GPU: {gpu}")

    if config.distributed:
        if config.dist_url == "env://" and config.rank == -1:
            config.rank = int(os.environ["RANK"])
        if config.multiprocessing_distributed:
            config.rank = config.rank * ngpus_per_node + gpu

        dist.init_process_group(
            backend=config.dist_backend,
            init_method=config.dist_url,
            world_size=config.world_size,
            rank=config.rank
        )

def setup_device(config: DistributedConfig, model: torch.nn.Module):
    """Setup model for distributed/GPU training"""
    if not torch.cuda.is_available():
        print('Using CPU - this will be slow')
        return model
        
    if config.distributed:
        if config.gpu is not None:
            torch.cuda.set_device(config.gpu)
            model.cuda(config.gpu)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[config.gpu]
            )
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif config.gpu is not None:
        torch.cuda.set_device(config.gpu)
        model = model.cuda(config.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()
        
    return model


class GatherLayer(Function):
    """Gather tensors from all processes, supporting backward propagation"""
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        input, = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0

def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)