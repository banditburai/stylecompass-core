from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision import transforms

from .datasets.wikiart import create_wikiart_datasets

@dataclass
class DataLoaderConfig:
    batch_size: int
    num_workers: int = 4
    pin_memory: bool = True
    distributed: bool = False
    world_size: Optional[int] = None
    feature_type: str = "normal"
    
    def __post_init__(self):
        if self.distributed and not self.world_size:
            raise ValueError("world_size must be provided for distributed training")

def create_dataloaders(
    dataset_query: Dataset,
    dataset_values: Dataset,
    config: DataLoaderConfig
) -> Tuple[DataLoader, DataLoader]:
    """Create query and value dataloaders with proper distributed handling."""
    
    # Adjust batch size for distributed training
    effective_batch_size = (
        int(config.batch_size / config.world_size) 
        if config.distributed 
        else config.batch_size
    )
    
    # Create samplers
    query_sampler = (
        DistributedSampler(dataset_query, shuffle=False)
        if config.distributed
        else None
    )
    values_sampler = (
        DistributedSampler(dataset_values, shuffle=False)
        if config.distributed
        else None
    )
    
    # Special batch size for gram features
    query_batch_size = (
        32 if config.feature_type == "gram" 
        else effective_batch_size
    )
    
    # Create loaders
    query_loader = DataLoader(
        dataset_query,
        batch_size=query_batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        sampler=query_sampler,
        drop_last=False
    )
    
    values_loader = DataLoader(
        dataset_values,
        batch_size=effective_batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        sampler=values_sampler,
        drop_last=False
    )
    
    return query_loader, values_loader

def get_dataloaders(
    dataset_name: str,
    data_dir: Path,
    transform: transforms.Compose,
    config: DataLoaderConfig
) -> Tuple[DataLoader, DataLoader]:
    """High-level function to get dataloaders for a specific dataset."""
    
    # Get appropriate datasets
    if dataset_name == "wikiart":
        datasets = create_wikiart_datasets(data_dir, transform)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Create and return dataloaders
    return create_dataloaders(
        dataset_query=datasets['query'],
        dataset_values=datasets['database'],
        config=config
    )