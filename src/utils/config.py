from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union, Literal
from functools import cached_property
from .logging import setup_logger
import yaml

logger = setup_logger(__name__)

@dataclass(frozen=True)
class DistributedConfig:
    world_size: int = 1
    rank: int = -1
    dist_url: str = "tcp://localhost:6001"
    dist_backend: str = "nccl"
    multiprocessing_distributed: bool = False

@dataclass(frozen=True)
class ModelConfig:
    backbone: str = "vit"
    embedding_dim: int = 512
    num_heads: int = 8
    pt_style: str = "csd"
    arch: str = "vit_large"
    content_proj_head: str = "default"
    eval_embed: Literal["head", "backbone"] = "head"
    device: str = "cuda"
    pretrained_path: Optional[Path] = None

@dataclass(frozen=True)
class FeatureConfig:
    type: str = "normal"
    projdim: int = 256
    layer: int = 1
    gram_dims: int = 1024
    multiscale: bool = False
    qsplit: str = "query"

@dataclass(frozen=True)
class DataConfig:
    dataset: str = "artbreeder"
    train_path: Path = Path("data/train")
    val_path: Path = Path("data/val")
    image_size: int = 224
    num_workers: int = 8
    query_count: int = -1

@dataclass(frozen=True)
class OutputConfig:
    embed_dir: Path = Path("./embeddings")
    skip_val: bool = False

@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    seed: Optional[int] = None
    use_fp16: bool = True
    distributed: DistributedConfig = field(default_factory=DistributedConfig)

class Config:
    """Configuration management with type-safe access."""
    
    def __init__(self, config_path: Optional[Union[Path, str]] = None):
        self._config: Dict[str, Any] = {}
        if config_path:
            self.load_config(Path(config_path))

    def load_config(self, config_path: Path) -> None:
        """Load configuration from YAML file."""
        try:
            with config_path.open() as f:
                self._config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            raise

    @cached_property
    def distributed(self) -> DistributedConfig:
        return DistributedConfig(**self._config.get("training", {}).get("distributed", {}))

    @cached_property
    def model(self) -> ModelConfig:
        return ModelConfig(**self._config.get("model", {}))

    @cached_property
    def features(self) -> FeatureConfig:
        return FeatureConfig(**self._config.get("features", {}))

    @cached_property
    def training(self) -> TrainingConfig:
        return TrainingConfig(**self._config.get("training", {}))

    @cached_property
    def data(self) -> DataConfig:
        return DataConfig(**self._config.get("data", {}))

    @cached_property
    def output(self) -> OutputConfig:
        return OutputConfig(**self._config.get("output", {}))

    def __getitem__(self, key: str) -> Any:
        """Enable dictionary-like access for backward compatibility."""
        return self._config[key]

    def get(self, key: str, default: Any = None) -> Any:
        """Dictionary-like get with default for backward compatibility."""
        return self._config.get(key, default)