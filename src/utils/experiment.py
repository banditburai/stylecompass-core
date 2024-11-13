from pathlib import Path
from typing import Any, Dict, Optional
import wandb
from .logging import setup_logger
from .config import Config
import torch

logger = setup_logger(__name__)

class Experiment:
    """Experiment tracking and management."""
    
    def __init__(
        self,
        name: str,
        config: Config,
        project: str = "stylecompass",
        save_dir: Optional[Path] = None,
    ):
        """Initialize experiment tracker.
        
        Args:
            name: Experiment name
            config: Configuration object
            project: W&B project name
            save_dir: Directory to save checkpoints
        """
        self.name = name
        self.config = config
        self.save_dir = save_dir or Path("experiments") / name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize W&B
        try:
            self.run = wandb.init(
                project=project,
                name=name,
                config=config.config,
                dir=str(self.save_dir),
            )
            logger.info(f"Initialized experiment '{name}' with wandb")
        except Exception as e:
            logger.warning(f"Failed to initialize wandb: {e}")
            self.run = None
    
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log metrics to wandb and local storage.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Optional step number
        """
        if self.run is not None:
            self.run.log(metrics, step=step)
        logger.info(f"Step {step}: {metrics}")
    
    def save_checkpoint(
        self,
        state: Dict[str, Any],
        filename: str = "checkpoint.pt"
    ) -> None:
        """Save model checkpoint.
        
        Args:
            state: State dictionary to save
            filename: Name of checkpoint file
        """
        save_path = self.save_dir / filename
        try:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(state, save_path)
            if self.run is not None:
                self.run.save(str(save_path))
            logger.info(f"Saved checkpoint to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise
    
    def finish(self) -> None:
        """Clean up experiment tracking."""
        if self.run is not None:
            self.run.finish()
            logger.info("Finished wandb run")