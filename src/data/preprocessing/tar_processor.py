from pathlib import Path
from typing import List, Optional, Dict
from dataclasses import dataclass, field
from src.utils import setup_logger, Config
import subprocess
import os

logger = setup_logger(__name__)

@dataclass
class TarProcessor:
    dataset_name: str
    config_path: Path = field(default=Path('configs/data_processing.yaml'))
    config: Config = field(init=False)
    paths: Dict[str, Path] = field(init=False)
    
    def __post_init__(self):
        """Initialize after creation"""
        self.config = Config(self.config_path)
        self.paths = {
            'cache_dir': Path(self.config.get("paths.cache_dir")),
            'checkpoint_path': Path(self.config.get("paths.checkpoint_path")),
            'embeddings_dir': Path(self.config.get("paths.embeddings_dir")),
            'output_dir': Path(self.config.get("paths.output_dir")),
            'stylepriors_path': Path(self.config.get("paths.stylepriors_path")),
            'temp_dir': Path(self.config.get("paths.temp_dir"))
        }
        self._setup_directories()
    
    def _setup_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        for name, path in self.paths.items():
            # Skip if it's a file path rather than a directory
            if name in ['checkpoint_path', 'stylepriors_path']:
                # These are files, not directories
                continue
            
            # Create directories
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)

    def download_tar_range(self, start_part: int, end_part: int) -> None:
        """Download specific tar files using huggingface-cli"""
        if start_part > end_part:
            raise ValueError(f"start_part ({start_part}) must be <= end_part ({end_part})")
            
        files_to_download = [
            f"dataset_part_{i:04d}.tar"
            for i in range(start_part, end_part + 1)
        ]
        
        logger.info(f"Downloading tar files {start_part} to {end_part} from {self.dataset_name}")
        cmd = [
            "huggingface-cli", 
            "download", 
            "--repo-type=dataset", 
            "--local-dir", str(self.paths["temp_dir"]),
            self.dataset_name
        ] + files_to_download
        
        try:
            # Remove capture_output=True to see progress bars
            result = subprocess.run(
                cmd, 
                check=True,
                text=True
            )
            logger.info("Download completed successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"Download failed: {e.stderr}")
            raise RuntimeError(f"Failed to download files") from e