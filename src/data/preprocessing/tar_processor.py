from pathlib import Path
from typing import List, Optional, Dict
from dataclasses import dataclass
from src.utils import setup_logger, Config
import subprocess
import os

logger = setup_logger(__name__)

@dataclass
class TarProcessor:
    dataset_name: str
    config_path: Optional[Path] = None
    
    def __post_init__(self) -> None:
        # Get project root directory
        project_root = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
        
        # Use provided config path or default
        default_config = project_root / "configs" / "data_processing.yaml"
        config_path = self.config_path or default_config
        
        logger.info(f"Loading config from: {config_path}")
        self.config = Config(config_path)
        self._setup_directories()
    
    def _setup_directories(self) -> None:
        """Create necessary directories from config paths."""
        # Get paths from config
        paths_dict: Dict[str, str] = self.config.get("paths", {})
        
        # Convert strings to Path objects and expand user paths
        self.paths = {
            key: Path(str(path)).expanduser().resolve()
            for key, path in paths_dict.items()
        }
        
        logger.info(f"Setting up directories: {self.paths}")
        
        # Create directories
        for path_key, path in self.paths.items():
            # Create directory and any parent directories
            path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directory: {path}")
            setattr(self, path_key, path)

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