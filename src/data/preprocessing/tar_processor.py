from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass
from src.utils import setup_logger, Config
import subprocess

logger = setup_logger(__name__)

@dataclass
class TarProcessor:
    dataset_name: str
    config_path: Optional[Path] = None
    
    def __post_init__(self) -> None:
        self.config = Config(self.config_path or Path("configs/data_processing.yaml"))
        self.dataset_name = self.dataset_name or self.config.dataset.name

    def download_tar_range(self, start_part: int, end_part: int) -> None:
        """Download specific tar files using huggingface-cli"""
        files_to_download = [
            f"dataset_part_{i:04d}.tar"
            for i in range(start_part, end_part + 1)
        ]
        
        logger.info(f"Downloading tar files {start_part} to {end_part}")
        cmd = ["huggingface-cli", "download", "--repo-type=dataset", self.dataset_name] + files_to_download
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info("Download completed successfully")
            return result.stdout
        except subprocess.CalledProcessError as e:
            logger.error(f"Download failed: {e.stderr}")
            raise RuntimeError(f"Failed to download files: {e.stderr}") from e