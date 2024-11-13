import logging
import tarfile
import pandas as pd
from pathlib import Path
from typing import Optional, Iterator, Generator
from PIL import Image
from src.utils import setup_logger, Config
from .tar_processor import TarProcessor
from contextlib import contextmanager
from tempfile import TemporaryDirectory
import shutil
from dataclasses import dataclass

logger = setup_logger(__name__)

@dataclass
class BatchDatasetPreparator:
    """Prepares batches of images from tar files for embedding generation."""
    
    output_dir: Path
    config_path: Optional[Path] = None
    temp_dir: Optional[Path] = None
    
    def __post_init__(self) -> None:
        self.config = Config(self.config_path or Path("configs/data_processing.yaml"))
        self.output_dir = Path(self.output_dir)
        self.temp_dir = Path(self.temp_dir) if self.temp_dir else Path(self.config.get("paths.temp_dir"))
        self.valid_extensions = tuple(self.config.get("dataset.valid_extensions"))
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def verify_image(self, path: Path) -> bool:
        """Check if image file is valid."""
        try:
            with Image.open(path) as img:
                img.verify()
            return True
        except Exception as e:
            logger.warning(f"Corrupted image {path}: {e}")
            return False

    @contextmanager
    def _temp_extraction(self) -> Generator[Path, None, None]:
        """Context manager for temporary extraction directory."""
        with TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
            
    def process_tar_batch(
        self,
        start_part: int,
        end_part: int,
        dataset_name: Optional[str] = None
    ) -> Optional[Path]:
        """Process a batch of tar files and create corresponding CSV.
        
        Args:
            start_part: Starting tar file number
            end_part: Ending tar file number
            dataset_name: Optional dataset name override
        """
        # Use config value if no override provided
        dataset_name = dataset_name or self.config.get("dataset.name")
        
        # Download tars (reusing your TarProcessor)
        processor = TarProcessor(dataset_name)
        processor.download_tar_range(start_part, end_part)
        
        # Process the downloaded tars
        valid_images = []
        batch_name = f"batch_{start_part}_{end_part}"
        batch_dir = self.output_dir / batch_name
        batch_dir.mkdir(exist_ok=True)
        
        cache_dir = Path(self.config.get("paths.cache_dir")).expanduser()
        
        with self._temp_extraction() as temp_dir:
            for tar_path in cache_dir.glob("*"):
                if not tar_path.is_file():
                    continue
                
                try:
                    with tarfile.open(tar_path, "r") as tar:
                        # Extract to temp
                        tar.extractall(path=temp_dir)
                        
                        # Process each image
                        for img_path in temp_dir.glob("**/*"):
                            if (img_path.suffix.lower() in self.valid_extensions and 
                                self.verify_image(img_path)):
                                
                                # Create relative path for CSV
                                dest_path = batch_dir / img_path.relative_to(temp_dir)
                                dest_path.parent.mkdir(parents=True, exist_ok=True)
                                
                                # Move valid image
                                img_path.rename(dest_path)
                                valid_images.append({
                                    'path': str(dest_path.relative_to(self.output_dir)),
                                    'batch_id': batch_name,
                                    'source_tar': tar_path.name
                                })
                
                    # Cleanup tar after processing
                    tar_path.unlink()
                    
                except Exception as e:
                    logger.error(f"Error processing {tar_path}: {e}")
                    continue
            
            # Create and save CSV for this batch
            if valid_images:
                df = pd.DataFrame(valid_images)
                csv_path = batch_dir / f"{batch_name}.csv"
                df.to_csv(csv_path, index=False)
                logger.info(f"Created batch CSV at {csv_path} with {len(df)} valid images")
                return csv_path
            
            return None

    def clean_up(self):
        """Clean up temporary directory."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)