import logging
import tarfile
import pandas as pd
from pathlib import Path
from typing import Optional, Iterator, Generator, List
from PIL import Image
from src.utils import setup_logger, Config
from .tar_processor import TarProcessor
from contextlib import contextmanager
from tempfile import TemporaryDirectory
import shutil
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing

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
            
    def process_single_image(
        self, 
        img_path: Path, 
        temp_dir: Path,
        batch_dir: Path,
        tar_name: str
    ) -> Optional[dict]:
        """Process a single image file."""
        if (img_path.suffix.lower() in self.valid_extensions and 
            self.verify_image(img_path)):
            
            # Create relative path for CSV
            dest_path = batch_dir / img_path.relative_to(temp_dir)
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy instead of rename
            shutil.copy2(img_path, dest_path)
            return {
                'path': str(dest_path.relative_to(self.output_dir)),
                'batch_id': batch_dir.name,
                'source_tar': tar_name
            }
        return None

    def process_single_tar(
        self,
        tar_path: Path,
        temp_dir: Path,
        batch_dir: Path,
        num_workers: int = 4
    ) -> List[dict]:
        """Process a single tar file."""
        valid_images = []
        
        try:
            extract_dir = temp_dir / tar_path.stem
            extract_dir.mkdir(exist_ok=True)
            
            with tarfile.open(tar_path, "r") as tar:
                tar.extractall(path=extract_dir)
            
            # Process images in parallel
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = []
                for img_path in extract_dir.glob("**/*"):
                    futures.append(
                        executor.submit(
                            self.process_single_image,
                            img_path,
                            extract_dir,
                            batch_dir,
                            tar_path.name
                        )
                    )
                
                for future in futures:
                    result = future.result()
                    if result:
                        valid_images.append(result)
            
            # Cleanup extracted files
            shutil.rmtree(extract_dir)
            tar_path.unlink()
            
        except Exception as e:
            logger.error(f"Error processing {tar_path}: {e}")
        
        return valid_images

    def process_tar_batch(
        self,
        start_part: int,
        end_part: int,
        dataset_name: Optional[str] = None,
        num_workers: int = min(multiprocessing.cpu_count(), 4)
    ) -> Optional[Path]:
        """Process a batch of tar files and create corresponding CSV."""
        dataset_name = dataset_name or self.config.get("dataset.name")
        
        logger.info(f"Processing batch {start_part} to {end_part} from {dataset_name}")
        logger.info(f"Using {num_workers} workers")
        
        # Download tars
        processor = TarProcessor(dataset_name)
        processor.download_tar_range(start_part, end_part)
        
        # Process the downloaded tars
        batch_name = f"batch_{start_part}_{end_part}"
        batch_dir = self.output_dir / batch_name
        batch_dir.mkdir(exist_ok=True)
        
        valid_images = []
        with self._temp_extraction() as temp_dir:
            tar_files = list(self.temp_dir.glob("dataset_part_*.tar"))
            
            # Process tars in parallel
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = []
                for tar_path in tar_files:
                    futures.append(
                        executor.submit(
                            self.process_single_tar,
                            tar_path,
                            temp_dir,
                            batch_dir,
                            num_workers
                        )
                    )
                
                # Collect results
                for future in futures:
                    valid_images.extend(future.result())
        
        # Create and save CSV
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