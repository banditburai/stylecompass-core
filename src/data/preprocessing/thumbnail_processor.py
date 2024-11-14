from pathlib import Path
from typing import Optional
from dataclasses import dataclass
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from src.utils import setup_logger, Config

logger = setup_logger(__name__)

@dataclass
class ThumbnailProcessor:
    config_path: Optional[Path] = None
    
    def __post_init__(self) -> None:
        # Use provided config path or default
        self.config = Config(self.config_path) if self.config_path else None
    
    def create_thumbnail(self, img_path: Path, output_dir: Path, target_size: int = 224) -> bool:
        """Create WebP thumbnail preserving aspect ratio.
        
        Args:
            img_path: Path to input image
            output_dir: Output directory
            target_size: Target size for longer dimension
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Read image
            img = cv2.imread(str(img_path))
            if img is None:
                logger.warning(f"Could not read image: {img_path}")
                return False
                
            # Calculate new dimensions preserving aspect ratio
            h, w = img.shape[:2]
            if h > w:
                new_h = target_size
                new_w = int(w * (target_size / h))
            else:
                new_w = target_size
                new_h = int(h * (target_size / w))
                
            # Resize using CPU
            resized = cv2.resize(img, (new_w, new_h))
            
            # Save as WebP
            output_path = output_dir / f"{img_path.stem}.webp"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), resized, [cv2.IMWRITE_WEBP_QUALITY, 80])
            return True
            
        except Exception as e:
            logger.error(f"Error processing {img_path}: {str(e)}")
            return False

    def process_batch(
        self,
        batch_dir: Path,
        output_dir: Path,
        target_size: int = 224,
        num_workers: int = 4
    ) -> int:
        """Process all images in a batch directory to create thumbnails.
        
        Args:
            batch_dir: Input batch directory
            output_dir: Output directory for thumbnails
            target_size: Target thumbnail size
            num_workers: Number of worker threads
            
        Returns:
            int: Number of successfully processed thumbnails
        """
        # Get all image files
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png']:
            image_files.extend(batch_dir.rglob(f"*{ext}"))
        
        if not image_files:
            logger.warning(f"No images found in {batch_dir}")
            return 0
            
        # Process in parallel
        successful = 0
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for img_path in image_files:
                # Preserve relative path structure
                rel_output_dir = output_dir / img_path.parent.relative_to(batch_dir)
                futures.append(
                    executor.submit(
                        self.create_thumbnail, 
                        img_path, 
                        rel_output_dir, 
                        target_size
                    )
                )
            
            # Show progress
            for future in tqdm(futures, total=len(image_files), desc="Creating thumbnails"):
                if future.result():
                    successful += 1
                    
        return successful