from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
from .base import ImageDataset

class BatchImageDataset(ImageDataset):
    """Dataset for loading processed image batches with metadata."""
    
    def __init__(self, root_dir: Path, batch_csv: Path, transform=None):
        """Initialize dataset.
        
        Args:
            root_dir: Root directory containing batch folders
            batch_csv: Path to batch CSV file with metadata
            transform: Optional transform to apply to images
        """
        super().__init__(root_dir, transform)
        
        # Load batch metadata
        self.metadata = pd.read_csv(batch_csv)
        self.image_files = [
            root_dir / path for path in self.metadata['path']
        ]
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get image and metadata.
        
        Returns:
            Dict containing:
                image: Transformed image tensor
                batch_id: Batch identifier
                source_tar: Original tar file
        """
        # Get image using parent class
        image = super().__getitem__(idx)
        
        # Add metadata
        return {
            'image': image,
            'batch_id': self.metadata.iloc[idx]['batch_id'],
            'source_tar': self.metadata.iloc[idx]['source_tar']
        }