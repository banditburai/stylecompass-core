from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
from torch.utils.data import Dataset

class BatchImageDataset(Dataset):  # Inherit from torch.utils.data.Dataset
    """Dataset for loading processed image batches with metadata."""
    
    def __init__(self, root_dir: Path, batch_csv: Path, transform=None):
        """Initialize dataset."""
        self.root_dir = root_dir
        self.transform = transform
        
        # Load batch metadata
        self.metadata = pd.read_csv(batch_csv)
        self.image_files = [
            root_dir / path for path in self.metadata['path']
        ]
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get image and metadata."""
        # Load and transform image
        image_path = self.image_files[idx]
        image = self.load_image(image_path)
        
        if self.transform:
            image = self.transform(image)
        
        # Add metadata
        return {
            'image': image,
            'batch_id': self.metadata.iloc[idx]['batch_id'],
            'source_tar': self.metadata.iloc[idx]['source_tar']
        }
        
    def __len__(self) -> int:
        return len(self.image_files)