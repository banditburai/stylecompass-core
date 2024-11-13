from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Optional, List, Dict
import pandas as pd
import numpy as np

class WikiArtDataset(Dataset):
    """WikiArt dataset loader"""
    
    def __init__(
        self,
        root_dir: Path,
        split: str = 'database',
        transform: Optional[transforms.Compose] = None,
        annotation_file: str = 'wikiart.csv',
        max_size: Optional[int] = None
    ):
        """
        Args:
            root_dir: Root directory containing images and annotations
            split: Dataset split ('query' or 'database')
            transform: Optional transform to be applied to images
            annotation_file: Name of the annotation CSV file
            max_size: Optional limit on dataset size
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.split = split
        
        # Load annotations
        annotation_path = self.root_dir / annotation_file
        assert annotation_path.exists(), f"Annotation file not found: {annotation_path}"
        
        # Load paths using pandas
        annotations = pd.read_csv(annotation_path)
        
        # Filter by split if specified
        if split in ['query', 'database']:
            annotations = annotations[annotations['split'] == split]
            
        self.image_paths = annotations['path'].tolist()
        self.image_names = [Path(p).name for p in self.image_paths]
        
        # Optional size limit
        if max_size is not None:
            indices = np.random.choice(
                len(self.image_names), 
                size=min(max_size, len(self.image_names)), 
                replace=False
            )
            self.image_paths = [self.image_paths[i] for i in indices]
            self.image_names = [self.image_names[i] for i in indices]

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, idx: int):
        """
        Returns:
            image: Transformed image tensor
            idx: Index for tracking
        """
        # Load and convert image
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        
        # Apply transforms if specified
        if self.transform:
            image = self.transform(image)
            
        return image, idx
    
    @property
    def filenames(self) -> List[str]:
        """Get list of image filenames"""
        return self.image_names

def create_wikiart_datasets(
    root_dir: Path,
    transform: transforms.Compose,
    max_size: Optional[int] = None
) -> Dict[str, Dataset]:
    """Create WikiArt datasets for query and database splits"""
    return {
        'query': WikiArtDataset(
            root_dir=root_dir,
            split='query',
            transform=transform,
            max_size=max_size
        ),
        'database': WikiArtDataset(
            root_dir=root_dir,
            split='database', 
            transform=transform,
            max_size=max_size
        )
    }