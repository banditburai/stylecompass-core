import torch
import torchvision.transforms as T
from typing import Tuple, Union, List
from src.utils import setup_logger

logger = setup_logger(__name__)

class StyleTransform:
    """Standard transformations for style-based models."""
    
    def __init__(
        self,
        image_size: Union[int, Tuple[int, int]] = 224,
        mean: List[float] = [0.485, 0.456, 0.406],  # ImageNet stats
        std: List[float] = [0.229, 0.224, 0.225],
        augment: bool = True
    ):
        """Initialize transforms.
        
        Args:
            image_size: Target image size (single int for square or tuple)
            mean: Normalization mean values
            std: Normalization standard deviation values
            augment: Whether to apply data augmentation
        """
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
            
        # Basic transforms
        transforms = [
            T.Resize(image_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ]
        
        # Add augmentations for training
        if augment:
            transforms = [
                T.RandomResizedCrop(image_size),
                T.RandomHorizontalFlip(),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                *transforms
            ]
        
        self.transform = T.Compose(transforms)
        logger.info(f"Created transforms: {'with' if augment else 'without'} augmentation")
    
    def __call__(self, img):
        """Apply transforms to image."""
        return self.transform(img)


def create_transforms(
    train: bool = True,
    image_size: int = 224,
) -> StyleTransform:
    """Factory function to create appropriate transforms.
    
    Args:
        train: Whether to include training augmentations
        image_size: Target image size
        
    Returns:
        StyleTransform instance
    """
    return StyleTransform(
        image_size=image_size,
        augment=train
    )