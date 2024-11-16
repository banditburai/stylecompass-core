import click
from pathlib import Path
import shutil
import os
from huggingface_hub import scan_cache_dir, HfApi
from huggingface_hub.utils import HFCacheInfo
from huggingface_hub import constants
from src.utils.logging import setup_logger
from src.utils.config import Config

script_name = Path(__file__).stem
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"{script_name}.log"
logger = setup_logger(__name__, log_file=log_file)

def get_dir_size(path):
    """Calculate total directory size in bytes"""
    try:
        return sum(f.stat().st_size for f in Path(path).rglob('*') if f.is_file())
    except (PermissionError, FileNotFoundError):
        return 0

def analyze_directory(path, min_size_mb=100):
    """Analyze directory and return sizes of subdirectories"""
    results = []
    try:
        for item in os.scandir(path):
            if item.is_dir():
                size = get_dir_size(item.path)
                if size > min_size_mb * 1024 * 1024:  # Convert MB to bytes
                    results.append((item.path, size))
            elif item.is_file() and item.stat().st_size > min_size_mb * 1024 * 1024:
                results.append((item.path, item.stat().st_size))
    except PermissionError:
        logger.warning(f"Permission denied for {path}")
    return sorted(results, key=lambda x: x[1], reverse=True)

@click.command()
@click.option('--config', type=click.Path(exists=True), 
              default='configs/cache_management.yaml',
              help='Path to config file')
@click.option('--analyze', is_flag=True, help='Analyze disk usage instead of cleaning cache')
@click.option('--min-size', default=100, help='Minimum size in MB to report')
def main(config: str, analyze: bool, min_size: int):
    """Analyze disk usage or clean cache"""
    if analyze:
        logger.info("Analyzing disk usage...")
        project_root = Path(__file__).parent.parent
        results = analyze_directory(project_root, min_size)
        
        total_size = sum(size for _, size in results)
        logger.info(f"\nTotal analyzed size: {total_size / (1024**3):.2f}GB")
        logger.info("\nLargest directories/files:")
        for path, size in results:
            logger.info(f"{size / (1024**3):6.2f}GB - {path}")
        return

    # Original cache cleaning code continues here...
    config = Config(Path(config))
    # Rest of your existing code...

if __name__ == "__main__":
    main() 