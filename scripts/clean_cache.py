from functools import partial
from pathlib import Path
from typing import Sequence, Tuple, Iterator

import click
from huggingface_hub import scan_cache_dir
from huggingface_hub.utils import HFCacheInfo
from src.utils.logging import setup_logger
from src.utils.config import Config

# Constants
MIN_SIZE_MB = 100
DEFAULT_CONFIG = 'configs/cache_management.yaml'

# Setup logging
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
log_file = LOG_DIR / f"{Path(__file__).stem}.log"
logger = setup_logger(__name__, log_file=log_file)


def get_dir_size(path: Path) -> int:
    """Calculate total directory size in bytes."""
    try:
        return sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
    except (PermissionError, FileNotFoundError):
        logger.warning(f"Permission denied or not found: {path}")
        return 0


def analyze_directory(path: Path, min_size_mb: int = MIN_SIZE_MB) -> list[tuple[Path, int]]:
    """Analyze directory and return sizes of subdirectories larger than min_size_mb."""
    min_size_bytes = min_size_mb * 1024 * 1024
    results = []

    try:
        # Handle both files and directories
        for item in path.iterdir():
            size = item.stat().st_size if item.is_file() else get_dir_size(item)
            if size > min_size_bytes:
                results.append((item, size))
    except PermissionError:
        logger.warning(f"Permission denied for {path}")

    return sorted(results, key=lambda x: x[1], reverse=True)


def format_size(size_bytes: int) -> str:
    """Format size in bytes to human-readable string."""
    return f"{size_bytes / (1024**3):.2f}GB"


def get_revision_hashes(cache_info: HFCacheInfo) -> list[str]:
    """Extract revision hashes from cache info."""
    return [
        revision.commit_hash
        for repo in cache_info.repos
        for revision in repo.revisions
    ]


def print_analysis_results(results: Sequence[Tuple[Path, int]]) -> None:
    """Print analysis results in a formatted way."""
    if not results:
        logger.info("No items found above minimum size threshold.")
        return

    total_size = sum(size for _, size in results)
    logger.info(f"\nTotal analyzed size: {format_size(total_size)}")
    logger.info("\nLargest directories/files:")
    
    for path, size in results:
        logger.info(f"{format_size(size):>10} - {path}")


@click.command()
@click.option(
    '--config', 
    type=click.Path(exists=True, path_type=Path),
    default=DEFAULT_CONFIG,
    help='Path to config file'
)
@click.option(
    '--analyze', 
    is_flag=True, 
    help='Analyze disk usage instead of cleaning cache'
)
@click.option(
    '--min-size', 
    default=MIN_SIZE_MB, 
    help='Minimum size in MB to report'
)
@click.option(
    '--force', 
    is_flag=True, 
    help='Skip deletion confirmation'
)
def main(config: Path, analyze: bool, min_size: int, force: bool) -> None:
    """Analyze disk usage or clean cache."""
    config_obj = Config(config)
    
    if analyze:
        logger.info("Analyzing disk usage...")
        project_root = Path(__file__).parent.parent
        results = analyze_directory(project_root, min_size)
        print_analysis_results(results)
        return

    # Cache cleaning logic
    logger.info("Scanning cache...")
    cache_info: HFCacheInfo = scan_cache_dir()
    logger.info(f"Current cache size: {format_size(cache_info.size_on_disk)}")
    
    revision_hashes = get_revision_hashes(cache_info)
    if not revision_hashes:
        logger.info("No revisions found in cache.")
        return
    
    strategy = cache_info.delete_revisions(*revision_hashes)
    logger.info(f"Will free {strategy.expected_freed_size_str}")
    
    if not force and config_obj.get("huggingface.cache.confirm_deletion", True):
        if input("Do you want to proceed with deletion? (y/N): ").lower() != 'y':
            logger.info("Deletion cancelled.")
            return
    
    strategy.execute()
    logger.info("Cache deletion complete!")


if __name__ == "__main__":
    main() 