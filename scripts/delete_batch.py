from pathlib import Path
from typing import Optional
import shutil
import os
import stat

import click

from src.utils import Config, setup_logger

# Constants
DEFAULT_CONFIG = Path('configs/data_processing.yaml')
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
log_file = LOG_DIR / "delete_batch.log"
logger = setup_logger(__name__, log_file=log_file)


def handle_readonly(exc: OSError) -> None:
    """Clear the readonly bit and reattempt the removal."""
    path = Path(exc.filename)
    os.chmod(path, stat.S_IWRITE)
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def safe_remove_tree(path: Path) -> None:
    """Safely remove a directory tree, handling readonly files."""
    if not path.exists():
        logger.info(f"Path {path} does not exist, skipping")
        return

    logger.info(f"Removing {path}")
    try:
        shutil.rmtree(path)
    except Exception as e:
        logger.warning(f"First attempt to remove {path} failed: {e}")
        try:
            shutil.rmtree(path, onexc=handle_readonly)
        except Exception as e:
            logger.error(f"Failed to remove {path}: {e}")


def get_batch_name(config: Config, batch_name: Optional[str] = None) -> str:
    """Get batch name from config or use provided name."""
    if batch_name is not None:
        return batch_name
    
    start = config.get("batch.current_start")
    end = config.get("batch.current_end")
    batch_name = f"batch_{start}_{end}"
    logger.info(f"Using current batch from config: {batch_name}")
    return batch_name


@click.command()
@click.option(
    '--config',
    type=click.Path(exists=True, path_type=Path),
    default=DEFAULT_CONFIG,
    help='Path to config file'
)
@click.option(
    '--batch-name',
    type=str,
    help='Name of the batch to clean (e.g., batch_1_100). Defaults to current batch from config.'
)
def main(config: Path, batch_name: Optional[str]) -> None:
    """Clean up temporary files from the current or specified batch."""
    config_obj = Config(config)
    batch = get_batch_name(config_obj, batch_name)
    batch_dir = Path(config_obj.get("paths.output_dir")) / batch
    
    # Paths to clean
    temp_paths = [
        batch_dir / "dataset_temp",
        # batch_dir / "thumbnails", 
    ]
    
    for path in temp_paths:
        safe_remove_tree(path)
    
    logger.info("Cleanup complete")


if __name__ == '__main__':
    main()