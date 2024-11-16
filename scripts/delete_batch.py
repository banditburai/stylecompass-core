import click
import shutil
from pathlib import Path
from src.utils import Config, setup_logger
import os
import stat

log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / "delete_batch.log"
logger = setup_logger(__name__, log_file=log_file)

def remove_readonly(func, path, _):
    """Clear the readonly bit and reattempt the removal"""
    os.chmod(path, stat.S_IWRITE)
    func(path)

@click.command()
@click.option('--config', type=click.Path(exists=True), 
              default='configs/data_processing.yaml',
              help='Path to config file')
@click.option('--batch-name', type=str, required=False,
              help='Name of the batch to clean (e.g., batch_1_100). Defaults to current batch from config.')
def main(config: str, batch_name: str | None):
    """Clean up temporary files from the current or specified batch"""
    config_path = Path(config)
    config_obj = Config(config_path)
    
    if batch_name is None:
        # Get current batch from config
        start = config_obj.get("batch.current_start")
        end = config_obj.get("batch.current_end")
        batch_name = f"batch_{start}_{end}"
        logger.info(f"Using current batch from config: {batch_name}")
    
    batch_dir = Path(config_obj.get("paths.output_dir")) / batch_name
    
    # Paths to clean (convert to Path objects)
    temp_paths = [
        batch_dir / "dataset_temp",
        batch_dir / "thumbnails",
        Path("temp_lance_dataset")  # Convert string to Path
    ]
    
    for path in temp_paths:
        if path.exists():
            logger.info(f"Removing {path}")
            try:
                # First try normal removal
                shutil.rmtree(path)
            except Exception as e:
                logger.warning(f"First attempt to remove {path} failed: {e}")
                try:
                    # Try again with readonly handler
                    shutil.rmtree(str(path), onerror=remove_readonly)  # Convert Path to string for rmtree
                except Exception as e:
                    logger.error(f"Failed to remove {path}: {e}")
        else:
            logger.info(f"Path {path} does not exist, skipping")
    
    logger.info("Cleanup complete")

if __name__ == '__main__':
    main()