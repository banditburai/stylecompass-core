import click
from pathlib import Path
from src.utils import setup_logger, Config, BatchHelper
from src.data.preprocessing import ThumbnailProcessor
from typing import Optional

logger = setup_logger(__name__)

@click.command()
@click.option('--config', type=click.Path(exists=True), 
              default='configs/data_processing.yaml',
              help='Path to config file')
@click.option('--next-batch', is_flag=True, help='Use next batch range')
@click.option('--output-dir', type=click.Path())
@click.option('--target-size', type=int, default=224, help='Target thumbnail size')
@click.option('--num-workers', type=int, default=4, help='Number of worker threads')
def main(config: str, next_batch: bool, output_dir: Optional[str], 
         target_size: int, num_workers: int):
    """Create thumbnails for a batch of processed images."""
    config_path = Path(config)
    config_obj = Config(config_path)
    batch_helper = BatchHelper(config_path)
    
    if next_batch:
        start_part, end_part = batch_helper.next_batch()
    else:
        start_part, end_part = batch_helper.get_current_batch()
    
    logger.info(f"Processing thumbnails for batch {start_part} to {end_part}")
    
    # Get directories
    batch_name = f"batch_{start_part}_{end_part}"
    input_dir = Path(config_obj.get("paths.output_dir")) / batch_name
    output_dir = Path(output_dir or config_obj.get("paths.thumbnail_dir")) / batch_name
    
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return
        
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Creating thumbnails in: {output_dir}")
    
    # Process thumbnails
    processor = ThumbnailProcessor(config_path=config_path)
    successful = processor.process_batch(
        input_dir, 
        output_dir,
        target_size=target_size,
        num_workers=num_workers
    )
    
    logger.info(f"Successfully created {successful} thumbnails")

if __name__ == '__main__':
    main()