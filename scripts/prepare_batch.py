import click
from pathlib import Path
from src.data.preprocessing import BatchDatasetPreparator
from src.utils import setup_logger, Config
from typing import Optional
from src.utils import BatchHelper

logger = setup_logger(__name__)

@click.command()
@click.option('--config', type=click.Path(exists=True), 
              default='configs/data_processing.yaml',
              help='Path to config file')
@click.option('--next-batch', is_flag=True, help='Use next batch range')
@click.option('--output-dir', type=click.Path())
def main(config: str, next_batch: bool, output_dir: Optional[str]):
    """Process a batch of images from tar files."""
    config_path = Path(config)
    config_obj = Config(config_path)
    batch_helper = BatchHelper(config_path)
    
    if next_batch:
        start_part, end_part = batch_helper.next_batch()
    else:
        start_part, end_part = batch_helper.get_current_batch()
    
    logger.info(f"Processing batch {start_part} to {end_part}")
    
    output_dir = output_dir or config_obj.get("paths.output_dir")
    preparator = BatchDatasetPreparator(
        output_dir=Path(output_dir),
        config_path=config_path
    )
    
    try:
        csv_path = preparator.process_tar_batch(
            start_part=start_part,
            end_part=end_part,
            dataset_name=config_obj.get("dataset.name")
        )
        
        if csv_path:
            logger.info(f"Successfully processed batch {start_part}-{end_part}")
            logger.info(f"CSV created at: {csv_path}")
    finally:
        preparator.clean_up()

if __name__ == '__main__':
    main()