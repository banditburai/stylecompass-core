import click
from pathlib import Path
from src.data.preprocessing import BatchDatasetPreparator
from src.utils import setup_logger, Config
from typing import Optional

logger = setup_logger(__name__)

@click.command()
@click.option('--config', type=click.Path(exists=True), 
              default='configs/data_processing.yaml',
              help='Path to config file')
@click.option('--start-part', type=int, required=True)
@click.option('--end-part', type=int, required=True)
@click.option('--output-dir', type=click.Path())
def main(config: str, start_part: int, end_part: int, output_dir: Optional[str]):
    """Process a batch of images from tar files."""
    config = Config(Path(config))
    
    # Use config with CLI override
    output_dir = output_dir or config.get("paths.output_dir")
    
    preparator = BatchDatasetPreparator(
        output_dir=Path(output_dir),
        config_path=Path(config)
    )
    
    try:
        csv_path = preparator.process_tar_batch(
            start_part=start_part,
            end_part=end_part,
            dataset_name=config.get("dataset.name")
        )
        
        if csv_path:
            logger.info(f"Successfully processed batch {start_part}-{end_part}")
            logger.info(f"CSV created at: {csv_path}")
    finally:
        preparator.clean_up()

if __name__ == '__main__':
    main()