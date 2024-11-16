import click
from pathlib import Path
import multiprocessing
from src.data.preprocessing import BatchDatasetPreparator
from src.utils import setup_logger, Config
from typing import Optional
from src.utils import BatchHelper

script_name = Path(__file__).stem
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"{script_name}.log"  # where script_name matches the script's purpose
logger = setup_logger(__name__, log_file=log_file)

@click.command()
@click.option('--config', type=click.Path(exists=True), 
              default='configs/data_processing.yaml',
              help='Path to config file')
@click.option('--next-batch', is_flag=True, help='Use next batch range')
@click.option('--output-dir', type=click.Path())
@click.option('--num-workers', type=int, 
              default=min(multiprocessing.cpu_count(), 4),
              help='Number of worker processes')
def main(config: str, next_batch: bool, output_dir: Optional[str], num_workers: int):
    """Process a batch of images from tar files."""
    config_path = Path(config)
    config_obj = Config(config_path)
    batch_helper = BatchHelper(config_path)
    
    if next_batch:
        start_part, end_part = batch_helper.next_batch()
    else:
        start_part, end_part = batch_helper.get_current_batch()
    
    logger.info(f"Processing batch {start_part} to {end_part}")
    logger.info(f"Using {num_workers} workers")
    
    output_dir_path = Path(output_dir) if output_dir else Path(config_obj.get("paths.output_dir"))
    
    preparator = BatchDatasetPreparator(
        output_dir=output_dir_path,
        config_path=config_path
    )
    
    try:
        csv_path = preparator.process_tar_batch(
            start_part=start_part,
            end_part=end_part,
            dataset_name=config_obj.get("dataset.name"),
            num_workers=num_workers
        )
        
        if csv_path:
            logger.info(f"Successfully processed batch {start_part}-{end_part}")
            logger.info(f"CSV created at: {csv_path}")
    finally:
        preparator.clean_up()

if __name__ == '__main__':
    main()