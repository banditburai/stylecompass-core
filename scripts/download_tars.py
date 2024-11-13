import click
from pathlib import Path
from src.utils import setup_logger, Config
from src.data.preprocessing import TarProcessor
from src.utils import BatchHelper
logger = setup_logger(__name__)

@click.command()
@click.option('--config', type=click.Path(exists=True), 
              default='configs/data_processing.yaml',
              help='Path to config file')
@click.option('--next-batch', is_flag=True, help='Use next batch range')
def main(config: str, next_batch: bool):
    """Download tar files from HuggingFace dataset."""
    config_path = Path(config)
    config_obj = Config(config_path)
    batch_helper = BatchHelper(config_path)
    
    if next_batch:
        start_part, end_part = batch_helper.next_batch()
    else:
        start_part, end_part = batch_helper.get_current_batch()
    
    logger.info(f"Processing batch {start_part} to {end_part}")
    
    dataset_name = config_obj.get("dataset.name")
    processor = TarProcessor(dataset_name)
    processor.download_tar_range(start_part, end_part)

if __name__ == "__main__":
    main()
